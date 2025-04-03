#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import json
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter
import os, sys
from utils.trajectory import get_pcdGenPoses
from thirdparty.StyleCineGAN.main_jih import VideoGenerator
import cv2

from utils.loss_utils import l1_loss
from torch.optim.lr_scheduler import StepLR,ExponentialLR
from thirdparty.cinemagraphy.demo import eulerian_estimation
from helpmotion import SceneFlow, flow2img, save_image

class MotionOptimization:
    def __init__(self, src_img, pcdgenpath='lookaround'):
        self.d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
        self.camera_params(src_img)
        self.render_poses = get_pcdGenPoses(pcdgenpath)      # default len(pcdgenpath) = 14
        self.src_depth = self.d(src_img)         
        self.center_depth = np.mean(self.src_depth[self.H//2-10:self.H//2+10, self.W//2-10:self.W//2+10])
        self.internel_render_poses = get_pcdGenPoses('hemisphere', {'center_depth': self.center_depth})       # internel_render_poses.shape = (5, 3, 4)


    def d(self, im):
        return self.d_model.infer_pil(im)

    def camera_params(self, src_img):
        w_in, h_in = src_img.size

        self.H = h_in
        self.W = w_in
        self.focal_length = 5.8269e+02
        self.aspect_ratio = self.W / self.H
        self.focal = (self.focal_length * self.aspect_ratio, self.focal_length)

        self.fov = (2*np.arctan(self.W / (2*self.focal[0])), 2*np.arctan(self.H / (2*self.focal[1])))

        self.K = np.array([
            [self.focal[0], 0., self.W/2],
            [0., self.focal[1], self.H/2],
            [0.,            0.,       1.],
        ]).astype(np.float32)


    def optimize_motion(self, rgb_cond, src_mask, train_data, non_frame_idx, train_iteration, pcdgenpath='lookaround'):
        coord = train_data['pcd_points']
        R0, T0 = self.render_poses[0,:3,:3], self.render_poses[0,:3,3:4]
        pts_coord_world = coord.copy()
        # pts_coord_world_2 = coord.copy()

        yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])

        # estimated flow to pixel coord 
        GT_list = []
        frame_idx = 0
        iterable_align = range(len(self.render_poses))
        for i in iterable_align:
            for j in range(len(self.internel_render_poses)):
                idx = i * len(self.internel_render_poses) + j
                print(f'{idx+1} / {len(self.render_poses)*len(self.internel_render_poses)}')
                if idx in non_frame_idx:
                    continue
                frame = train_data["frames"][frame_idx]
                GT_flow = frame["T2C_flow"][0]     #torch.Size([1, 2, 512, 512])
                frame_idx +=1

                x, y = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32), indexing='xy') # pixels
                grid = np.stack((x,y), axis=-1).reshape(-1,2)

                ### Transform world to pixel
                Rw2i = self.render_poses[i,:3,:3]
                Tw2i = self.render_poses[i,:3,3:4]
                Ri2j = self.internel_render_poses[j,:3,:3]
                Ti2j = self.internel_render_poses[j,:3,3:4]

                Rw2j = np.matmul(Ri2j, Rw2i)
                Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j

                # Transfrom cam2 to world + change sign of yz axis
                Rj2w = np.matmul(yz_reverse, Rw2j).T
                Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
                Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

                pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
                pixel_coord_camj = np.matmul(self.K, pts_coord_camj)
                
                valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0, 
                                                                            pixel_coord_camj[0]/pixel_coord_camj[2]>=0, 
                                                                            pixel_coord_camj[0]/pixel_coord_camj[2]<=self.W-1, 
                                                                            pixel_coord_camj[1]/pixel_coord_camj[2]>=0, 
                                                                            pixel_coord_camj[1]/pixel_coord_camj[2]<=self.H-1)))[0]
                if len(valid_idxj) == 0:
                    continue
                pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]

                GT_flow = GT_flow.permute(2, 3, 1, 0).squeeze().reshape(self.H*self.W,2)
                GT_flow_numpy = GT_flow.cpu().clone().numpy()

                GT_flow_pixel_coord_camj = interp_grid(grid, GT_flow_numpy, pixel_coord_camj.transpose(1,0), method='linear', fill_value=0)            
                GT_flow_pixel_coord_camj_tensor = torch.tensor(GT_flow_pixel_coord_camj.T)
                GT_list.append(GT_flow_pixel_coord_camj_tensor)
        

        model = SceneFlow(coord)
        tensor_pts_coord_world = torch.from_numpy(pts_coord_world.copy())

        flow_optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        scene_flow = model()
        scheduler = ExponentialLR(flow_optimizer, gamma=0.97)


        """TRAIN SceneFlow"""
        for epoch in tqdm(range(train_iteration)):
            avg_loss = 0
            GT_num = 0
            for i in iterable_align:
                for j in range(len(self.internel_render_poses)):
                    idx = i * len(self.internel_render_poses) + j
                    if idx not in non_frame_idx:
                        GT_flow_pixel_coord_camj_tensor = GT_list[GT_num]
                        GT_num += 1
                    else:
                        continue
                
                    ### Transform world to pixel
                    Rw2i = self.render_poses[i,:3,:3]
                    Tw2i = self.render_poses[i,:3,3:4]
                    Ri2j = self.internel_render_poses[j,:3,:3]
                    Ti2j = self.internel_render_poses[j,:3,3:4]

                    Rw2j = np.matmul(Ri2j, Rw2i)
                    Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j

                    # Transfrom cam2 to world + change sign of yz axis
                    Rj2w = np.matmul(yz_reverse, Rw2j).T
                    Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
                    Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
                    Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

                    pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
                    pixel_coord_camj = np.matmul(self.K, pts_coord_camj)

                    Rw2j_tensor = torch.tensor(Rw2j,dtype=torch.float32)
                    Tw2j_tensor = torch.tensor(Tw2j,dtype=torch.float32)
                    K_tensor = torch.tensor(self.K,dtype=torch.float32)

                    scene_flow = model()
                    flow_pts_coord_world = tensor_pts_coord_world+scene_flow

                    flow_pts_coord_camj = torch.matmul(Rw2j_tensor,flow_pts_coord_world) + Tw2j_tensor
                    flow_pixel_coord_camj = torch.matmul(K_tensor, flow_pts_coord_camj)
                    valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0, 
                                                                pixel_coord_camj[0]/pixel_coord_camj[2]>=0, 
                                                                pixel_coord_camj[0]/pixel_coord_camj[2]<=self.W-1, 
                                                                pixel_coord_camj[1]/pixel_coord_camj[2]>=0, 
                                                                pixel_coord_camj[1]/pixel_coord_camj[2]<=self.H-1)))[0]
                    if len(valid_idxj) == 0:
                        continue
                    pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]
                    flow_pixel_coord_camj = flow_pixel_coord_camj[:2, valid_idxj]/flow_pixel_coord_camj[-1:, valid_idxj]
                    
                    pixel_coord_camj_tensor = torch.tensor(pixel_coord_camj,dtype=torch.float32)
                    new_flow_coord_camj = flow_pixel_coord_camj - pixel_coord_camj_tensor         

                    # loss
                    Ll1 = l1_loss(new_flow_coord_camj, GT_flow_pixel_coord_camj_tensor)
                    avg_loss +=  Ll1
                    loss = avg_loss/(idx+1)
                    if GT_num == len(GT_list):
                        flow_optimizer.zero_grad()
                        loss.backward()
                        flow_optimizer.step()

                    # 두 world coordinate을 image 좌표계로 변환 후 차이를 구함(픽셀 레벨의 차이)
                    if epoch == (train_iteration-1):
                        new_flow_coord_camj_2 = new_flow_coord_camj.detach().cpu().numpy()
                        final_flow = interp_grid(pixel_coord_camj.transpose(1,0), new_flow_coord_camj_2.T, grid, method='linear', fill_value=0).reshape(self.H,self.W,2)
                        flow_world_tensor = torch.tensor(np.transpose(final_flow,(2,0,1))).unsqueeze(0)
                        train_data['frames'][idx]['our_flow'].append(flow_world_tensor)


            scheduler.step()
            print('Epoch:', epoch, 'LR:', scheduler.get_lr(), 'loss:', loss)

        scene_flow = model()
        return train_data, scene_flow 

    

    def render_PCD(self, src_img, src_mask, hints):

        src_mask = Image.fromarray(np.repeat(np.array(src_mask)[..., np.newaxis], 3, axis=-1).astype(np.uint8))
        x, y = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32), indexing='xy') # pixels
        edgeN = 2
        edgemask = np.ones((self.H-2*edgeN, self.W-2*edgeN))
        edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))

        ### initialize 
        R0, T0 = self.render_poses[0,:3,:3], self.render_poses[0,:3,3:4]
        pts_coord_cam = np.matmul(np.linalg.inv(self.K), np.stack((x*self.src_depth, y*self.src_depth, 1*self.src_depth), axis=0).reshape(3,-1))     # pixel to camera coord
        new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## c
        new_pts_colors2 = (np.array(src_img).reshape(-1,3).astype(np.float32)/255.) ## new_pts_colors2
        mask_pts_colors2 = (np.array(src_mask).reshape(-1,3).astype(np.float32)/255.) ## mask_pts_colors

        pts_coord_world, pts_colors, mask_pts_colors= new_pts_coord_world2.copy(), new_pts_colors2.copy(), mask_pts_colors2.copy()

        ### hint (pixel -> world)
        hint_start_world_coord = []
        for h in range(len(hints[0])):
            h_x = hints[0][h]
            h_y = hints[1][h]
            depth_x_y = self.src_depth[h_y, h_x]
            pixel_coords = np.array([[h_y], [h_x], [1]]) * depth_x_y
            cam_coords = np.linalg.inv(self.K).dot(pixel_coords)
            hint_pts_coord_world2 = (np.linalg.inv(R0).dot(cam_coords) - np.linalg.inv(R0).dot(T0)).astype(np.float32)
            hint_start_world_coord.append(hint_pts_coord_world2)

        hint_end_world_coord = []
        for l in range(len(hints[0])):
            h_x = hints[2][l]
            h_y = hints[3][l]
            depth_x_y = self.src_depth[h_y, h_x]
            pixel_coords = np.array([[h_y], [h_x], [1]]) * depth_x_y
            cam_coords = np.linalg.inv(self.K).dot(pixel_coords)
            hint_pts_coord_world2 = (np.linalg.inv(R0).dot(cam_coords) - np.linalg.inv(R0).dot(T0)).astype(np.float32)
            hint_end_world_coord.append(hint_pts_coord_world2)


        yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
        traindata = {
            'camera_angle_x': self.fov[0],
            'camera_angle_y': self.fov[1],
            'W': self.W,
            'H': self.H,
            'pcd_points': pts_coord_world,
            'pcd_colors': pts_colors,
            'pcd_masks': mask_pts_colors,
            'frames': [],
        }

        iterable_align = range(len(self.render_poses))

        none_idx = []
        for i in iterable_align:
            for j in range(len(self.internel_render_poses)):
                idx = i * len(self.internel_render_poses) + j
                print(f'{idx+1} / {len(self.render_poses)*len(self.internel_render_poses)}')

                ### Transform world to pixel
                Rw2i = self.render_poses[i,:3,:3]
                Tw2i = self.render_poses[i,:3,3:4]
                Ri2j = self.internel_render_poses[j,:3,:3]
                Ti2j = self.internel_render_poses[j,:3,3:4]

                Rw2j = np.matmul(Ri2j, Rw2i)
                Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j

                # Transfrom cam2 to world + change sign of yz axis
                Rj2w = np.matmul(yz_reverse, Rw2j).T
                Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
                Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

                pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
                pixel_coord_camj = np.matmul(self.K, pts_coord_camj)

                valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]<=self.W-1, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]>=0, 
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]<=self.H-1)))[0]
                if len(valid_idxj) == 0:
                    none_idx.append(idx)
                    continue
                pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
                pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]
                round_coord_camj = np.round(pixel_coord_camj).astype(np.int32)

                x, y = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32), indexing='xy') # pixels
                grid = np.stack((x,y), axis=-1).reshape(-1,2)

                """ RGB rendering """
                imagej = interp_grid(pixel_coord_camj.transpose(1,0), pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(self.H,self.W,3)
                imagej = edgemask[...,None]*imagej + (1-edgemask[...,None])*np.pad(imagej[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')
                
                depthj = interp_grid(pixel_coord_camj.transpose(1,0), pts_depthsj.T, grid, method='linear', fill_value=0).reshape(self.H,self.W)
                depthj = edgemask*depthj + (1-edgemask)*np.pad(depthj[1:-1,1:-1], ((1,1),(1,1)), mode='edge')
                
                maskj = np.zeros((self.H,self.W), dtype=np.float32)
                maskj[round_coord_camj[1], round_coord_camj[0]] = 1
                maskj = maximum_filter(maskj, size=(9,9))
                imagej = maskj[...,None]*imagej + (1-maskj[...,None])*(-1)

                maskj = minimum_filter((imagej.sum(-1)!=-3)*1, size=(11,11))
                imagej = maskj[...,None]*imagej + (1-maskj[...,None])*0

                """ Mask rendering """
                imagej_2 = interp_grid(pixel_coord_camj.transpose(1,0), mask_pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(self.H,self.W,3)
                imagej_2 = edgemask[...,None]*imagej_2 + (1-edgemask[...,None])*np.pad(imagej_2[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

                maskj_2 = np.zeros((self.H,self.W), dtype=np.float32)
                maskj_2[round_coord_camj[1], round_coord_camj[0]] = 1
                maskj_2 = maximum_filter(maskj_2, size=(9,9))
                imagej_2 = maskj_2[...,None]*imagej_2 + (1-maskj[...,None])*(-1)

                maskj_2 = minimum_filter((imagej_2.sum(-1)!=-3)*1, size=(11,11))
                imagej_2 = maskj_2[...,None]*imagej_2 + (1-maskj_2[...,None])*0
                maskp = imagej_2
                mask = maskp[:,:,0]
                
                final_hint_start_x = []
                final_hint_start_y = []
                
                final_hint_end_x = []
                final_hint_end_y = []
                
                for hint_coord in hint_start_world_coord:
                    pts_coord_camj = Rw2j.dot(hint_coord) + Tw2j
                    pixel_coord_camj = np.matmul(self.K, pts_coord_camj)
                    pixel_coord_camj /= pixel_coord_camj[2]

                    final_hint_start_y.append(pixel_coord_camj[0])
                    final_hint_start_x.append(pixel_coord_camj[1])

                for hint_coord in hint_end_world_coord:
                    pts_coord_camj = Rw2j.dot(hint_coord) + Tw2j
                    pixel_coord_camj = np.matmul(self.K, pts_coord_camj)
                    pixel_coord_camj /= pixel_coord_camj[2]

                    final_hint_end_y.append(pixel_coord_camj[0])
                    final_hint_end_x.append(pixel_coord_camj[1])

                traindata['frames'].append({
                    'image': Image.fromarray(np.round(imagej*255.).astype(np.uint8)), 
                    'transform_matrix': Pc2w.tolist(),
                    'mask': Image.fromarray(np.round(mask*255.).astype(np.uint8)),
                    'final_hint_start_x' : final_hint_start_x,
                    'final_hint_start_y' : final_hint_start_y,
                    'final_hint_end_x' : final_hint_end_x,
                    'final_hint_end_y' : final_hint_end_y,
                    'T2C_flow' : [],
                    'our_flow' : [],
                })

        return traindata, none_idx

    def estimate_flow(self, train_data):
        frames = train_data["frames"]
        for idx, frame in enumerate(frames): 
            flow = eulerian_estimation(args, frame)
            train_data['frames'][idx]['T2C_flow'].append(flow)
        
        return train_data

def read_json(file_path):
    # hints
    hint_y_start = []
    hint_x_start = []
    hint_x_end = []
    hint_y_end = []

    data = json.load(open(file_path))
    for shape in data['shapes']:
        if shape['label'].startswith('hint'):
            start, end = np.array(shape["points"])
            hint_x_start.append(int(start[0]))
            hint_y_start.append(int(start[1]))
            hint_x_end.append(int(end[0]))
            hint_y_end.append(int(end[1]))
    
    return [hint_x_start, hint_y_start, hint_x_end, hint_y_end]

def viz_flow(train_data, viz_dir):
    for idx, frame in enumerate(train_data['frames']):
        our_flow = frame["our_flow"][0]     #torch.Size([1, 2, 512, 512])
        viz_flow = flow2img(our_flow[0])
        our_flow_path = os.path.join(viz_dir, str(idx).zfill(3)+'.png')
        save_image(viz_flow, our_flow_path)


def save_video(frames, output_path, W, H):
    resized_frames = []
    output_path = os.path.join(output_path, 'video')
    os.makedirs(output_path, exist_ok=True)
    for i, frame in enumerate(frames):
        frame = (frame * 255).astype(np.uint8)
        image = Image.fromarray(frame).resize((W, H))
        resized_frames.append(image)
        
        filename = os.path.join(output_path ,f'{str(i).zfill(6)}.png')
        image.save(filename) 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_path, 'sampled_video.mp4')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (image.size[0], image.size[1]))

    for image in resized_frames:
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        video_writer.write(open_cv_image)

    video_writer.release()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    
    parser.add_argument('--input_dir', type=str, help='input folder that contains src images', required=True)
    parser.add_argument('--train_iteration', type=int, default=200)
    parser.add_argument('-c', '--config', type=str, default='thirdparty/cinemagraphy/config.yaml', help='config file path')
    parser.add_argument('--local_rank', type=int, default=0, help='rank for distributed training')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument("--cinema_ckpt", type=str, default='thirdparty/cinemagraphy/ckpts',help='specific weights file to reload')
    
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--no_load_opt", action='store_true', help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true', help='do not load scheduler when reloading')
    args = parser.parse_args(sys.argv[1:])

    MOM_dir = os.path.join(args.input_dir,"MOM")
    os.makedirs(MOM_dir, exist_ok=True)

    # Input Image, Mask & Hints (from Labelme.)
    src_img = Image.open(os.path.join(args.input_dir, 'image.png'))
    src_mask = Image.open(os.path.join(args.input_dir, 'image_json', 'mask.png'))
    hints = read_json(os.path.join(args.input_dir, 'image.json'))

    MOM = MotionOptimization(src_img)
    train_data, none_idx = MOM.render_PCD(src_img, src_mask, hints)
    train_data = MOM.estimate_flow(train_data)  
    train_data, scene_flow = MOM.optimize_motion(src_img, src_mask, train_data, none_idx, args.train_iteration)
    torch.save(train_data, os.path.join(MOM_dir, 'tran_data.pth'))

    frames = VideoGenerator(train_data)
    save_video(frames, MOM_dir, MOM.W, MOM.H)
    
    viz_dir = os.path.join(MOM_dir, 'Flow_viz')
    os.makedirs(viz_dir, exist_ok=True)
    viz_flow(train_data, viz_dir)
        
    torch.save(train_data, os.path.join(MOM_dir, 'train_data.pth'))
    torch.save(scene_flow, os.path.join(MOM_dir, 'scene_flow.pth'))



