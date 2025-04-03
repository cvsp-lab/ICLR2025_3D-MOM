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

import os
import sys
from PIL import Image
from scene.cameras import Camera

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text 
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
    frame_num: int
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    train_cameras_2 : list
    test_cameras: list
    # video_cameras: list
    video_cameras_up: list
    video_cameras_side: list 
    video_cameras_zoom: list
    video_cameras_circle: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time = float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # breakpoint()
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def load_json(path):
    R_list = []
    T_list = []
    
    with open(path) as json_file:
        contents = json.load(json_file)
        FoVx = contents["camera_angle_x"]
        # FoVy = focal2fov(fov2focal(FoVx, W), H)
        zfar = 100.0
        znear = 0.01

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if c2w.shape[0] == 3:
                one = np.zeros((1, 4))
                one[0, -1] = 1
                c2w = np.concatenate((c2w, one), axis=0)

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            R_list.append(R)
            T_list.append(T)
            # w2c = torch.as_tensor(getWorld2View(R, T)).T.cuda()
            # proj = getProjectionMatrix(znear, zfar, FoVx, FoVy).T.cuda()
            # cams.append(MiniCam(W, H, FoVx, FoVy, znear, zfar, w2c, w2c @ proj))
    return R_list, T_list, FoVx


def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float() 

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(512,512))
        break
    R_list, T_list, fovx =load_json("./llff.json")

    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        # R = -np.transpose(matrix[:3,:3])
        # R[:,0] = -R[:,0]
        # T = -matrix[:3, 3]
        
        R = R_list[idx]
        T = T_list[idx]

        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        if idx == len(R_list):
            break
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None,frame_num=idx))
    return cam_infos
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = focal2fov(contents['fl_x'],contents['w'])
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = mapper[frame["time"]]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])    #issue 2> lucid dream의 loaded_mask가 없음!
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(800,800))
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])       # issue1> image shape 체크 -> fovy 값이 달라짐!
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None,frame_num=idx))
            
    return cam_infos

def readCamerasFromTransforms_for_LUCID(train_data, path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = focal2fov(contents['fl_x'],contents['w'])
        
        """Load LUCID & T2C data"""
        fovx = torch.load('/home/jsh/LucidDreamer/move_to_4D/fovx.pth')
        R_list = torch.load('/home/jsh/LucidDreamer/move_to_4D/R_list.pth')
        T_list = torch.load('/home/jsh/LucidDreamer/move_to_4D/T_list.pth')

        R_list_copy = copy.deepcopy(R_list)
        T_list_copy = copy.deepcopy(T_list)

        for R_sublist in R_list_copy:
            R_list.append(copy.deepcopy(R_sublist))
        
        for T_sublist in T_list_copy:
            T_list.append(copy.deepcopy(T_sublist))

        # Lucid dream에서 512,512,3 input을 rgba로 변형 시키는거라 필요 없음
        # norm_data_list = torch.load('/home/jsh/LucidDreamer/move_to_4D/norm_data_list.pth')
        # norm_data_list_copy = copy.deepcopy(norm_data_list)
        
        # for idx,norm_data_sublist in enumerate(norm_data_list_copy):
        #     norm_data_list.append(copy.deepcopy(norm_data_sublist))

        T2C_image_list = torch.load('/home/jsh/LucidDreamer/move_to_4D/inpainted_image/final_image_list_1.pth')
        T2C_image_list_2 = torch.load('/home/jsh/LucidDreamer/move_to_4D/inpainted_image/final_image_list_2.pth')        
        
        for idx, T2C_image_sublist in enumerate(T2C_image_list_2):
            T2C_image_list.append(T2C_image_sublist)
            T2C_image_sublist.save('/database/jih/Lucid_dream/T2c_image/' + str(idx) +'.png')
        
        arr_list = torch.load('/home/jsh/LucidDreamer/move_to_4D/arr_list.pth')     # 잘 돌아감
        arr_list_copy = copy.deepcopy(arr_list)
        
        for arr_sublist in arr_list_copy:
            arr_list.append(copy.deepcopy(arr_sublist))
        
        print("len(T_list)",len(T_list))
        print("len(R_list)",len(R_list))
        print("len(T2C_image_list)",len(T2C_image_list))
        print("len(arr_list)",len(arr_list))



        # loaded_mask_list = torch.load('/home/jsh/LucidDreamer/move_to_4D/loaded_mask_list.pth')

        iter = len(T2C_image_list)
        
        frames = contents["frames"][:10]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = mapper[frame["time"]]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            
            # shape, type 비교
            image = Image.open(image_path)
            
            """T2C image & alpha channel(norm_mask)"""
            image = T2C_image_list[idx]   
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

            """Lucid dream arr data """
            # arr = arr_list[idx]
            
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(arr.shape[0],arr.shape[1]))
            print("type(image)",type(image))
            print("image.shape",image.shape)

            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
            FovY = fovy 
            FovX = fovx

            """ R,T,loaded_mask data load """
            R = R_list[idx]
            T = T_list[idx]

            """ prevent gradient to masked"""
            # if loaded_mask is not None:
            #     image *= loaded_mask          ## GT에 곱하여 gradient가 안 흘러가도록 설정
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
            if idx==iter-1:
                print(f"stop at idx = {idx}")

                break
    return cam_infos

def readCamerasFromTransforms_for_ours(TrainData_path, path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    train_data = torch.load(TrainData_path)
    fovx = train_data["camera_angle_x"]
    frames = train_data["frames"]
    end_idx = len(frames)
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        """ time 받아오기 """
        frames_ = contents["frames"]
        num_time = 0
        for idx, frame in tqdm(enumerate(frames)):
            if idx % 60 == 0:
                num_time = 0
            time = mapper[frames_[num_time]["time"]]
            num_time += 1
            print(f"idx {idx} time {time}")
            # time = mapper[frames_[idx]["time"]]
            # numbers = np.linspace(0, 1, 60,  dtype=np.float32)
            # time = mapper[numbers[num_time]]

            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

        
            image = frame["image"] if "image" in frame else None
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(arr.shape[0],arr.shape[1]))
            
            # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx
            # fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path='', image_name='', width=image.shape[1], height=image.shape[2],
                            time = time, mask=None, ))
            if idx == end_idx-1:
                break
    return cam_infos

def readCamerasFromTransforms_for_ours_select(TrainData_path, path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    train_data = torch.load(TrainData_path)
    fovx = train_data["camera_angle_x"]
    frames = train_data["frames"]
    end_idx = len(frames)
    
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        """ time 받아오기 """
        frames_ = contents["frames"]
        num_time = 0
        for idx, frame in tqdm(enumerate(frames)):
            if idx % 60 == 0:
                num_time = 0
            time = mapper[frames_[num_time]["time"]]

            if idx != num_time*60 + num_time:
                num_time += 1
                continue

            # print(f"select frame on idx = {idx}")
            # print(f"select frame on num_time = {num_time}")
            # print(f"time{time}")
            
            # frame_num = idx%60
            # frame["image"].save(f"/database2/InHwanJin/4DGS/optimization/test_0_select/4DGS_traindata/{num_time}_{frame_num}.png")
            
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

        
            image = frame["image"] if "image" in frame else None
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(arr.shape[0],arr.shape[1]))
            
            # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx
            # fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path='', image_name='', width=image.shape[1], height=image.shape[2],
                            time = time, mask=None, ))
            if idx == end_idx-1:
                break
        # aaaa
    return cam_infos

def readCamerasFromTransforms_for_ours_one(TrainData_path, path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    train_data = torch.load(TrainData_path)
    fovx = train_data["camera_angle_x"]
    frames = train_data["frames"]
    end_idx = len(frames)
    

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        """ time 받아오기 """
        frames_ = contents["frames"]
        num_time = 0
        
        # jih
        # time_line 미리 세팅
        time_line = np.linspace(0, 2, 60,  dtype=np.float32)
        '''
        for idx, frame in tqdm(enumerate(frames)):
            if idx % 60 == 0:
                num_time = 0
            # dataset의 time이 아닌 3d-cinemagraphy와 동일한 세팅
            # time = mapper[frames_[num_time]["time"]]
            
            # 이걸 하면 0-2를 0-1로 매핑 시킴!
            time = mapper[time_line[num_time]]

            num_time += 1
            if idx>=60:            # if idx>60: (1 frame), if idx>=300: (5 frame) 
                break

            print(f"{idx} idx {num_time} frame_num frame's time = {time}")
            
            # frame_num = idx%60
            # frame["image"].save(f"/database2/InHwanJin/4DGS/optimization/test_0_select/4DGS_traindata/{num_time}_{frame_num}.png")
            
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

        
            image = frame["image"] if "image" in frame else None
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(arr.shape[0],arr.shape[1]))
            
            # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx
            # fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
            
            # jih
            # Stage 1의 traindata에서 마스크 불러오기
            # 이미지랑 대응하는지, 0,1 값을 갖는지 확인 필요
            # mask = train_data_Stage1[num_time//60]["frames"]["mask"]
            # mask = np.array(mask)/255
            



            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path='', image_name='', width=image.shape[1], height=image.shape[2],
                            time = time, mask=None, ))
            if idx == end_idx-1:
                break
        '''
        for idx, frame in tqdm(enumerate(frames)):
            if idx % 60 == 0:
                frame_num = idx //60
                # dataset의 time이 아닌 3d-cinemagraphy와 동일한 세팅
                # time = mapper[frames_[num_time]["time"]]
                
                # 이걸 하면 0-2를 0-1로 매핑 시킴!
                time = mapper[time_line[0]]

                print(f"{idx} idx {frame_num} frame_num frame's time = {time}")
                
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

            
                image = frame["image"] if "image" in frame else None
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                image = PILtoTorch(image,(arr.shape[0],arr.shape[1]))
                
                # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

                fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
                FovY = fovy 
                FovX = fovx
                # fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
                
                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path='', image_name='', width=image.shape[1], height=image.shape[2],
                                time = time, mask=None, ))
                if idx == end_idx-1:
                    break
    return cam_infos
def readCamerasFromTransforms_slr(TrainData_path, slr_dir ,path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []
    
    train_data = torch.load(TrainData_path)
    fovx = train_data["camera_angle_x"]
    fovy = train_data["camera_angle_y"]
    frames = train_data["frames"]


    """ SLR frame data load """
    subfolders = [d for d in os.listdir(slr_dir) if os.path.isdir(os.path.join(slr_dir, d))]
    
    # 2
    import re
    numbers = [re.search(r'\d+', subfolder).group() for subfolder in subfolders]

    for idx,subfolder in enumerate(subfolders):
        fluidimg_folder = os.path.join(slr_dir, subfolder, 'PredImg')
        images = [img for img in os.listdir(fluidimg_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
        images.sort()
        
        # frame 갯수에 맞게 timeline 및 mapper 조정
        mapper = frame_timeline(len(images))
        time_line = np.linspace(0, 2, len(images),  dtype=np.float32)

        # map_time_line = np.linspace(0, 1, len(images),  dtype=np.float32)

        # 폴더 명으로 frame index 정보 추출
        input_frame = frames[int(numbers[idx])]

        num_time = 0
        for idx, image in enumerate(images):
            time = mapper[time_line[num_time]]
            num_time += 1
            # print(f"image name :{image} frame's time = {time}")

            c2w = np.array(input_frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
        
            image_path = os.path.join(fluidimg_folder, image)
            image = Image.open(image_path)
            
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            # image = PILtoTorch(pil_image)

            # fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0]) 
            FovY = fovy 
            FovX = fovx
            
            image = torch.Tensor(arr).permute(2,0,1)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path='', image_name='', width=image.shape[2], height=image.shape[1],
                            time = time, mask=None, frame_num=idx))
    
    for idx, frame in tqdm(enumerate(frames)):
        time = mapper[time_line[0]]

        # print(f"{idx} idx frame_num frame's time = {time}")
        
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
    
        image = frame["image"] if "image" in frame else None
        im_data = np.array(image.convert("RGBA"))
        
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        # image = PILtoTorch(pil_image)
        
        # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

        # fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
        FovY = fovy 
        FovX = fovx
        
        image = torch.Tensor(arr).permute(2,0,1)
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path='', image_name='', width=image.shape[2], height=image.shape[1],
                        time = time, mask=None, frame_num=0))

        # print("width",image.shape[2])
        # print("height",image.shape[1])
        
    return cam_infos, time_line, mapper

def readCamerasFromTransforms_styleCineGAN(TrainData_path, slr_dir ,path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []
    
    train_data = torch.load(TrainData_path)
    fovx = train_data["camera_angle_x"]
    fovy = train_data["camera_angle_y"]
    frames = train_data["frames"]

    
    fluidimg_folder = os.path.join(slr_dir, 'stage_sytleGan')
    images = [img for img in os.listdir(fluidimg_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    images.sort()
    
    End_frame = 60
    mapper = frame_timeline(End_frame)
    time_line = np.linspace(0, 2, End_frame,  dtype=np.float32)

    center_view = 0
    print(f"Assume center view frame at {center_view}")
    input_frame = frames[center_view]

    num_time = 0
    for idx, image in enumerate(images):
        if idx == End_frame:
            break
        time = mapper[time_line[num_time]]
        num_time += 1
        # print(f"image name :{image} frame's time = {time}")

        c2w = np.array(input_frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
    
        image_path = os.path.join(fluidimg_folder, image)
        image = Image.open(image_path)
        
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        # image = PILtoTorch(pil_image)

        # fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
        FovY = fovy 
        FovX = fovx
    
        image = torch.Tensor(arr).permute(2,0,1)
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path='', image_name='', width=image.shape[2], height=image.shape[1],
                        time = time, mask=None, frame_num=idx))
    
    for idx, frame in tqdm(enumerate(frames)):
        time = mapper[time_line[0]]

        # print(f"{idx} idx frame_num frame's time = {time}")
        
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
    
        image = frame["image"] if "image" in frame else None
        im_data = np.array(image.convert("RGBA"))
        
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        # image = PILtoTorch(pil_image)
        
        # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

        # fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
        FovY = fovy 
        FovX = fovx
        
        image = torch.Tensor(arr).permute(2,0,1)
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path='', image_name='', width=image.shape[2], height=image.shape[1],
                        time = time, mask=None, frame_num=0))

        # print("width",image.shape[2])
        # print("height",image.shape[1])
        
    return cam_infos, time_line, mapper

def readCamerasFromTransforms_T2C(TrainData_path, slr_path ,path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []
    
    train_data = torch.load(TrainData_path)
    fovx = train_data["camera_angle_x"]
    frames = train_data["frames"]
    end_idx = len(frames)

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        """ time 받아오기 """
        frames_ = contents["frames"]
        num_time = 0
        time_line = np.linspace(0, 2, 60,  dtype=np.float32)

        for idx, frame in tqdm(enumerate(frames)):
            time = mapper[time_line[0]]

            print(f"{idx} idx frame_num frame's time = {time}")
            
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
        
            image = frame["image"] if "image" in frame else None
            im_data = np.array(image.convert("RGBA"))
            
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            # image = PILtoTorch(pil_image)
            
            # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

            fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
            FovY = fovy 
            FovX = fovx
            # fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
            
            image = torch.Tensor(arr).permute(2,0,1)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path='', image_name='', width=image.shape[1], height=image.shape[2],
                            time = time, mask=None, ))
            if idx == end_idx-1:
                break
        
        """ SLR frame data load """
        # fluidimg_folder = '/database2/InHwanJin/4DGS_2/test_4/Stage_slr/output/stage_slr/FluidImg'
        # /database2/InHwanJin/4DGS_2/Qualitative/test_10/T2C_video/2
        base_folder = slr_path
        subfolders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

        # 2,17,32
        import re
        numbers = [re.search(r'\d+', subfolder).group() for subfolder in subfolders]
            
        for idx,subfolder in enumerate(subfolders):
            subfolder_path = os.path.join(base_folder, subfolder)
            # SLR output의 pred, fluid 중 predImg 사용
            # fluidimg_folder = os.path.join(subfolder_path, 'PredImg')
            print("fluidimg_folder at ", subfolder_path)

            images = [img for img in os.listdir(subfolder_path) if img.endswith((".jpg", ".jpeg", ".png"))]
            images.sort()

            train_index = int(numbers[idx])
            input_frame = frames[train_index]
            print("train_index", train_index)

            num_time = 0
            for idx, image in enumerate(images):
                time = mapper[time_line[num_time]]
                num_time += 1
                print(f"image name :{image} frame's time = {time}")
                
                # if idx == 0:
                #     test_image = input_frame["image"]
                #     test_image.save(f'/database2/InHwanJin/4DGS_2/test/{train_index}.png')

                c2w = np.array(input_frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]
            
                image_path = os.path.join(subfolder_path, image)
                image = Image.open(image_path)
                
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                # image = PILtoTorch(pil_image)

                fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
                FovY = fovy 
                FovX = fovx
                # fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
                
                image = torch.Tensor(arr).permute(2,0,1)
                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path='', image_name='', width=image.shape[1], height=image.shape[2],
                                time = time, mask=None, ))
        
    return cam_infos

def readCamerasFromTransforms_slr_single(TrainData_path, slr_path ,path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []
    
    train_data = torch.load(TrainData_path)
    fovx = train_data["camera_angle_x"]
    frames = train_data["frames"]
    end_idx = len(frames)

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        """ time 받아오기 """
        frames_ = contents["frames"]
        num_time = 0
        time_line = np.linspace(0, 2, 60,  dtype=np.float32)

        for idx, frame in tqdm(enumerate(frames)):
            time = mapper[time_line[0]]

            print(f"{idx} idx frame_num frame's time = {time}")
            
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
        
            image = frame["image"] if "image" in frame else None
            im_data = np.array(image.convert("RGBA"))
            
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            # image = PILtoTorch(pil_image)
            
            # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

            fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
            FovY = fovy 
            FovX = fovx
            # fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
            
            image = torch.Tensor(arr).permute(2,0,1)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path='', image_name='', width=image.shape[1], height=image.shape[2],
                            time = time, mask=None, ))
            if idx == end_idx-1:
                break
        
        """ SLR frame data load """
        # fluidimg_folder = '/database2/InHwanJin/4DGS_2/test_4/Stage_slr/output/stage_slr/FluidImg'
        base_folder = slr_path
        subfolders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

        # 2,17,32
        import re
        numbers = [re.search(r'\d+', subfolder).group() for subfolder in subfolders]
            
        for idx,subfolder in enumerate(subfolders):
            subfolder_path = os.path.join(base_folder, subfolder)
            # SLR output의 pred, fluid 중 predImg 사용
            fluidimg_folder = os.path.join(subfolder_path, 'PredImg')

            images = [img for img in os.listdir(fluidimg_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
            images.sort()

            train_index = int(numbers[idx])
            if train_index != 2:
                break
            input_frame = frames[train_index]
            print("fluidimg_folder at ", fluidimg_folder)

            num_time = 0
            for idx, image in enumerate(images):
                time = mapper[time_line[num_time]]
                num_time += 1
                print(f"image name :{image} frame's time = {time}")
                
                # if idx == 0:
                #     test_image = input_frame["image"]
                #     test_image.save(f'/database2/InHwanJin/4DGS_2/test/{train_index}.png')

                c2w = np.array(input_frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]
            
                image_path = os.path.join(fluidimg_folder, image)
                image = Image.open(image_path)
                
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                # image = PILtoTorch(pil_image)

                fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
                FovY = fovy 
                FovX = fovx
                # fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
                
                image = torch.Tensor(arr).permute(2,0,1)
                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path='', image_name='', width=image.shape[1], height=image.shape[2],
                                time = time, mask=None, ))
        
    return cam_infos


def generateCamerasFromTransforms_one_path(time_line, cam_path, R_list, T_list, path, template_transformsfile, extension, maxtime, input_width, input_height, mapper = {}):

    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    # render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    H, W = input_height, input_width


    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image, (W, H))  # (W, H)로 변환합니다.
        break

    # _, _, fovx =load_json("/home/jsh/4DGaussians/llff.json")
    # time_line = np.linspace(0, 2, 60,  dtype=np.float32)
    # focal = (5.8269e+02, 5.8269e+02)
    # fov = (2*np.arctan(W / (2*focal[0])), 2*np.arctan(H / (2*focal[1])))
    # fovx = fov[0] 


    # input size에 맞게 focal length 변경
    focal_length = 5.8269e+02
    aspect_ratio = W / H
    f_x = focal_length * aspect_ratio
    f_y = focal_length
    focal = (f_x,f_y)
    fov = (2*np.arctan(W / (2*focal[0])), 2*np.arctan(H / (2*focal[1])))

    fovx = fov[0] 
    fovy = fov[1] 

    '''
    if cam_path == 'up':
        timestamp_mapper = {}
        max_time_float = max(time_line)
        for index, time in enumerate(time_line):
            timestamp_mapper[time] = time/max_time_float
        mapper = timestamp_mapper
        
        frame_idx = 0
        
        for idx in range(len(time_line)):
            time = mapper[time_line[frame_idx]]
            frame_idx += 1

            R = np.eye(3)
            T = [0., 0.08-0.005*idx, 0.]
            T = np.transpose(np.array(T))

            FovY = fovy 
            FovX = fovx
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=None, image_name=None, width=W, height=H,
                                time = time, mask=None, frame_num=idx))   

    if cam_path == 'zoom':
        timestamp_mapper = {}
        max_time_float = max(time_line)
        for index, time in enumerate(time_line):
            timestamp_mapper[time] = time/max_time_float
        mapper = timestamp_mapper
        
        frame_idx = 0
        
        for idx in range(len(time_line)):
            time = mapper[time_line[frame_idx]]
            frame_idx += 1

            # R = R_list[idx].clone().cpu().numpy()
            # T = T_list[idx].clone().cpu().numpy()

            R = np.eye(3)
            # for Drone view (y랑 z 축만 변경)
            T = [0.*idx, 0.00271186*idx, -0.0040678*idx]
            T = np.transpose(np.array(T))

            FovY = fovy 
            FovX = fovx
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=None, image_name=None, width=W, height=H,
                                time = time, mask=None, frame_num=idx))   
    
    if cam_path == 'side':
        timestamp_mapper = {}
        max_time_float = max(time_line)
        for index, time in enumerate(time_line):
            timestamp_mapper[time] = time/max_time_float
        mapper = timestamp_mapper
        
        frame_idx = 0
        theta = np.linspace(0, 2 * np.pi, len(time_line))  # 

        for idx in range(len(time_line)):
            time = mapper[time_line[frame_idx]]
            frame_idx += 1
            R = np.eye(3)

            
            # T = [0.*idx, 0.00271186*idx, -0.0040678*idx]
            T = [0.09-0.003*idx, 0, -0.0040678*idx]
            T = np.transpose(np.array(T))
            

            a = 0.05  # x축 반경
            b = 0.09  # y축 반경
            c = 0.04  # z축 타원 반경


            # T = [0.09-0.0015*idx, b * np.sin(theta[idx]), c * np.sin(theta[idx])]
            T = [a * np.cos(theta[idx]), b * np.sin(theta[idx]), c * np.sin(theta[idx])]
            

            # 이동할 값 정의
            x_shift = -0.002 * idx  # x축: 오른쪽으로 점진적 이동
            y_shift = 0.005 * idx  # y축: 위로 점진적 상승
            z_shift = -0.001 * idx  # z축: 안쪽으로 점진적 감소

            T = [x_shift, y_shift, z_shift]
            

            T = np.transpose(np.array(T))


            # fovy = focal2fov(fov2focal(fovx, W), H) 
            FovY = fovy 
            FovX = fovx
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=None, image_name=None, width=W, height=H,
                                time = time, mask=None, frame_num=idx))
        '''

    for idx, poses in enumerate(R_list):
        if idx >= 60:
            break
        time = mapper[time_line[idx]]

        R = R_list[idx].clone().cpu().numpy()
        T = T_list[idx].clone().cpu().numpy()

        FovY = fovy 
        FovX = fovx
        if idx == len(R_list)-1:
            break
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=W, height=H,
                            time = time, mask=None, frame_num=idx))   
    return cam_infos


def readCamerasFromTransforms_MVS(TrainData_path, path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    train_data = torch.load(TrainData_path)
    fovx = train_data["camera_angle_x"]
    fovy = train_data["camera_angle_y"]

    frames = train_data["frames"]
    end_idx = len(frames)
    

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        # time_line 세팅
        time_line = np.linspace(0, 2, 60,  dtype=np.float32)
        for idx, frame in tqdm(enumerate(frames)):
            time = mapper[time_line[0]]

            # print(f"{idx} idx frame_num frame's time = {time}")
            
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
        
            image = frame["image"] if "image" in frame else None
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            # image = PILtoTorch(pil_image)
            
            # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

            # fovy = focal2fov(fov2focal(fovx, image.size[1]), image.size[0])
            FovY = fovy 
            FovX = fovx
            
            image = torch.Tensor(arr).permute(2,0,1)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path='', image_name='', width=image.shape[2], height=image.shape[1],
                            time = time, mask=None, frame_num=0))
           
        # print("width",image.shape[2])
        # print("height",image.shape[1])
    return cam_infos

def readCamerasFromTransforms_for_ours_test(TrainData_path, path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    train_data = torch.load(TrainData_path)
    fovx = train_data["camera_angle_x"]
    frames = train_data["frames"][:20]
    end_idx = len(frames)
    
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        """ time 받아오기 """
        frames_ = contents["frames"]
        num_time = 0
        # jih
        # time_line 미리 세팅
        time_line = np.linspace(0, 2, 60,  dtype=np.float32)
        for idx, frame in tqdm(enumerate(frames)):
            if idx % 60 == 0:
                num_time = 0
            # dataset의 time이 아닌 3d-cinemagraphy와 동일한 세팅
            # time = mapper[frames_[num_time]["time"]]
            
            # 이걸 하면 0-2를 0-1로 매핑 시킴!
            time = mapper[time_line[num_time]]

            num_time += 1
            # if idx>60:            # if idx>60: (1 frame), if idx>=300: (5 frame) 
            #     continue

            # print(f"{idx} idx {num_time} frame_num frame's time = {time}")
            
            # frame_num = idx%60
            # frame["image"].save(f"/database2/InHwanJin/4DGS/optimization/test_0_select/4DGS_traindata/{num_time}_{frame_num}.png")
            
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

        
            image = frame["image"] if "image" in frame else None
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(arr.shape[0],arr.shape[1]))
            
            # loaded_mask = np.ones_like(norm_data[:, :, 3:4])

            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx
            # fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])     
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path='', image_name='', width=image.shape[1], height=image.shape[2],
                            time = time, mask=None, frame_num= 0))
            if idx == end_idx-1:
                break
    return cam_infos


def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)  
    #print("train_json[frames][60]",train_json["frames"][60]) 0.40268456375838924
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in train_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    # time_line = time_line[:60]
    # print("time_line",time_line)
    
    # 3D-cinemagraphy와 동일한 세팅 
    # 기존 데이터셋 보다 시간 간격이 큼
    time_line = np.linspace(0, 2, 60,  dtype=np.float32)
    # print("time_line",time_line)

    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        # timestamp_mapper[time] = index
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float

def frame_timeline(num_frames):
    time_line = np.linspace(0, 2, num_frames,  dtype=np.float32)

    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper

def readNerfSyntheticInfo(TrainData_path, slr_path, path, selected_frame, white_background, eval, extension=".png"):
    timestamp_mapper, max_time = read_timeline(path)        # len(timestamp) = 150
    
    # '''
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms_MVS(TrainData_path, path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("MVS data len: ",len(train_cam_infos))
    # train_cam_infos_2, time_line, mapper = readCamerasFromTransforms_slr(TrainData_path, slr_path, path, "transforms_train.json", white_background, extension, timestamp_mapper)
    train_cam_infos_2, time_line, mapper = readCamerasFromTransforms_styleCineGAN(TrainData_path, slr_path, path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("stage 2 MVS & frame data ",len(train_cam_infos_2))
    test_cam_infos = readCamerasFromTransforms_for_ours_test(TrainData_path, path, "transforms_train.json", white_background, extension, timestamp_mapper)      # readCamerasFromTransforms_for_ours_test
    # video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    
    
    R_list_up = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/up-down_R_list')
    T_list_up = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/up-down_t_list')
    R_list_side = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/side_R_list')
    T_list_side = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/side_t_list')
    # R_list_zoom = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/zoom-in_R_list')
    # T_list_zoom = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/zoom-in_t_list')
    R_list_zoom = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/vfx_R_list')
    T_list_zoom = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/vfx_t_list')
    R_list_circle = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/circle_R_list')
    T_list_circle = torch.load('/home/cvsp/InHwanJIn/4DGaussians_ours/CameraPath/cinemagraphy_cameraPath/circle_t_list')

    # load input image
    input_list = TrainData_path.split('/')[:-2]
    input_dir = '/'.join(input_list)

    input_image = Image.open(os.path.join(input_dir, 'image.png'))
    input_width, input_height = input_image.size
    
    video_cam_infos_up = generateCamerasFromTransforms_one_path(time_line, "up", R_list_up, T_list_up, path, "transforms_train.json", extension, max_time, input_width, input_height, mapper)
    video_cam_infos_side = generateCamerasFromTransforms_one_path(time_line, "side", R_list_side, T_list_side, path, "transforms_train.json", extension, max_time, input_width, input_height, mapper)
    video_cam_infos_zoom = generateCamerasFromTransforms_one_path(time_line, "zoom", R_list_zoom, T_list_zoom, path, "transforms_train.json", extension, max_time, input_width, input_height, mapper)
    video_cam_infos_circle = generateCamerasFromTransforms_one_path(time_line, "circle", R_list_circle, T_list_circle, path, "transforms_train.json", extension, max_time, input_width, input_height, mapper)


    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "fused.ply")
    '''
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        print("xyz.shape",xyz.shape)
        print("type(xyz)",type(xyz))
        print("xyz",xyz)
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        pcd_ = np.array(pcd)
        # print(pcd_.shape)
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)
        # xyz = -np.array(pcd.points)
        # pcd = pcd._replace(points=xyz)
    # '''

    """ Ours PCD Load"""
    # TrainData = torch.load(TrainData_path)
    # pcd_color = TrainData['pcd_colors']*255
    # pcd = BasicPointCloud(points=TrainData['pcd_points'].T, colors=pcd_color, normals=None)

    
    TrainData = torch.load(TrainData_path)
    pcd = BasicPointCloud(points=TrainData['pcd_points'].T, colors=TrainData['pcd_colors'], normals=None)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           train_cameras_2=train_cam_infos_2,
                           test_cameras=test_cam_infos,
                           video_cameras_up=video_cam_infos_up, 
                           video_cameras_side=video_cam_infos_side, 
                           video_cameras_zoom=video_cam_infos_zoom,
                           video_cameras_circle=video_cam_infos_circle,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info, time_line
def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras


def readHyperDataInfos(datadir,use_bg_points,eval):
    train_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split="test")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train")
    print("format finished")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"


    ply_path = os.path.join(datadir, "points3D_downsample.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)

    pcd = pcd._replace(points=xyz)
    nerf_normalization = getNerfppNorm(train_cam)
    plot_camera_orientations(train_cam_infos, pcd.points)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info
def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def add_points(pointsclouds, xyz_min, xyz_max):
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
    # breakpoint()
    # new_
def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load
    pcd = fetchPly(ply_path)
    print("origin points,",pcd.points.shape[0])
    
    print("after points,",pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam
def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")
    # breakpoint()
def readPanopticmeta(datadir, json_path):
    with open(os.path.join(datadir,json_path)) as f:
        test_meta = json.load(f)
    w = test_meta['w']
    h = test_meta['h']
    max_time = len(test_meta['fn'])
    cam_infos = []
    for index in range(len(test_meta['fn'])):
        focals = test_meta['k'][index]
        w2cs = test_meta['w2c'][index]
        fns = test_meta['fn'][index]
        cam_ids = test_meta['cam_id'][index]

        time = index / len(test_meta['fn'])
        # breakpoint()
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image_path = os.path.join(datadir,"ims")
            image_name=fn
            
            # breakpoint()
            image = Image.open(os.path.join(datadir,"ims",fn))
            im_data = np.array(image.convert("RGBA"))
            # breakpoint()
            im_data = PILtoTorch(im_data,None)[:3,:,:]
            # breakpoint()
            # print(w2c,focal,image_name)
            camera = setup_camera(w, h, focal, w2c)
            cam_infos.append({
                "camera":camera,
                "time":time,
                "image":im_data})
            
    cam_centers = np.linalg.inv(test_meta['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    # breakpoint()
    return cam_infos, max_time, scene_radius 
def readPanopticSportsinfos(datadir):
    train_cam_infos, max_time, scene_radius = readPanopticmeta(datadir, "train_meta.json")
    test_cam_infos,_, _ = readPanopticmeta(datadir, "test_meta.json")
    nerf_normalization = {
        "radius":scene_radius,
        "translate":torch.tensor([0,0,0])
    }

    ply_path = os.path.join(datadir, "pointd3D.ply")

        # Since this data set has no colmap data, we start with random points
    plz_path = os.path.join(datadir, "init_pt_cld.npz")
    data = np.load(plz_path)["data"]
    xyz = data[:,:3]
    rgb = data[:,3:6]
    num_pts = xyz.shape[0]
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.ones((num_pts, 3)))
    storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    # breakpoint()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           )
    return scene_info
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "PanopticSports" : readPanopticSportsinfos

}
