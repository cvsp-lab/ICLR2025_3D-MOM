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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
# from gaussian_renderer_warp import render
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures


def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, delta_scale=1):
    render_path = os.path.join(model_path, 'frame_result', name)
    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    # breakpoint()
    crop = 32
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        # breakpoint()
        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type, delta_scale=delta_scale)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rendering = to8b(rendering).transpose(1,2,0)
        rendering = rendering[crop:-crop, crop:-crop]
        render_images.append(rendering)
        # print(to8b(rendering).shape)
        render_list.append(rendering)
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    multithread_write(render_list, render_path)

    video_path = os.path.join(model_path,'vid_result')
    os.makedirs(video_path,exist_ok=True)
    imageio.mimwrite(os.path.join(video_path, name+'.mp4'), render_images, fps=30)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool,  TrainData_path: str, Gaussian_path: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(TrainData_path, Gaussian_path, dataset, gaussians, load_iteration=iteration, shuffle=False)

        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(Gaussian_path,"up_down",scene.loaded_iter,scene.getVideoCameras_up(),gaussians,pipeline,background,cam_type)
        render_set(Gaussian_path,"side",scene.loaded_iter,scene.getVideoCameras_side(),gaussians,pipeline,background,cam_type)
        render_set(Gaussian_path,"zoom",scene.loaded_iter,scene.getVideoCameras_zoom(),gaussians,pipeline,background,cam_type)
        render_set(Gaussian_path,"circle",scene.loaded_iter,scene.getVideoCameras_circle(),gaussians,pipeline,background,cam_type)

            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", default = "arguments/dnerf/hellwarrior.py",type=str)
    parser.add_argument('--input_dir', type=str, help='input folder that contains src images', required=True)

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    MOM_dir = os.path.join(args.input_dir,"MOM")
    TrainData_path = os.path.join(MOM_dir,"train_data.pth")
    Gaussian_path = args.input_dir

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, TrainData_path, Gaussian_path)