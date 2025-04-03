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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON 
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
class Scene:

    gaussians : GaussianModel

    def __init__(self, TrainData_path, Gaussian_path, args : ModelParams, gaussians : GaussianModel, flow_scale=1, viewcrafter=False, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """

        self.model_path = Gaussian_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        self.video_cameras_up = {}
        self.video_cameras_side = {}
        self.video_cameras_zoom = {}
        self.video_cameras_circle = {}

        scene_info, time_line = sceneLoadTypeCallbacks["Blender"](TrainData_path, args.source_path, args.white_background, args.eval, viewcrafter, args.extension)
        dataset_type="blender"
        
        self.time_line = time_line
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        self.train_camera_2 = FourDGSdataset(scene_info.train_cameras_2, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
                
        self.video_cameras_up = FourDGSdataset(scene_info.video_cameras_up, args, dataset_type)
        self.video_cameras_side = FourDGSdataset(scene_info.video_cameras_side, args, dataset_type)
        self.video_cameras_zoom = FourDGSdataset(scene_info.video_cameras_zoom, args, dataset_type)
        self.video_cameras_circle = FourDGSdataset(scene_info.video_cameras_circle, args, dataset_type)

        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime, TrainData_path, flow_scale)

    def save(self, iteration, stage):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)


    def getTrainCameras(self, scale=1.0):
        return self.train_camera
    def getTrainCameras_2(self, scale=1.0):
        return self.train_camera_2
    def getTestCameras(self, scale=1.0):
        return self.test_camera
    # def getVideoCameras(self, scale=1.0):
    #     return self.video_camera
    def getTimeline(self):
        return self.time_line
    def getVideoCameras_up(self, scale=1.0):
        return self.video_cameras_up
    def getVideoCameras_side(self, scale=1.0):
        return self.video_cameras_side
    def getVideoCameras_zoom(self, scale=1.0):
        return self.video_cameras_zoom
    def getVideoCameras_circle(self, scale=1.0):
        return self.video_cameras_circle
    