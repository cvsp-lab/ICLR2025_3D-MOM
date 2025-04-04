# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from kornia.morphology import opening, erosion
from kornia.filters import gaussian_blur2d
from lib.model.inpaint.networks.inpainting_nets import Inpaint_Depth_Net, Inpaint_Color_Net
from lib.utils.render_utils import masked_median_blur
import imageio
import numpy as np

from diffusers import (
    StableDiffusionInpaintPipeline, StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline)
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, InterpolationMode
import torchvision
import os
from third_party.DPT.run_monodepth import run_dpt
from lib.utils.general_utils import *
from lib.utils.render_utils import *

def refine_near_depth_discontinuity(depth, alpha, kernel_size=11):
    '''
    median filtering the depth discontinuity boundary
    '''
    depth = depth * alpha
    depth_median_blurred = masked_median_blur(depth, alpha, kernel_size=kernel_size) * alpha
    alpha_eroded = erosion(alpha, kernel=torch.ones(kernel_size, kernel_size).to(alpha.device))
    depth[alpha_eroded == 0] = depth_median_blurred[alpha_eroded == 0]
    return depth


def define_inpainting_bbox(alpha, border=40):
    '''
    define the bounding box for inpainting
    :param alpha: alpha map [1, 1, h, w]
    :param border: the minimum distance from a valid pixel to the border of the bbox
    :return: [1, 1, h, w], a 0/1 map that indicates the inpainting region
    '''
    assert alpha.ndim == 4 and alpha.shape[:2] == (1, 1)
    x, y = torch.nonzero(alpha)[:, -2:].T
    h, w = alpha.shape[-2:]
    row_min, row_max = x.min(), x.max()
    col_min, col_max = y.min(), y.max()
    out = torch.zeros_like(alpha)
    x0, x1 = max(row_min - border, 0), min(row_max + border, h - 1)
    y0, y1 = max(col_min - border, 0), min(col_max + border, w - 1)
    out[:, :, x0:x1, y0:y1] = 1
    return out


class Inpainter():
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading depth model...")
        depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load('ckpts/depth-model.pth', map_location=torch.device(device))
        depth_feat_model.load_state_dict(depth_feat_weight)
        depth_feat_model = depth_feat_model.to(device)
        depth_feat_model.eval()
        self.depth_feat_model = depth_feat_model.to(device)
        print("Loading rgb model...")
        rgb_model = Inpaint_Color_Net()
        rgb_feat_weight = torch.load('ckpts/color-model.pth', map_location=torch.device(device))
        rgb_model.load_state_dict(rgb_feat_weight)
        rgb_model.eval()
        self.rgb_model = rgb_model.to(device)
        
        self.default_model = 'SD1.5 (default)'
        self.stable_model = StableDiffusionInpaintPipeline.from_pretrained(
            'runwayml/stable-diffusion-inpainting', revision='fp16', torch_dtype=torch.float16).to('cuda')
        self.current_model = self.default_model

        # kernels
        self.context_erosion_kernel = torch.ones(10, 10).to(self.device)
        self.alpha_kernel = torch.ones(3, 3).to(self.device)

    @staticmethod
    def process_depth_for_network(depth, context, log_depth=True):
        if log_depth:
            log_depth = torch.log(depth + 1e-8) * context
            mean_depth = torch.mean(log_depth[context > 0])
            zero_mean_depth = (log_depth - mean_depth) * context
        else:
            zero_mean_depth = depth
            mean_depth = 0
        return zero_mean_depth, mean_depth

    @staticmethod
    def deprocess_depth(zero_mean_depth, mean_depth, log_depth=True):
        if log_depth:
            depth = torch.exp(zero_mean_depth + mean_depth)
        else:
            depth = zero_mean_depth
        return depth

    def inpaint_rgb(self, holes, context, context_rgb, edge):
        # inpaint rgb
        with torch.no_grad():
            inpainted_rgb = self.rgb_model.forward_3P(holes, context, context_rgb, edge,
                                                      unit_length=128, cuda=self.device)
        inpainted_rgb = inpainted_rgb.detach() * holes + context_rgb
        inpainted_a = holes + context
        inpainted_a = opening(inpainted_a, self.alpha_kernel)
        inpainted_rgba = torch.cat([inpainted_rgb, inpainted_a], dim=1)
        return inpainted_rgba

    def inpaint_depth(self, depth, holes, context, edge, depth_range):
        zero_mean_depth, mean_depth = self.process_depth_for_network(depth, context)
        with torch.no_grad():
            inpainted_depth = self.depth_feat_model.forward_3P(holes, context, zero_mean_depth, edge,
                                                               unit_length=128, cuda=self.device)
        inpainted_depth = self.deprocess_depth(inpainted_depth.detach(), mean_depth)
        inpainted_depth[context > 0.5] = depth[context > 0.5]
        inpainted_depth = gaussian_blur2d(inpainted_depth, (3, 3), (1.5, 1.5))
        inpainted_depth[context > 0.5] = depth[context > 0.5]
        # if the inpainted depth in the background is smaller that the foreground depth,
        # then the inpainted content will mistakenly occlude the foreground.
        # Clipping the inpainted depth in this situation.
        mask_wrong_depth_ordering = inpainted_depth < depth
        inpainted_depth[mask_wrong_depth_ordering] = depth[mask_wrong_depth_ordering] * 1.01
        inpainted_depth = torch.clamp(inpainted_depth, min=min(depth_range) * 0.9)
        return inpainted_depth
    
    def min_max(self, depth_layer, d_1, d_2):
        data_min = depth_layer.min()
        data_max = depth_layer.max()
        # print("type(depth_layer)",type(depth_layer))
        # print("type(data_max)",type(data_max))
        # print("type(data_min)",type(data_min))
        # print("type(d_2)",type(d_2))
        # print("type(d_1)",type(d_1))

        data_normalized = (d_2 - d_1)*(depth_layer - data_min) / (data_max - data_min)
        depth_normalized = (data_normalized) + d_1
        return depth_normalized

    def sequential_inpainting(self, rgb, depth, depth_bins, prompt, input_dir):
        '''
        :param rgb: [1, 3, H, W]
        :param depth: [1, 1, H, W]
        :return: rgba_layers: [N, 1, 3, H, W]: the inpainted RGBA layers
                 depth_layers: [N, 1, 1, H, W]: the inpainted depth layers
                 mask_layers:  [N, 1, 1, H, W]: the original alpha layers (before inpainting)
        '''

        num_bins = len(depth_bins) - 1

        rgba_layers = []
        depth_layers = []
        mask_layers = []

        for i in range(num_bins):
            alpha_i = (depth >= depth_bins[i]) * (depth < depth_bins[i + 1])
            alpha_i = alpha_i.float()

            if i == 0:
                rgba_i = torch.cat([rgb * alpha_i, alpha_i], dim=1)
                rgba_layers.append(rgba_i)
                depth_i = refine_near_depth_discontinuity(depth, alpha_i)
                
                d_1 = depth_bins[0].clone().cpu().numpy()
                d_2 = depth_bins[1]
                depth_zero = depth_i.clone().cpu().numpy()
                print("depth_zero.shape",depth_zero.shape)
                print("depth_zero", depth_zero)
                norm_depth = self.min_max(depth_zero, d_1, d_2)
                norm_depth_i = torch.tensor(norm_depth)
                norm_depth_i = norm_depth_i.to('cuda')
                norm_depth_i = norm_depth_i * alpha_i
                
                print("norm_depth_i.shape",norm_depth_i.shape)
                print("norm_depth_i", norm_depth_i)
                
                depth_layers.append(norm_depth_i)
                mask_layers.append(alpha_i)
                pre_alpha = alpha_i.bool()
                pre_inpainted_depth = depth * alpha_i
            else:
                alpha_i_eroded = erosion(alpha_i, self.context_erosion_kernel)
                if alpha_i_eroded.sum() < 10:
                    continue

                context = erosion((depth >= depth_bins[i]).float(), self.context_erosion_kernel)
                context_2 = erosion((depth >= depth_bins[i+1]).float(), self.context_erosion_kernel)

                holes = 1. - context
                
                """ inpainted alpha channel"""
                holes_2 = 1. - context_2
                input_hole = holes.clone()
                input_hole_2 = holes_2.clone()
                # inpainted_hole = input_hole+input_hole_2

                bbox = define_inpainting_bbox(context, border=40)
                holes *= bbox
                edge = torch.zeros_like(holes)
                context_rgb = rgb * context
                
                
                # '''
                """ MK4D inpainting """
                inpainting_dir = os.path.join(input_dir,'inpainting')
                os.makedirs(inpainting_dir,exist_ok=True)

                # inpaint rgb
                input_rgb = context_rgb.clone().cpu().numpy()
                input_rgb = (input_rgb * 255).astype(np.uint8)
                input_rgb = np.squeeze(input_rgb)
                image_in = input_rgb.transpose(1, 2, 0) 
                
                input_hole = input_hole.cpu().numpy()
                input_hole = (input_hole * 255).astype(np.uint8)
                mask_in = np.squeeze(input_hole)

                ''' Visualize input image & mask
                pil_image = Image.fromarray(image_in)
                image_path = os.path.join(inpainting_dir, "image.png")
                pil_image.save(image_path)

                pil_mask = Image.fromarray(mask_in)
                mask_path = os.path.join(inpainting_dir, "mask.png")
                pil_mask.save(mask_path)
                '''

                image2 = np.array(Image.fromarray(image_in)).astype(float) / 255.0
                mask2 = np.array(Image.fromarray(mask_in)).astype(float) / 255.0
                negative_prompt = 'unrealistic, fantasy, fairy tale'
                input_prompt = prompt
                seed = 1

                generator=torch.Generator(device='cuda').manual_seed(seed)

                image_curr = self.rgb(
                    prompt=input_prompt,
                    image=image2,
                    negative_prompt=negative_prompt, generator=generator,
                    mask_image=mask2,
                )                
                
                to_tensor = ToTensor()
                
                image_curr = np.array(image_curr)/255.
                image_curr_tensor  = to_tensor(image_curr).float()[None]
                image_curr_tensor  = image_curr_tensor.to('cuda')
                
                H,W = image_curr.shape[:2]
                
                image_curr_np = image_curr_tensor[0].clone().cpu().permute(1, 2, 0).numpy()
                image_curr_np = (image_curr_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_curr_np)
                image_path = os.path.join(inpainting_dir, "image.png")
                pil_image.save(image_path)      # 저장은 뒷 레이어랑 합쳐서 (depth 추정을 위함)

                inpainted_rgba_i = torch.cat([image_curr_tensor , input_hole_2], dim=1) # 레이어는 뒷 레이어 영역을 알파 채널로 넣어서 전달
                
                # inpainted_rgba_i = self.inpaint_rgb(holes, context, context_rgb, edge)  # inpainted_rgba_i.shape torch.Size([1, 4, 512, 512]), type(inpainted_rgba_i) <class 'torch.Tensor'>
                
                """MK4D depth estimation"""
                dpt_out_dir = os.path.join(inpainting_dir, 'depth_layer')
                os.makedirs(dpt_out_dir,exist_ok=True)
                dpt_model_path = 'ckpts/dpt_hybrid-midas-501f0c75.pt'
                run_dpt(input_path=inpainting_dir, output_path=dpt_out_dir, model_path=dpt_model_path, optimize=False)
                disp_file = os.path.join(dpt_out_dir, 'image.png')


                inpainted_depth_i = imageio.imread(disp_file) / 65535.
                inpainted_depth_i = remove_noise_in_dpt_disparity(inpainted_depth_i)
                inpainted_depth_i = 1. / np.maximum(inpainted_depth_i, 1e-6)
                
                d_1 = depth_bins[i]

                if i == num_bins-1:
                    d_2 = depth_bins[i+1].clone().cpu().numpy()
                else:
                    d_2 = depth_bins[i+1]

                inpainted_depth_i = self.min_max(inpainted_depth_i, d_1, d_2)
                inpainted_depth_i  = to_tensor(inpainted_depth_i).float()[None]
                inpainted_depth_i  = inpainted_depth_i.to('cuda')
                print("inpainted_depth_i",inpainted_depth_i)
                inpainted_depth_i = inpainted_depth_i * input_hole_2
                print("inpainted_depth_i",inpainted_depth_i)
                
                
                # '''
                # inpaint depth
                # inpainted_depth_i = self.inpaint_depth(depth, holes, context, edge, (depth_bins[i], depth_bins[i + 1]))
                depth_near_mask = (inpainted_depth_i < depth_bins[i + 1]).float()
                # '''
                
                if i < num_bins - 1:
                    # only keep the content whose depth is smaller than the upper limit of the current layer
                    # otherwise the inpainted content on the far-depth edge will falsely occlude the next layer.
                    inpainted_rgba_i *= depth_near_mask
                    inpainted_depth_i = refine_near_depth_discontinuity(inpainted_depth_i, inpainted_rgba_i[:, [-1]])

                inpainted_alpha_i = inpainted_rgba_i[:, [-1]].bool()
                mask_wrong_ordering = (inpainted_depth_i <= pre_inpainted_depth) * inpainted_alpha_i
                inpainted_depth_i[mask_wrong_ordering] = pre_inpainted_depth[mask_wrong_ordering] * 1.05


                rgba_layers.append(inpainted_rgba_i)
                depth_layers.append(inpainted_depth_i)
                mask_layers.append(context * depth_near_mask)  # original mask

                pre_alpha[inpainted_alpha_i] = True
                pre_inpainted_depth[inpainted_alpha_i > 0] = inpainted_depth_i[inpainted_alpha_i > 0]

        rgba_layers = torch.stack(rgba_layers)
        depth_layers = torch.stack(depth_layers)
        mask_layers = torch.stack(mask_layers)

        return rgba_layers, depth_layers, mask_layers
    
    def rgb(self, prompt, image, negative_prompt='', generator=None, num_inference_steps=50, mask_image=None):
        image_pil = Image.fromarray(np.round(image * 255.).astype(np.uint8))
        mask_pil = Image.fromarray(np.round((mask_image) * 255.).astype(np.uint8))
        if self.current_model == self.default_model:
            return self.stable_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                image=image_pil,
                mask_image=mask_pil,
            ).images[0]

        kwargs = {
            'negative_prompt': negative_prompt,
            'generator': generator,
            'strength': 0.9,
            'num_inference_steps': num_inference_steps,
            'height': self.cam.H,
            'width': self.cam.W,
        }

        image_np = np.round(np.clip(image, 0, 1) * 255.).astype(np.uint8)
        mask_sum = np.clip((image.prod(axis=-1) == 0) + (1 - mask_image), 0, 1)
        mask_padded = pad_mask(mask_sum, 3)
        masked = image_np * np.logical_not(mask_padded[..., None])

        if self.lama is not None:
            lama_image = Image.fromarray(self.lama(masked, mask_padded).astype(np.uint8))
        else:
            lama_image = image

        mask_image = Image.fromarray(mask_padded.astype(np.uint8) * 255)
        control_image = self.make_controlnet_inpaint_condition(lama_image, mask_image)

        return self.stable_model(
            prompt=prompt,
            image=lama_image,
            control_image=control_image,
            mask_image=mask_image,
            **kwargs,
        ).images[0]