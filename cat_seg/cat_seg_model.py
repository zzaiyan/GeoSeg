# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import sys
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange

sys.path.insert(0, '/data1/ruizhong_data/GeoSeg/Depth-Anything-V2')
from depth_anything_v2.dinov2 import DINOv2 as DepthAnythingEncoder

@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)

        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sliding_window = sliding_window
        if clip_pretrained == "ViT-B/16": 
            self.clip_resolution = (384, 384)
            depth_pretrained = 'vitb'
            depth_path = '/data1/ruizhong_data/GeoSeg/pretrained/depth_anything_v2_vitb.pth'
            self.depth_dim = 768
            self.depth_num_layers = 12
            self.depth_layer_indexes = [3, 7]
        else: 
            self.clip_resolution = (336, 336)
            depth_pretrained = 'vitl'
            depth_path = '/data1/ruizhong_data/GeoSeg/pretrained/depth_anything_v2_vitl.pth'
            self.depth_dim = 1024
            self.depth_num_layers = 24
            self.depth_layer_indexes = [7, 15]
        self.depth_resolution = (672, 672)
        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        
        self.depth_model = DepthAnythingEncoder(model_name=depth_pretrained)
        depth_full_state = torch.load(depth_path, map_location='cpu')
        depth_encoder_state = {k.replace('pretrained.', ''): v for k, v in depth_full_state.items() if k.startswith('pretrained.')}
        self.depth_model.load_state_dict(depth_encoder_state, strict=True)
        for param in self.depth_model.parameters():
            param.requires_grad = False


        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)


        self.depth_decod_proj1 = nn.Conv2d(in_channels = self.depth_dim, out_channels=256, kernel_size=1, stride=1, padding=0) if self.depth_model else None
        self.depth_decod_proj2 = nn.ConvTranspose2d(in_channels = self.depth_dim, out_channels=128, kernel_size=2, stride=2) if self.depth_model else None

        self.depth_down_sample = nn.Conv2d(in_channels = self.depth_dim, out_channels=768, kernel_size=2, stride=2, padding=0) if self.depth_model else None
        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))


    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]

        if self.training:
            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
            clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        elif not self.sliding_window:
            with torch.no_grad():
                clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
                clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
                clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        else:
            with torch.no_grad():
                kernel = 384
                overlap = 0.333
                out_res = [640, 640]
                stride = int(kernel * (1 - overlap))
                unfold = nn.Unfold(kernel_size=kernel, stride=stride)
                fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

                image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
                image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
                global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
                image = torch.cat((image, global_image), dim=0)

                clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
                clip_images_resized = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False)

        # CLIP: 4 rotations
        clip_images_90 = torch.rot90(clip_images_resized, k=1, dims=(2, 3))
        clip_images_180 = torch.rot90(clip_images_resized, k=2, dims=(2, 3))
        clip_images_270 = torch.rot90(clip_images_resized, k=3, dims=(2, 3))

        self.layers = []
        clip_features_0 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        layers_0 = list(self.layers)

        self.layers = []
        clip_features_90 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_90, dense=True)

        self.layers = []
        clip_features_180 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_180, dense=True)

        self.layers = []
        clip_features_270 = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_270, dense=True)

        image_features = clip_features_0[:, 1:, :]
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(layers_0[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(layers_0[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        clip_guidance = {'res5': res5, 'res4': res4, 'res3': res3}

        # Depth Anything V2
        if self.training:
            depth_input = clip_images.tensor
        elif not self.sliding_window:
            depth_input = clip_images.tensor
        else:
            depth_input = clip_images

        depth_images = F.interpolate(depth_input, size=self.depth_resolution, mode='bilinear', align_corners=False)
        with torch.no_grad():
            depth_feats = self.depth_model.get_intermediate_layers(depth_images, n=self.depth_num_layers)
        depth_patch_feat = rearrange(depth_feats[-1], "B (H W) C -> B C H W", H=48)
        depth_feat_down = self.depth_down_sample(depth_patch_feat)
        depth_feat_l4 = rearrange(depth_feats[self.depth_layer_indexes[0]], "B (H W) C -> B C H W", H=48)
        depth_feat_l8 = rearrange(depth_feats[self.depth_layer_indexes[1]], "B (H W) C -> B C H W", H=48)
        depth_feat_l4_proj = self.depth_decod_proj1(depth_feat_l4) if self.depth_decod_proj1 is not None else None
        depth_feat_l8_proj = self.depth_decod_proj2(depth_feat_l8) if self.depth_decod_proj2 is not None else None
        depth_decoder_guidance = [depth_feat_l4_proj, depth_feat_l8_proj]

        outputs = self.sem_seg_head(
            [clip_features_0, clip_features_90, clip_features_180, clip_features_270],
            clip_guidance, depth_feat_down, depth_decoder_guidance
        )

        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses

        elif self.sliding_window:
            outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
            outputs = outputs.sigmoid()
            
            global_output = outputs[-1:]
            global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
            outputs = outputs[:-1]
            outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
            outputs = (outputs + global_output) / 2.

            height = batched_inputs[0].get("height", out_res[0])
            width = batched_inputs[0].get("width", out_res[1])
            output = sem_seg_postprocess(outputs[0], out_res, height, width)
            return [{'sem_seg': output}]

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results
