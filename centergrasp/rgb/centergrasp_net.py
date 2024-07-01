import torch

from centergrasp.rgb.pose_utils import pose_image_procrustes
from simnet.lib.net.models.basic_stem import RGBDStem
from simnet.lib.net.models.panoptic_net import DepthHead
from simnet.lib.net.models.panoptic_backbone import (
    ShapeSpec,
    output_shape,
    build_resnet_fpn_backbone,
    SemSegFPNHead,
    PoseFPNHead,
)


class CenterGraspNet(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        assert hparams.img_width % 32 == 0, "Image width must be a multiple of 32"
        assert hparams.img_height % 32 == 0, "Image height must be a multiple of 32"
        input_shape = ShapeSpec(channels=3, height=hparams.img_height, width=hparams.img_width)
        stereo_stem = RGBDStem(hparams)
        self.backbone = build_resnet_fpn_backbone(
            input_shape,
            stereo_stem,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )
        backbone_output_shape = output_shape(self.backbone)
        self.depth_head = DepthHead(backbone_output_shape, hparams)
        # Add heatmap head. Only 1 class (object)
        self.heatmap_head = SemSegFPNHead(
            backbone_output_shape,
            num_classes=1,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )
        self.latent_embedding_head = PoseFPNHead(
            backbone_output_shape,
            num_classes=32,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )
        self.abs_pose_head = PoseFPNHead(
            backbone_output_shape,
            num_classes=12,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )
        return

    def forward(self, image):
        features, small_disp_output = self.backbone.forward(image)
        heatmap_output = self.heatmap_head.forward(features).squeeze(dim=1)
        heatmap_output = torch.sigmoid(heatmap_output)
        latent_emb_output = self.latent_embedding_head.forward(features)
        abs_pose_output = self.abs_pose_head.forward(features)
        abs_pose_output = pose_image_procrustes(abs_pose_output)
        return heatmap_output, abs_pose_output, latent_emb_output
