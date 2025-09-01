import sys
import os
import torch
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Get current file's absolute path
current_file = os.path.abspath(__file__)

# Get the directory of the current file (models folder)
current_dir = os.path.dirname(current_file)

# Get the parent directory (previous directory)
parent_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(parent_dir, 'dinov3'))

from dinov3.eval.detection.models.backbone import build_backbone
from dinov3.eval.detection.config import DetectionHeadConfig
from dinov3.eval.detection.models.position_encoding import PositionEncoding

# Relative to parent fasterrcnn directory.
REPO_DIR = 'dinov3'
# Relative to parent fasterrcnn directory or the absolute path.
WEIGHTS_URL = '/media/sovit/crucial1tb/my_data/Data_Science/projects/Computer_Vision/dinov3_exps/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'

def make_dinov3_detector_backbone():
    detection_kwargs = dict(
        layers_to_use=[2, 5, 8, 11],
        with_box_refine=True,
        two_stage=True,
        mixed_selection=True,
        look_forward_twice=True,
        k_one2many=6,
        lambda_one2many=1.0,
        num_queries_one2one=1500,
        num_queries_one2many=1500,
        reparam=True,
        position_embedding=PositionEncoding.SINE,
        num_feature_levels=4,
        dec_layers=6,
        dim_feedforward=1536,
        dropout=0.0,
        norm_type="pre_norm",
        proposal_feature_levels=4,
        proposal_min_size=50,
        decoder_type="global_rpe_decomp",
        decoder_use_checkpoint=False,
        decoder_rpe_hidden_dim=512,
        decoder_rpe_type="linear",
        blocks_to_train=None,
        add_transformer_encoder=True,
        num_encoder_layers=6,
        backbone_use_layernorm=False,
        num_classes=91,  # 91 classes in COCO
        aux_loss=True,
        topk=1500,
        hidden_dim=384,
        nheads=8,
    )

    backbone = torch.hub.load(
        REPO_DIR, 
        "dinov3_vits16", 
        source='local', 
        weights=WEIGHTS_URL
    )

    n_windows_sqrt = 0

    config = DetectionHeadConfig(**detection_kwargs)
    config.n_windows_sqrt = n_windows_sqrt
    config.proposal_in_stride = backbone.patch_size
    config.proposal_tgt_strides = [int(m * backbone.patch_size) for m in (0.5, 1, 2, 4)]

    if config.layers_to_use is None:
        # e.g. [2, 5, 8, 11] for a backbone with 12 blocks, similar to depth evaluation
        config.layers_to_use = [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]

    detector_backbone = build_backbone(backbone, config)

    return backbone

def create_model(num_classes=81, pretrained=True, coco_model=False):
    backbone = make_dinov3_detector_backbone()

    backbone.out_channels = 384

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

if __name__ == '__main__':
    from models.model_summary import summary

    model = create_model(81, pretrained=True)
    summary(model)