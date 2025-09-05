import sys
import os
import torch
import torchvision
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from models.model_summary import summary

# Get current file's absolute path
current_file = os.path.abspath(__file__)

# Get the directory of the current file (models folder)
current_dir = os.path.dirname(current_file)

# Get the parent directory (previous directory)
parent_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(parent_dir, 'dinov3'))

# Relative to parent fasterrcnn directory.
REPO_DIR = 'dinov3'
# Relative to parent fasterrcnn directory or the absolute path.
WEIGHTS_URL = 'weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'

class Dinov3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            REPO_DIR, 
            'dinov3_convnext_small', 
            source='local', 
            weights=WEIGHTS_URL
        )
        
    def forward(self, x):
        out = self.backbone.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=False, norm=True
        )
        return out[0]
    

def create_model(num_classes=81, pretrained=True, coco_model=False):
    backbone = Dinov3Backbone()

    backbone.out_channels = 768

    for name, params in backbone.named_parameters():
        params.requires_grad_(False)

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
    
    random_tensor = torch.randn(1, 3, 640, 640)

    _ = model.eval()

    summary(model)