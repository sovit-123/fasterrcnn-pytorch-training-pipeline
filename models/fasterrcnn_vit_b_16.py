import torchvision
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from vision_transformers.models import vit

class Vit_B_P16_224(nn.Module):
    def __init__(self):
        super(Vit_B_P16_224, self).__init__()

        self.backbone = vit.vit_b_p16_224(pretrained=True)
        self.layers = self.get_layers()

    def get_layers(self):
        self.layers = nn.Sequential(
            self.backbone.patches,
            self.backbone.transformer
        )
        return self.layers
    
    def forward(self, x):
        x = self.layers(x)
        bs, _, _ = x.shape
        x = x.view(bs, 768, 50, -1)
        return x

def create_model(num_classes, pretrained=True, coco_model=False):
    backbone = Vit_B_P16_224()

    backbone.out_channels = 768

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
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)
    # print(model)