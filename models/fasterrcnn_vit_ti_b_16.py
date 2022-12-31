import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from vision_transformers.models import vit

class Vit_Ti_P16_224(nn.Module):
    def __init__(self):
        super(Vit_Ti_P16_224, self).__init__()

        # The pretrained ViT backbone.
        self.backbone = vit.vit_ti_p16_224(pretrained=True)
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # The following Conv2d has 
        # in_channels = previous out_channels, hidden_dim, 1
        self.conv = nn.Conv2d(256, 192, 1)
    
    def forward(self, inputs):
        x = self.seq1(inputs)
        h = self.conv(x)
        x = self.backbone.transformer(h.flatten(2).permute(0, 2, 1))
        bs, _, _ = x.shape
        x = x.view(bs, 192, 48, -1)
        # x = x.view(bs, 192, 3, -1)
        return x

def create_model(num_classes, pretrained=True, coco_model=False):
    backbone = Vit_Ti_P16_224()

    backbone.out_channels = 192

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