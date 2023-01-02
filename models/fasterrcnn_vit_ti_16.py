import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from vision_transformers.models import vit

class Vit_Ti_P16_224(nn.Module):
    def __init__(self):
        super(Vit_Ti_P16_224, self).__init__()

        hidden_dim = 192

        # The pretrained ViT backbone.
        self.backbone = vit.vit_ti_p16_224(pretrained=True)
        self.seq1 = nn.Sequential(*list(torchvision.models.resnet50(weights='DEFAULT').children())[:-2])
        # The following Conv2d has 
        # in_channels = previous out_channels, hidden_dim, 1
        self.conv = nn.Conv2d(2048, 192, 1)

        self.decoder_layer = nn.TransformerDecoderLayer(192, 8)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, 6
        )

        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))


    def forward(self, inputs):
        x = self.seq1(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        x = self.backbone.transformer(pos + h.flatten(2).permute(2, 0, 1))
        x = self.decoder(x, self.query_pos.unsqueeze(1)).transpose(0, 1)
        bs, _, _ = x.shape
        x = x.view(bs, 192, 25, -1)
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