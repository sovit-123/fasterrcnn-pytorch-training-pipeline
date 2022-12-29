import torchvision

from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# A DarkNet model with reduced output channels for each layer.
class DarkNet(nn.Module):
    def __init__(self, initialize_weights=True, num_classes=1000):
        super(DarkNet, self).__init__()

        self.num_classes = num_classes
        self.features = self._create_conv_layers()
        self.pool = self._pool()
        self.fcs = self._create_fc_layers()

        if initialize_weights:
            # Random initialization of the weights
            # just like the original paper.
            self._initialize_weights()

    def _create_conv_layers(self):
        conv_layers = nn.Sequential(
            nn.Conv2d(3, 4, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(4, 8, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        return conv_layers

    def _pool(self):
        pool = nn.Sequential(
            nn.AvgPool2d(7),
        )
        return pool
    
    def _create_fc_layers(self):
        fc_layers = nn.Sequential(
            nn.Linear(128, self.num_classes)
        )
        return fc_layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in',
                    nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.fcs(x)
        return x

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the Mini DarkNet model features.
    backbone = DarkNet(num_classes=10).features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 128 for this custom Mini DarkNet model.
    backbone.out_channels = 128

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