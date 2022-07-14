from models import *

def return_fasterrcnn_resnet50_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_mobilenetv3_large_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_mobilenetv3_large_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_mobilenetv3_large_320_fpn(
    num_classes, pretrained=True, coco_model=False
):    
    model = fasterrcnn_mobilenetv3_large_320_fpn.create_model(
        num_classes
    )
    return model

def return_fasterrcnn_resnet50(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_resnet18(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet18.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_custom_resnet(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_custom_resnet.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_darknet(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_darknet.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_squeezenet1_0(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_squeezenet1_0.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_squeezenet1_1(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_squeezenet1_1.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_mini_darknet(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_mini_darknet.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_mini_darknet_mini_head(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_mini_darknet_mini_head.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_squeezenet1_1_mini_head(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_squeezenet1_1_mini_head.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_mini_squeezenet1_1_mini_head(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_mini_squeezenet1_1_mini_head.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_mini_squeezenet1_1_tiny_head(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_mini_squeezenet1_1_tiny_head.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_mbv3_small_nano_head(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_mbv3_small_nano_head.create_model(
        num_classes, pretrained, coco_model
    )
    return model

create_model = {
    'fasterrcnn_resnet50_fpn': return_fasterrcnn_resnet50_fpn,
    'fasterrcnn_mobilenetv3_large_fpn': return_fasterrcnn_mobilenetv3_large_fpn,
    'fasterrcnn_mobilenetv3_large_320_fpn': return_fasterrcnn_mobilenetv3_large_320_fpn,
    'fasterrcnn_resnet50': return_fasterrcnn_resnet50,
    'fasterrcnn_resnet18': return_fasterrcnn_resnet18,
    'fasterrcnn_custom_resnet': return_fasterrcnn_custom_resnet,
    'fasterrcnn_darknet': return_fasterrcnn_darknet,
    'fasterrcnn_squeezenet1_0': return_fasterrcnn_squeezenet1_0,
    'fasterrcnn_squeezenet1_1': return_fasterrcnn_squeezenet1_1,
    'fasterrcnn_mini_darknet': return_fasterrcnn_mini_darknet,
    'fasterrcnn_mini_darknet_mini_head': return_fasterrcnn_mini_darknet_mini_head,
    'fasterrcnn_squeezenet1_1_mini_head': return_fasterrcnn_squeezenet1_1_mini_head,
    'fasterrcnn_mini_squeezenet1_1_mini_head': return_fasterrcnn_mini_squeezenet1_1_mini_head,
    'fasterrcnn_mini_squeezenet1_1_tiny_head': return_fasterrcnn_mini_squeezenet1_1_tiny_head,
    'fasterrcnn_mbv3_small_nano_head': return_fasterrcnn_mbv3_small_nano_head
}