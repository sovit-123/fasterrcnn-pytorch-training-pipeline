import torchinfo
import torch

def summary(model):
    # Torchvision Faster RCNN models are enclosed within a tuple ().
    if type(model) == tuple:
        model = model[0]
    device = 'cpu'
    batch_size = 4
    channels = 3
    img_height = 640
    img_width = 640
    torchinfo.summary(
        model, 
        device=device, 
        input_size=[batch_size, channels, img_height, img_width],
        row_settings=["var_names"]
    )