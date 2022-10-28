import torchinfo

def summary(model):
    batch_size = 4
    channels = 3
    img_height = 650
    img_width = 640
    torchinfo.summary(
        model, input_size=(batch_size, channels, img_height, img_width)
    )