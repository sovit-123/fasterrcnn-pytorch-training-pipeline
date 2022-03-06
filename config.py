import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 50 # number of epochs to train for
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
# Images and labels direcotry should be relative to train.py
TRAIN_DIR_IMAGES = 'data/Aquarium Combined.v2-raw-1024.voc/train'
TRAIN_DIR_LABELS = 'data/Aquarium Combined.v2-raw-1024.voc/train'
VALID_DIR_IMAGES = 'data/Aquarium Combined.v2-raw-1024.voc/valid'
VALID_DIR_LABELS = 'data/Aquarium Combined.v2-raw-1024.voc/valid'

# classes: 0 index is reserved for background
CLASSES = [
    '__background__',
    'fish', 'jellyfish', 'penguin', 'shark', 
    'puffin', 'stingray', 'starfish'
]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = 'outputs'