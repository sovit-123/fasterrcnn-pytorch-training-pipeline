import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 10 # number of epochs to train for
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = 'data/Chess Pieces.v23-raw.voc/train'
# validation images and XML files directory
VALID_DIR = 'data/Chess Pieces.v23-raw.voc/valid'

# classes: 0 index is reserved for background
CLASSES = ['__background__', 'bishop', 'black-bishop', 'black-king', 'black-knight', 
    'black-pawn', 'black-queen', 'black-rook', 'white-bishop', 
    'white-king', 'white-knight', 'white-pawn', 'white-queen', 
    'white-rook']

NUM_CLASSES = len(CLASSES)

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = 'outputs'