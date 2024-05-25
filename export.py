"""
Export to ONNX.

Requirements:
pip install onnx onnxruntime

USAGE:
python export.py --weights outputs/training/fasterrcnn_resnet18_train/best_model.pth --data data_configs/coco.yaml --out model.onnx
"""

import torch
import argparse
import yaml
import os

from models.create_fasterrcnn_model import create_model

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '--out',
        help='output model name, e.g. model.onnx',
        required=True, 
        type=str
    )
    parser.add_argument(
        '--width',
        default=640,
        type=int,
        help='onnx model input width'
    )
    parser.add_argument(
        '--height',
        default=640,
        type=int,
        help='onnx model input height'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    OUT_DIR = 'weights'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    # Load the data configurations.
    data_configs = None
    if args['data'] is not None:
        with open(args['data']) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']
    DEVICE = args['device']
    # Load weights if path provided.
    checkpoint = torch.load(args['weights'], map_location=DEVICE)
    # If config file is not given, load from model dictionary.    
    if data_configs is None:
        data_configs = True
        NUM_CLASSES = checkpoint['data']['NC']
    try:
        print('Building from model name arguments...')
        build_model = create_model[str(args['model'])]
    except:
        build_model = create_model[checkpoint['model_name']]
    model = build_model(num_classes=NUM_CLASSES, coco_model=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Input to the model
    x = torch.randn(1, 3, args['height'], args['width'])

    # Export the model
    torch.onnx.export(
        model,
        x,
        os.path.join(OUT_DIR, args['out']),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names = ['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output' : {0 : 'batch_size'}
        }
    )
    print(f"Model saved to {os.path.join(OUT_DIR, args['out'])}")

if __name__ == '__main__':
    args = parse_opt()
    main(args)