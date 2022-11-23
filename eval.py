"""
Run evaluation on a trained model to get mAP and class wise AP.

USAGE:
python eval.py --config data_configs/voc.yaml --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small
"""
from datasets import (
    create_valid_dataset, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from torch_utils import utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from tqdm import tqdm

import torch
import argparse
import yaml
import torchvision
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', 
        default='data_configs/test_image_config.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', default='fasterrcnn_resnet50_fpn',
        help='name of the model'
    )
    parser.add_argument(
        '-mw', '--weights', default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-ims', '--img-size', dest='img_size', default=640, type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-w', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch-size', dest='batch_size', default=8, type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    args = vars(parser.parse_args())

    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)

    # Validation settings and constants.
    try: # Use test images if present.
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    except: # Else use the validation images.
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch_size']
    # Load the pretrained model
    create_model = create_model[args['model']]
    if args['weights'] is None:
        model = create_model(num_classes=NUM_CLASSES, coco_model=True)

    # Load weights.
    if args['weights'] is not None:
        model = create_model(num_classes=NUM_CLASSES, coco_model=False)
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    # Model configurations
    IMAGE_WIDTH = args['img_size']
    IMAGE_HEIGHT = args['img_size']
    
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, VALID_DIR_LABELS, 
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)

    @torch.inference_mode()
    def evaluate(
        model, 
        data_loader, 
        device, 
        out_dir=None,
        classes=None,
        colors=None
    ):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        target = []
        preds = []
        counter = 0
        for images, targets in tqdm(metric_logger.log_every(data_loader, 100, header), total=len(data_loader)):
            counter += 1
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            with torch.no_grad():
                outputs = model(images)

            #####################################
            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                target.append(true_dict)
            #####################################

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        torch.set_num_threads(n_threads)
        metric = MeanAveragePrecision(class_metrics=True)
        metric.update(preds, target)
        metric_summary = metric.compute()
        return metric_summary

    stats = evaluate(
        model, 
        valid_loader, 
        device=DEVICE,
        classes=CLASSES,
    )

    print('\n')
    pprint(stats)
    print('\n')
    pprint(f"Classes: {CLASSES}")
    print('\n')
    print('AP per class')
    empty_string = ''
    if len(CLASSES) > 2: 
        print(len(stats['map_per_class']))
        num_hyphens = 51
        print('-'*num_hyphens)
        print(f"|     Class{empty_string:<16} | AP{empty_string:<18}|")
        print('-'*num_hyphens)
        class_counter = 0
        for i in range(0, len(CLASSES)-1, 1):
            class_counter += 1
            print(f"|{class_counter:<3} | {CLASSES[i+1]:<20} | {np.array(stats['map_per_class'][i]):.3f}{empty_string:<15}|")
        print('-'*num_hyphens)
        print(f"|mAP{empty_string:<23} | {np.array(stats['map']):.3f}{empty_string:<15}|")
    else:
        print('-'*40)
        print(f"|Class{empty_string:<10} | AP{empty_string:<18}|")
        print('-'*40)
        print(f"|{CLASSES[1]:<15} | {np.array(stats['map']):.3f}{empty_string:<15}|")
        print('-'*40)
        print(f"|mAP{empty_string:<12} | {np.array(stats['map']):.3f}{empty_string:<15}|")