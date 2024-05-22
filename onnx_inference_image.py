"""
Script to run inference on images using ONNX models.
`--input` can take the path either an image or a directory containing images.

USAGE:
python onnx_inference_image.py --input ../inference_data/ --weights weights/fasterrcnn_resnet18.onnx --data data_configs/voc.yaml --show --imgsz 640
"""

import torch
import onnxruntime
import cv2
import numpy as np
import os
import glob
import argparse
import yaml
import time
import matplotlib.pyplot as plt

from utils.transforms import infer_transforms, resize
from utils.general import set_infer_dir
from utils.annotations import (
    inference_annotations, convert_detections
)
from utils.logging import LogJSON

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-w', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', 
        default=0.3, 
        type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show',  
        action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', 
        dest='mpl_show', 
        action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=640,
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-nlb', '--no-labels',
        dest='no_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        default=None,
        help='filter classes by visualization, --classes 1 2 3'
    )
    parser.add_argument(
        '--track',
        action='store_true'
    )
    parser.add_argument(
        '--log-json',
        dest='log_json',
        action='store_true',
        help='store a json log file in COCO format in the output directory'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    np.random.seed(42)
    # Load model.
    ort_session = onnxruntime.InferenceSession(
        args['weights'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    OUT_DIR = set_infer_dir()
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    if args['input'] == None:
        DIR_TEST = data_configs['image_path']
        test_images = collect_all_images(DIR_TEST)
    else:
        DIR_TEST = args['input']
        test_images = collect_all_images(DIR_TEST)
    print(f"Test instances: {len(test_images)}")

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    if args['log_json']:
        log_json = LogJSON(os.path.join(OUT_DIR, 'log.json'))

    # To count the total number of frames iterated through.
    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0
    for i in range(len(test_images)):
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        orig_image = cv2.imread(test_images[i])
        frame_height, frame_width, _ = orig_image.shape
        if args['imgsz'] != None:
            RESIZE_TO = args['imgsz']
        else:
            RESIZE_TO = frame_width
        # orig_image = image.copy()
        image_resized = resize(orig_image, RESIZE_TO, square=True)
        image = image_resized.copy()
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        print(image.shape)
        start_time = time.time()
        preds = ort_session.run(
            None, {ort_session.get_inputs()[0].name: to_numpy(image)}
        )
        end_time = time.time()
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        outputs = {}
        outputs['boxes'] = torch.tensor(preds[0])
        outputs['labels'] = torch.tensor(preds[1])
        outputs['scores'] = torch.tensor(preds[2])
        outputs = [outputs]

        # Log to JSON?
        if args['log_json']:
            log_json.update(orig_image, image_name, outputs[0], CLASSES)

        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            draw_boxes, pred_classes, scores = convert_detections(
                outputs, detection_threshold, CLASSES, args
            )
            orig_image = inference_annotations(
                draw_boxes, 
                pred_classes, 
                scores,
                CLASSES,
                COLORS, 
                orig_image, 
                image_resized,
                args
            )
            if args['show']:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
            if args['mpl_show']:
                plt.imshow(orig_image[:, :, ::-1])
                plt.axis('off')
                plt.show()
        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()

    # Save JSON log file.
    if args['log_json']:
        log_json.save(os.path.join(OUT_DIR, 'log.json'))

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

if __name__ == '__main__':
    args = parse_opt()
    main(args)