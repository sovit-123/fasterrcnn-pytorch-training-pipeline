"""
SAHI image inference with Faster RCNN pretrained models.
Only available for torchvision models.
Model Keys that can be used:
- fasterrcnn_resnet50_fpn_v2
- fasterrcnn_resnet50_fpn
- fasterrcnn_mobilenet_v3_large_fpn
- fasterrcnn_mobilenetv3_large_320_fpn
"""

import numpy as np
import cv2
import torch
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt
import pandas
import glob as glob

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import list_files

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations, convert_detections
from utils.general import set_infer_dir
from utils.transforms import resize
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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', 
        '--input', 
        help='folder path to input image (one image or a folder path)'
    )
    parser.add_argument(
        '-o', 
        '--output', 
        default=None, 
        help='folder path to output data'
    )
    parser.add_argument(
        '--data', 
        default=None, 
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', 
        '--model', 
        default=None, 
        help='name of the model'
    )
    parser.add_argument(
        '-w', 
        '--weights', 
        default=None, 
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', 
        '--threshold', 
        default=0.3, 
        type=float, 
        help='detection threshold'
    )
    parser.add_argument(
        '-si', 
        '--show', 
        action='store_true', 
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', 
        '--mpl-show', 
        dest='mpl_show', 
        action='store_true', 
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-d', 
        '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-ims', 
        '--imgsz', 
        default=None, 
        type=int, 
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-nlb', 
        '--no-labels', 
        dest='no_labels', 
        action='store_true', 
        help='do not show labels on top of bounding boxes'
    )
    parser.add_argument(
        '--square-img', 
        dest='square_img', 
        action='store_true', 
        help='whether to use square image resize, else use aspect ratio resize'
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
    parser.add_argument(
        '-t', 
        '--table', 
        dest='table', 
        action='store_true', 
        help='outputs a csv file with a table summarizing the predicted boxes'
    )
    parser.add_argument(
        '--slice-height', 
        type=int, 
        default=512, 
        help='slice height for SAHI'
    )
    parser.add_argument(
        '--slice-width', 
        type=int, 
        default=512, 
        help='slice width for SAHI'
    )
    parser.add_argument(
        '--overlap-height-ratio', 
        type=float, 
        default=0.2, 
        help='overlap height ratio for SAHI'
    )
    parser.add_argument(
        '--overlap-width-ratio', 
        type=float, 
        default=0.2, 
        help='overlap width ratio for SAHI'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    np.random.seed(42)

    data_configs = None
    if args['data'] is not None:
        with open(args['data']) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    DEVICE = args['device']
    OUT_DIR = args['output'] if args['output'] is not None else set_infer_dir()
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if args['weights'] is None:
        if data_configs is None:
            with open(os.path.join('data_configs', 'test_image_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        try:
            build_model = create_model[args['model']]
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            build_model = create_model['fasterrcnn_resnet50_fpn_v2']
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    else:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        if data_configs is None:
            data_configs = True
            NUM_CLASSES = checkpoint['data']['NC']
            CLASSES = checkpoint['data']['CLASSES']
        try:
            print('Building from model name arguments...')
            build_model = create_model[str(args['model'])]
        except:
            build_model = create_model[checkpoint['model_name']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='torchvision',
        model=model,
        confidence_threshold=args['threshold'],
        device=args['device'],
        category_mapping={str(i): CLASSES[i] for i in range(1, len(CLASSES))},
        # category_remapping={CLASSES[i]: i for i in range(1, len(CLASSES))}
    )

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    if args['input'] is None:
        DIR_TEST = data_configs['image_path']
    else:
        DIR_TEST = args['input']
    test_images = collect_all_images(DIR_TEST)
    print(f"Test instances: {len(test_images)}")

    detection_threshold = args['threshold']
    pred_boxes = {}
    box_id = 1

    if args['log_json']:
        log_json = LogJSON(os.path.join(OUT_DIR, 'log.json'))

    frame_count = 0
    total_fps = 0
    for i, image_path in enumerate(test_images):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        orig_image = cv2.imread(image_path)
        frame_height, frame_width, _ = orig_image.shape
        RESIZE_TO = args['imgsz'] if args['imgsz'] is not None else frame_width

        start_time = time.time()
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=args['slice_height'],
            slice_width=args['slice_width'],
            overlap_height_ratio=args['overlap_height_ratio'],
            overlap_width_ratio=args['overlap_width_ratio']
        )
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1

        boxes = []
        scores = []
        pred_classes = []
        for object_prediction in result.object_prediction_list:
            boxes.append(object_prediction.bbox.to_xyxy())
            scores.append(object_prediction.score.value)
            pred_classes.append(object_prediction.category.name)

        if len(boxes) > 0:
            draw_boxes = np.array(boxes)
            orig_image = inference_annotations(
                draw_boxes, 
                pred_classes, 
                scores,
                CLASSES,
                COLORS, 
                orig_image, 
                resize(orig_image, RESIZE_TO, square=args['square_img']),
                args
            )

            if args['show']:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
            if args['mpl_show']:
                plt.imshow(orig_image[:, :, ::-1])
                plt.axis('off')
                plt.show()

            if args['table']:
                for box, label in zip(draw_boxes, pred_classes):
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin
                    pred_boxes[box_id] = {
                        "image": image_name,
                        "label": str(label),
                        "xmin": xmin,
                        "xmax": xmax,
                        "ymin": ymin,
                        "ymax": ymax,
                        "width": width,
                        "height": height,
                        "area": width * height
                    }                    
                    box_id += 1

            if args['log_json']:
                log_json.update(orig_image, image_name, draw_boxes, pred_classes, CLASSES)

        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()

    if args['log_json']:
        log_json.save(os.path.join(OUT_DIR, 'log.json'))

    if args['table']:
        df = pandas.DataFrame.from_dict(pred_boxes, orient='index')
        df = df.fillna(0)
        df.to_csv(f"{OUT_DIR}/boxes.csv", index=False)
        
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    print('Path to output files: '+OUT_DIR)

if __name__ == '__main__':
    args = parse_opt()
    main(args)