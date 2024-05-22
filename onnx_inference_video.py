"""
Script to run inference on videos using ONNX model.
`--input` takes the path to a video.

USAGE:
python onnx_inference_video.py --input ../inference_data/video_4_trimmed_1.mp4 --weights weights/fasterrcnn_resnet18.onnx --data data_configs/voc.yaml --show --imgsz 640
"""

import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import onnxruntime

from utils.general import set_infer_dir
from utils.annotations import (
    inference_annotations, 
    annotate_fps, 
    convert_detections,
    convert_pre_track,
    convert_post_track
)
from utils.transforms import infer_transforms, resize
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.logging import LogJSON

def read_return_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the video's frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
    return cap, frame_width, frame_height

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def parse_opt():
        # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='path to input video',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default=None,
        help='name of the model'
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
        default=None,
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
    if args['track']: # Initialize Deep SORT tracker if tracker is selected.
        tracker = DeepSort(max_age=30)
    # Load model.
    ort_session = onnxruntime.InferenceSession(
        args['weights'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    OUT_DIR = set_infer_dir()
    VIDEO_PATH = None
    if args['input'] == None:
        VIDEO_PATH = data_configs['video_path']
    else:
        VIDEO_PATH = args['input']
    assert VIDEO_PATH is not None, 'Please provide path to an input video...'
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    cap, frame_width, frame_height = read_return_video_data(VIDEO_PATH)

    save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))
    if args['imgsz'] != None:
        RESIZE_TO = args['imgsz']
    else:
        RESIZE_TO = frame_width

    if args['log_json']:
        log_json = LogJSON(os.path.join(OUT_DIR, 'log.json'))

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret:
            orig_frame = frame.copy()
            frame = resize(frame, RESIZE_TO, square=True)
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            # Add batch dimension.
            image = torch.unsqueeze(image, 0)
            # Get the start time.
            start_time = time.time()
            preds = ort_session.run(
                None, {ort_session.get_inputs()[0].name: to_numpy(image)}
            )
            forward_end_time = time.time()

            forward_pass_time = forward_end_time - start_time
            # Get the current fps.
            fps = 1 / (forward_pass_time)
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
                log_json.update(frame, save_name, outputs[0], CLASSES)

            # Carry further only if there are detected boxes.
            if len(outputs[0]['boxes']) != 0:
                draw_boxes, pred_classes, scores = convert_detections(
                    outputs, detection_threshold, CLASSES, args
                )
                if args['track']:
                    tracker_inputs = convert_pre_track(
                        draw_boxes, pred_classes, scores
                    )
                    # Update tracker with detections.
                    tracks = tracker.update_tracks(tracker_inputs, frame=frame)
                    draw_boxes, pred_classes, scores = convert_post_track(tracks) 
                frame = inference_annotations(
                    draw_boxes, 
                    pred_classes, 
                    scores,
                    CLASSES, 
                    COLORS, 
                    orig_frame, 
                    frame,
                    args
                )
            else:
                frame = orig_frame
            frame = annotate_fps(frame, fps)

            final_end_time = time.time()
            forward_and_annot_time = final_end_time - start_time
            print_string = f"Frame: {frame_count}, Forward pass FPS: {fps:.3f}, "
            print_string += f"Forward pass time: {forward_pass_time:.3f} seconds, "
            print_string += f"Forward pass + annotation time: {forward_and_annot_time:.3f} seconds"
            print(print_string)            
            out.write(frame)
            if args['show']:
                cv2.imshow('Prediction', frame)
                # Press `q` to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            break

    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
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