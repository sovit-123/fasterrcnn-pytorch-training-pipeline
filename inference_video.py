import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt

from models.create_fasterrcnn_model import create_model
from custom_utils import set_infer_dir
from torchvision import transforms as transforms

def read_return_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the video's frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
    return cap, frame_width, frame_height

def transform_image(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='path to input video',
    )
    parser.add_argument(
        '-c', '--config', 
        default='data_configs/test_video_config.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', default='fasterrcnn_resnet50',
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', default=0.3, type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show-image', dest='show_image', action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', dest='mpl_show', action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    args = vars(parser.parse_args())

    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)

    # Inference settings and constants.
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    OUT_DIR = set_infer_dir()
    VIDEO_PATH = None
    if args['input'] == None:
        VIDEO_PATH = data_configs['video_path']
    else:
        VIDEO_PATH = args['input']
    assert VIDEO_PATH is not None, 'Please provide path to an input video...'

    # Load the pretrained model
    create_model = create_model[args['model']]
    if args['weights'] is None:
        model = create_model(num_classes=NUM_CLASSES, coco_model=True)
    # Load weights if path provided.
    if args['weights'] is not None:
        model = create_model(num_classes=NUM_CLASSES, coco_model=False)
        checkpoint = torch.load(args['weights'])
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    cap, frame_width, frame_height = read_return_video_data(VIDEO_PATH)

    save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))
    RESIZE_TO = (frame_width, frame_height)

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, RESIZE_TO)
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform_image(image)
            # Add batch dimension.
            image = torch.unsqueeze(image, 0)
            # Get the start time.
            start_time = time.time()
            with torch.no_grad():
                # Get predictions for the current frame.
                outputs = model(image.to(DEVICE))
            forward_end_time = time.time()

            forward_pass_time = forward_end_time - start_time
            
            # Get the current fps.
            fps = 1 / (forward_pass_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1
            
            # Load all detection to CPU for further operations.
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            # Carry further only if there are detected boxes.
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                # Filter out boxes according to `detection_threshold`.
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # Get all the predicited class names.
                pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
                
                # dDaw the bounding boxes and write the class name on top of it.
                for j, box in enumerate(draw_boxes):
                    class_name = pred_classes[j]
                    color = COLORS[CLASSES.index(class_name)]
                    cv2.rectangle(frame,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                color, 2)
                    cv2.putText(frame, class_name, 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                                2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f"{fps:.1f} FPS", 
                        (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)

            final_end_time = time.time()
            forward_and_annot_time = final_end_time - start_time
            print_string = f"Frame: {frame_count}, Forward pass FPS: {fps:.3f}, "
            print_string += f"Forward pass time: {forward_pass_time:.3f} seconds, "
            print_string += f"Forward pass + annotation time: {forward_and_annot_time:.3f} seconds"
            print(print_string)            
            out.write(frame)
            if args['show_image']:
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

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")