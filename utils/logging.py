import logging
import os
import pandas as pd
import wandb
import cv2
import numpy as np
import json

from torch.utils.tensorboard.writer import SummaryWriter

# Initialize Weights and Biases.
def wandb_init(name):
    wandb.init(name=name)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def set_log(log_dir):
    logging.basicConfig(
        # level=logging.DEBUG,
        format='%(message)s',
        # datefmt='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

def log(content, *args):
    for arg in args:
        content += str(arg)
    logger.info(content)

def coco_log(log_dir, stats):
    log_dict_keys = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    ]
    log_dict = {}
    # for i, key in enumerate(log_dict_keys):
    #     log_dict[key] = stats[i]

    with open(f"{log_dir}/train.log", 'a+') as f:
        f.writelines('\n')
        for i, key in enumerate(log_dict_keys):
            out_str = f"{key} = {stats[i]}"
            logger.debug(out_str) # DEBUG model so as not to print on console.
        logger.debug('\n'*2) # DEBUG model so as not to print on console.
    # f.close()

def set_summary_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def tensorboard_loss_log(name, loss_np_arr, writer, epoch):
    """
    To plot graphs for TensorBoard log. The save directory for this
    is the same as the training result save directory.
    """
    writer.add_scalar(name, loss_np_arr[-1], epoch)

def tensorboard_map_log(name, val_map_05, val_map, writer, epoch):
    writer.add_scalars(
        name,
        {
            'mAP@0.5': val_map_05[-1], 
            'mAP@0.5_0.95': val_map[-1]
        },
        epoch
    )

def create_log_csv(log_dir):
    cols = [
        'epoch', 
        'map', 
        'map_05',
        'train loss',
        'train cls loss',
        'train box reg loss',
        'train obj loss',
        'train rpn loss'
    ]
    results_csv = pd.DataFrame(columns=cols)
    results_csv.to_csv(os.path.join(log_dir, 'results.csv'), index=False)

def csv_log(
    log_dir, 
    stats, 
    epoch,
    train_loss_list,
    loss_cls_list,
    loss_box_reg_list,
    loss_objectness_list,
    loss_rpn_list
):
    if epoch+1 == 1:
        create_log_csv(log_dir) 
    
    df = pd.DataFrame(
        {
            'epoch': int(epoch+1),
            'map_05': [float(stats[0])],
            'map': [float(stats[1])],
            'train loss': train_loss_list[-1],
            'train cls loss': loss_cls_list[-1],
            'train box reg loss': loss_box_reg_list[-1],
            'train obj loss': loss_objectness_list[-1],
            'train rpn loss': loss_rpn_list[-1]
        }
    )
    df.to_csv(
        os.path.join(log_dir, 'results.csv'), 
        mode='a', 
        index=False, 
        header=False
    )

def overlay_on_canvas(bg, image):
    bg_copy = bg.copy()
    h, w = bg.shape[:2]
    h1, w1 = image.shape[:2]
    # Center of canvas (background).
    cx, cy = (h - h1) // 2, (w - w1) // 2
    bg_copy[cy:cy + h1, cx:cx + w1] = image
    return bg_copy * 255.

def wandb_log(
    epoch_loss, 
    loss_list_batch,
    loss_cls_list,
    loss_box_reg_list,
    loss_objectness_list,
    loss_rpn_list,
    val_map_05, 
    val_map,
    val_pred_image,
    image_size
):
    """
    :param epoch_loss: Single loss value for the current epoch.
    :param batch_loss_list: List containing loss values for the current 
        epoch's loss value for each batch.
    :param val_map_05: Current epochs validation mAP@0.5 IoU.
    :param val_map: Current epochs validation mAP@0.5:0.95 IoU. 
    """
    # WandB logging.
    for i in range(len(loss_list_batch)):
        wandb.log(
            {'train_loss_iter': loss_list_batch[i],},
        )
    # for i in range(len(loss_cls_list)):
    wandb.log(
        {
            'train_loss_cls': loss_cls_list[-1],
            'train_loss_box_reg': loss_box_reg_list[-1],
            'train_loss_obj': loss_objectness_list[-1],
            'train_loss_rpn': loss_rpn_list[-1]
        }
    )
    wandb.log(
        {
            'train_loss_epoch': epoch_loss
        },
    )
    wandb.log(
        {'val_map_05_95': val_map}
    )
    wandb.log(
        {'val_map_05': val_map_05}
    )

    bg = np.full((image_size * 2, image_size * 2, 3), 114, dtype=np.float32)

    if len(val_pred_image) == 1:
        log_image = overlay_on_canvas(bg, val_pred_image[0])
        wandb.log({'predictions': [wandb.Image(log_image)]})

    if len(val_pred_image) == 2:
        log_image = cv2.hconcat(
            [
                overlay_on_canvas(bg, val_pred_image[0]), 
                overlay_on_canvas(bg, val_pred_image[1])
            ]
        )
        wandb.log({'predictions': [wandb.Image(log_image)]})

    if len(val_pred_image) > 2 and len(val_pred_image) <= 8:
        log_image = overlay_on_canvas(bg, val_pred_image[0])
        for i in range(len(val_pred_image)-1):
            log_image = cv2.hconcat([
                log_image, 
                overlay_on_canvas(bg, val_pred_image[i+1])
            ])
        wandb.log({'predictions': [wandb.Image(log_image)]})
    
    if len(val_pred_image) > 8:
        log_image = overlay_on_canvas(bg, val_pred_image[0])
        for i in range(len(val_pred_image)-1):
            if i == 7:
                break
            log_image = cv2.hconcat([
                log_image, 
                overlay_on_canvas(bg, val_pred_image[i-1])
            ])
        wandb.log({'predictions': [wandb.Image(log_image)]})

def wandb_save_model(model_dir):
    """
    Uploads the models to Weights&Biases.

    :param model_dir: Local disk path where models are saved.
    """
    wandb.save(os.path.join(model_dir, 'best_model.pth'))

class LogJSON():
    def __init__(self, output_filename):
        """
        :param output_filename: Path where the JSOn file should be saved.
        """
        if not os.path.exists(output_filename):
        # Initialize file with basic structure if it doesn't exist
            with open(output_filename, 'w') as file:
                json.dump({"images": [], "annotations": [], "categories": []}, file, indent=4)

        with open(output_filename, 'r') as file:
            self.coco_data = json.load(file)

        self.annotations = self.coco_data['annotations']
        self.images = self.coco_data['images']
        self.categories = set(cat['id'] for cat in self.coco_data['categories'])
        self.annotation_id = max([ann['id'] for ann in self.annotations], default=0) + 1
        self.image_id = len(self.images) + 1

    def update(self, image, file_name, boxes, labels, classes):
        """
        Update the log file metrics with the current image or current frame information.

        :param image: The original image/frame.
        :param file_name: image file name.
        :param output: Model outputs.
        :param classes: classes in the model.
        """
        image_info = {
            "file_name": file_name, "width": image.shape[1], "height": image.shape[0]
        }

        # Add image entry
        self.images.append({
            "id": self.image_id, 
            "file_name": image_info['file_name'], 
            "width": image_info['width'], 
            "height": image_info['height']
        })

        boxes = np.array(boxes, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64)

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "category_id": label,
                "iscrowd": 0
            }
            self.annotations.append(annotation)
            self.annotation_id += 1
            self.categories.add(int(label))

        # Update categories
        self.coco_data['categories'] = [{"id": cat_id, "name": classes[cat_id]} for cat_id in self.categories]

    def save(self, output_filename):
        """
        :param output_filename: Path where the JSOn file should be saved.
        """
        with open(output_filename, 'w') as file:
            json.dump(self.coco_data, file, indent=4)