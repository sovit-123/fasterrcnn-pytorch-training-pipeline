import logging
import os
import pandas as pd
import wandb
import cv2
import time

from torch.utils.tensorboard.writer import SummaryWriter

# Initialize Weights and Biases.
wandb.init()

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

def tensorboard_loss_log(name, loss_np_arr, writer):
    """
    To plot graphs for TensorBoard log. The save directory for this
    is the same as the training result save directory.
    """
    for i in range(len(loss_np_arr)):
        writer.add_scalar(name, loss_np_arr[i], i)

def tensorboard_map_log(name, val_map_05, val_map, writer):
    for i in range(len(val_map)):
        writer.add_scalars(
            name,
            {
                'mAP@0.5': val_map_05[i], 
                'mAP@0.5_0.95': val_map[i]
            },
            i
        )

def create_log_csv(log_dir):
    cols = ['epoch', 'map', 'map_05']
    results_csv = pd.DataFrame(columns=cols)
    results_csv.to_csv(os.path.join(log_dir, 'results.csv'), index=False)

def csv_log(log_dir, stats, epoch):
    if epoch+1 == 1:
        create_log_csv(log_dir) 
    
    df = pd.DataFrame(
        {
            'epoch': int(epoch+1),
            'map_05': [float(stats[0])],
            'map': [float(stats[1])],
        }
    )
    df.to_csv(
        os.path.join(log_dir, 'results.csv'), 
        mode='a', 
        index=False, 
        header=False
    )

def wandb_log(
    epoch_loss, 
    loss_list_batch, 
    val_map_05, 
    val_map,
    val_pred_image
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
            {'train_loss_iter': loss_list_batch[i]},
        )
    wandb.log(
        {'train_loss_epoch': epoch_loss},
    )
    wandb.log(
        {'val_map_05_95': val_map}
    )
    wandb.log(
        {'val_map_05': val_map_05}
    )

    # for i, image in enumerate(val_pred_image):
        # wandb.log({'img'+str(i): [wandb.Image(image)]})

    if len(val_pred_image) == 2:
        log_image = cv2.hconcat([val_pred_image[0], val_pred_image[1]])
        wandb.log({'predictions': [wandb.Image(log_image)]})

    if len(val_pred_image) > 2 and len(val_pred_image) <= 8:
        log_image = val_pred_image[0]
        for i in range(len(val_pred_image)-1):
            log_image = cv2.hconcat([log_image, val_pred_image[i+1]])
        wandb.log({'predictions': [wandb.Image(log_image)]})
    
    if len(val_pred_image) > 8:
        log_image = val_pred_image[0]
        for i in range(len(val_pred_image)-1):
            if i == 7:
                break
            log_image = cv2.hconcat([log_image, val_pred_image[i-1]])
        wandb.log({'predictions': [wandb.Image(log_image)]})