import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import yaml
import random

from pathlib import Path

plt.style.use('ggplot')

def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    # if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
    #     torch.use_deterministic_algorithms(True)
    #     torch.backends.cudnn.deterministic = True
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #     os.environ['PYTHONHASHSEED'] = str(seed)

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation mAP @0.5:0.95 IoU higher than the previous highest, then save the
    model state.
    """
    def __init__(
        self, best_valid_map=float(0)
    ):
        self.best_valid_map = best_valid_map
        
    def __call__(
        self, 
        model, 
        current_valid_map, 
        epoch, 
        OUT_DIR,
        config,
        model_name
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'data': config,
                'model_name': model_name
                }, f"{OUT_DIR}/best_model.pth")

def show_tranformed_image(train_loader, device, classes, colors):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    """
    if len(train_loader) > 0:
        for i in range(2):
            images, targets = next(iter(train_loader))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            # Get all the predicited class names.
            pred_classes = [classes[i] for i in targets[i]['labels'].cpu().numpy()]
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)

            lw = max(round(sum(sample.shape) / 2 * 0.003), 2)  # Line width.
            tf = max(lw - 1, 1) # Font thickness.

            for box_num, box in enumerate(boxes):
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                class_name = pred_classes[box_num]
                color = colors[classes.index(class_name)]
                cv2.rectangle(
                    sample,
                    p1,
                    p2,
                    color, 
                    2,
                    cv2.LINE_AA
                )
                w, h = cv2.getTextSize(
                    class_name, 
                    0, 
                    fontScale=lw / 3, 
                    thickness=tf
                )[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.putText(
                    sample, 
                    class_name,
                    (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    color, 
                    2, 
                    cv2.LINE_AA
                )
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def save_loss_plot(
    OUT_DIR, 
    train_loss_list, 
    x_label='iterations',
    y_label='train loss',
    save_name='train_loss_iter'
):
    """
    Function to save both train loss graph.
    
    :param OUT_DIR: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{OUT_DIR}/{save_name}.png")
    print('SAVING PLOTS COMPLETE...')

def save_mAP(OUT_DIR, map_05, map):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.

    :param OUT_DIR: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU.
    :param map: List containing mAP values at 0.5:0.95 IoU.
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:orange', linestyle='-', 
        label='mAP@0.5'
    )
    ax.plot(
        map, color='tab:red', linestyle='-', 
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/map.png")

def visualize_mosaic_images(boxes, labels, image_resized, classes):
    print(boxes)
    print(labels)
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    for j, box in enumerate(boxes):
        color = (0, 255, 0)
        classn = labels[j]
        cv2.rectangle(image_resized,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 2)
        cv2.putText(image_resized, classes[classn], 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
    cv2.imshow('Mosaic', image_resized)
    cv2.waitKey(0)

def save_model(
    epoch, 
    model, 
    optimizer, 
    train_loss_list,
    train_loss_list_epoch, 
    val_map,
    val_map_05,
    OUT_DIR,
    config,
    model_name
):
    """
    Function to save the trained model till current epoch, or whenever called.
    Saves many other dictionaries and parameters as well helpful to resume training.
    May be larger in size.

    :param epoch: The epoch number.
    :param model: The neural network model.
    :param optimizer: The optimizer.
    :param optimizer: The train loss history.
    :param train_loss_list_epoch: List containing loss for each epoch.
    :param val_map: mAP for IoU 0.5:0.95.
    :param val_map_05: mAP for IoU 0.5.
    :param OUT_DIR: Output directory to save the model.
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
                'train_loss_list_epoch': train_loss_list_epoch,
                'val_map': val_map,
                'val_map_05': val_map_05,
                'data': config,
                'model_name': model_name
                }, f"{OUT_DIR}/last_model.pth")

def save_model_state(model, OUT_DIR, config, model_name):
    """
    Saves the model state dictionary only. Has a smaller size compared 
    to the the saved model with all other parameters and dictionaries.
    Preferable for inference and sharing.

    :param model: The neural network model.
    :param OUT_DIR: Output directory to save the model.
    """
    torch.save({
                'model_state_dict': model.state_dict(),
                'data': config,
                'model_name': model_name
                }, f"{OUT_DIR}/last_model_state.pth")

def denormalize(x, mean=None, std=None):
    # Shape of x here should be [B, 3, H, W].
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    # Returns tensor of shape [B, 3, H, W].
    return torch.clamp(t, 0, 1)

def save_validation_results(images, detections, counter, out_dir, classes, colors):
    """
    Function to save validation results.
    :param images: All the images from the current batch.
    :param detections: All the detection results.
    :param counter: Step counter for saving with unique ID.
    """
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    image_list = [] # List to store predicted images to return.
    for i, detection in enumerate(detections):
        image_c = images[i].clone()
        image_c = image_c.detach().cpu().numpy().astype(np.float32)
        image = np.transpose(image_c, (1, 2, 0))

        image = np.ascontiguousarray(image, dtype=np.float32)

        scores = detection['scores'].cpu().numpy()
        labels = detection['labels']
        bboxes = detection['boxes'].detach().cpu().numpy()
        boxes = bboxes[scores >= 0.5].astype(np.int32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Get all the predicited class names.
        pred_classes = [classes[i] for i in labels.cpu().numpy()]
        for j, box in enumerate(boxes):
            class_name = pred_classes[j]
            color = colors[classes.index(class_name)]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2, lineType=cv2.LINE_AA
            )
            cv2.putText(image, class_name, 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
        cv2.imwrite(f"{out_dir}/image_{i}_{counter}.jpg", image*255.)
        image_list.append(image[:, :, ::-1])
    return image_list

def set_infer_dir():
    """
    This functions counts the number of inference directories already present
    and creates a new one in `outputs/inference/`. 
    And returns the directory path.
    """
    if not os.path.exists('outputs/inference'):
        os.makedirs('outputs/inference')
    num_infer_dirs_present = len(os.listdir('outputs/inference/'))
    next_dir_num = num_infer_dirs_present + 1
    new_dir_name = f"outputs/inference/res_{next_dir_num}"
    os.makedirs(new_dir_name, exist_ok=True)
    return new_dir_name

def set_training_dir(dir_name=None, project_dir=None):
    """
    This functions counts the number of training directories already present
    and creates a new one in `outputs/training/`. 
    And returns the directory path.
    """
    if project_dir != None:
        os.makedirs(project_dir, exist_ok=True)
        return project_dir
    if not os.path.exists('outputs/training'):
        os.makedirs('outputs/training')
    if dir_name:
        new_dir_name = f"outputs/training/{dir_name}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
    else:
        num_train_dirs_present = len(os.listdir('outputs/training/'))
        next_dir_num = num_train_dirs_present + 1
        new_dir_name = f"outputs/training/res_{next_dir_num}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name

def yaml_save(file_path=None, data={}):
    with open(file_path, 'w') as f:
        yaml.safe_dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, 
            f, 
            sort_keys=False
        )

class EarlyStopping():
    """
    Early stopping to stop the training when the mAP does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping mAP 
                is not improving
        :param min_delta: minimum difference between new mAP and old mAP for
               new mAP to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_map = None
        self.early_stop = False

    def __call__(self, map):
        if self.best_map == None:
            self.best_map = map
        elif map - self.best_map > self.min_delta:
            self.best_map = map
            # reset counter if validation loss improves
            self.counter = 0
        elif map - self.best_map < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True