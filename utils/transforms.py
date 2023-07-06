import albumentations as A
import numpy as np
import cv2

from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms

def resize(im, img_size=640, square=False):
    # Aspect ratio resize
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im

# Define the training tranforms
def get_train_aug():
    return A.Compose([
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.ToGray(p=0.1),
        A.RandomBrightnessContrast(p=0.1),
        A.ColorJitter(p=0.1),
        A.RandomGamma(p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels'],
    })
def get_train_aug_custom(augmentation_choices):
    transform_list = []
    for key in augmentation_choices.keys():
        if key == 'blur':
            transform_list.append(A.Blur(p=augmentation_choices[key]['p'],blur_limit=augmentation_choices[key]['blur_limit']))
        elif key == 'motion_blur':
            transform_list.append(A.MotionBlur(p=augmentation_choices[key]['p'],blur_limit=augmentation_choices[key]['blur_limit']))
        elif key == 'median_blur':
            transform_list.append(A.MedianBlur(p=augmentation_choices[key]['p'],blur_limit=augmentation_choices[key]['blur_limit']))
        elif key == 'to_gray':
            transform_list.append(A.ToGray(p=augmentation_choices[key]['p']))
        elif key == 'random_brightness_contrast':
            transform_list.append(A.RandomBrightnessContrast(p=augmentation_choices[key]['p']))
        elif key == 'color_jitter':
            transform_list.append(A.ColorJitter(p=augmentation_choices[key]['p']))
        elif key == 'random_gamma':
            transform_list.append(A.RandomGamma(p=augmentation_choices[key]['p']))
        elif key == 'horizontal_flip':
            transform_list.append(A.HorizontalFlip(p=augmentation_choices[key]['p']))
        elif key == 'vertical_flip':
            transform_list.append(A.VerticalFlip(p=augmentation_choices[key]['p']))
        elif key == 'rotate':
            transform_list.append(A.Rotate(limit=augmentation_choices[key]['p']))
        elif key == 'shift_scale_rotate':
            transform_list.append(A.ShiftScaleRotate(shift_limit=augmentation_choices[key]['shift_limit'], 
                                                     scale_limit=augmentation_choices[key]['scale_limit'], 
                                                     rotate_limit=augmentation_choices[key]['rotate_limit']))
        elif key == 'Cutout':
            transform_list.append(A.ShiftScaleRotate(num_holes=augmentation_choices[key]['num_holes'], 
                                                     max_h_size=augmentation_choices[key]['max_h_size'], 
                                                     fill_value=augmentation_choices[key]['fill_value'],
                                                     p=augmentation_choices[key]['p']))    
        elif key == 'rotate':
            transform_list.append(A.ChannelShuffle(p=augmentation_choices[key]['p']))    
        transform_list.append(ToTensorV2(p=1.0))

        bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']}

        return A.Compose(transform_list, bbox_params=bbox_params)
def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def transform_mosaic(mosaic, boxes, img_size=640):
    """
    Resizes the `mosaic` image to `img_size` which is the desired image size
    for the neural network input. Also transforms the `boxes` according to the
    `img_size`.

    :param mosaic: The mosaic image, Numpy array.
    :param boxes: Boxes Numpy.
    :param img_resize: Desired resize.
    """
    aug = A.Compose(
        [A.Resize(img_size, img_size, always_apply=True, p=1.0)
    ])
    sample = aug(image=mosaic)
    resized_mosaic = sample['image']
    transformed_boxes = (np.array(boxes) / mosaic.shape[0]) * resized_mosaic.shape[1]
    for box in transformed_boxes:
        # Bind all boxes to correct values. This should work correctly most of
        # of the time. There will be edge cases thought where this code will
        # mess things up. The best thing is to prepare the dataset as well as 
        # as possible.
        if box[2] - box[0] <= 1.0:
            box[2] = box[2] + (1.0 - (box[2] - box[0]))
            if box[2] >= float(resized_mosaic.shape[1]):
                box[2] = float(resized_mosaic.shape[1])
        if box[3] - box[1] <= 1.0:
            box[3] = box[3] + (1.0 - (box[3] - box[1]))
            if box[3] >= float(resized_mosaic.shape[0]):
                box[3] = float(resized_mosaic.shape[0])
    return resized_mosaic, transformed_boxes

# Define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)
