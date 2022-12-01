from functools import lru_cache

import torch
import cv2
import numpy as np
import os
import glob as glob
import random
from copy import copy

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from utils.transforms import (
    get_train_transform, get_valid_transform,
    get_train_aug
)



# the dataset class
class CustomDataset(Dataset):
    def __init__(
        self, images_path, labels_path, 
        width, height, classes, transforms=None, 
        use_train_aug=False,
        train=False, mosaic=False, cache_size=100
    ):
        self.transforms = transforms
        self.use_train_aug = use_train_aug
        self.images_path = images_path
        self.labels_path = labels_path
        self.height = height
        self.width = width
        self.classes = classes
        self.train = train
        self.mosaic = mosaic
        self.cache_size = cache_size
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        
        # get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        # Remove all annotations and images when no object is present.
        self.read_and_clean()

        # these cannot be pickled, thus I added set and get state methods to handle them
        self.load_image_and_labels = lru_cache(maxsize=self.cache_size)(self._load_image_and_labels)

    def __getstate__(self):
        result = copy(self.__dict__)
        result["load_image_and_labels"] = None
        return result

    def __setstate__(self, state):
        self.__dict__ = state
        self.load_image_and_labels = lru_cache(maxsize=self.cache_size)(self._load_image_and_labels)

    def read_and_clean(self):
        # Discard any images and labels when the XML 
        # file does not contain any object.
        for annot_path in self.all_annot_paths:
            tree = et.parse(annot_path)
            root = tree.getroot()
            object_present = False
            for member in root.findall('object'):
                if member.find('bndbox'):
                    object_present = True
            if object_present == False:
                image_name = annot_path.split(os.path.sep)[-1].split('.xml')[0]
                image_root = self.all_image_paths[0].split(os.path.sep)[:-1]
                # remove_image = f"{'/'.join(image_root)}/{image_name}.jpg"

                # TODO Is this code necessary?
                # for img_type in self.image_file_types:
                #     remove_image = os.path.join(os.sep.join(image_root), image_name+img_type.replace("*",""))
                #     if remove_image in self.all_image_paths:
                #         print(f"Removing {annot_path} and corresponding {remove_image}")
                #         self.all_annot_paths.remove(annot_path)
                #         self.all_image_paths.remove(remove_image)
                #         break

        # Discard any image file when no annotation file 
        # is not found for the image. 
        for image_name in self.all_images:
            possible_xml_name = os.path.join(self.labels_path, os.path.splitext(image_name)[0]+'.xml')
            if possible_xml_name not in self.all_annot_paths:
                print(f"{possible_xml_name} not found...")
                print(f"Removing {image_name} image")
                # items = [item for item in items if item != element]
                self.all_images = [image_instance for image_instance in self.all_images if image_instance != image_name]
                # self.all_images.remove(image_name)

        # for image_path in self.all_image_paths:
        #     image_name = image_path.split(os.path.sep)[-1].split('.jpg')[0]
        #     possible_xml_name = f"{self.labels_path}/{image_name.split('.jpg')[0]}.xml"
        #     if possible_xml_name not in self.all_annot_paths:
        #         print(f"{possible_xml_name} not found...")
        #         print(f"Removing {image_name} image")
        #         self.all_image_paths.remove(image_path)

    def _load_image_and_labels(self, index):
        image_name = self.all_images[index]
        image_path = os.path.join(self.images_path, image_name)

        # Read the image.
        image = cv2.imread(image_path)
        # Convert BGR to RGB color format.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized0 = cv2.resize(image, (self.width, self.height))
        image_resized = copy(image_resized0)/255.0
        
        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.labels_path, annot_filename)

        boxes = []
        orig_boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # Get the height and width of the image.
        image_width = image.shape[1]
        image_height = image.shape[0]
                
        # Box coordinates for xml files are extracted and corrected for image size given.
        for member in root.findall('object'):
            # Map the current object name to `classes` list to get
            # the label index and append to `labels` list.
            labels.append(self.classes.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            ymax, xmax = self.check_image_and_annotation(
                xmax, ymax, image_width, image_height
            )

            orig_boxes.append([xmin, ymin, xmax, ymax])
            
            # Resize the bounding boxes according to the
            # desired `width`, `height`.
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Bounding box to tensor.
        boxes_length = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Area of the bounding boxes.

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes_length > 0 else torch.as_tensor(boxes, dtype=torch.float32)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)
        return image_resized0, image_resized, orig_boxes, \
            boxes, labels, area, iscrowd, (image_width, image_height)

    def check_image_and_annotation(self, xmax, ymax, width, height):
        """
        Check that all x_max and y_max are not more than the image
        width or height.
        """
        if ymax > height:
            ymax = height
        if xmax > width:
            xmax = width
        return ymax, xmax


    def load_cutmix_image_and_boxes(self, index, resize_factor=512):
        """ 
        Adapted from: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet
        """
        image, _, _, _, _, _, _, _ = self.load_image_and_labels(index=index)
        #orig_image = image.copy()
        # Resize the image according to the `confg.py` resize.
        image = cv2.resize(image, resize_factor)
        h, w, c = image.shape
        s = h // 2

        xc, yc = [int(random.uniform(h * 0.25, w * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]

        # Create empty image with the above resized image.
        result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_classes = []

        for i, index in enumerate(indexes):
            image, image_resized, orig_boxes, boxes, \
            labels, area, iscrowd, dims = self.load_image_and_labels(
                index=index
            )

            # Resize the current image according to the above resize,
            # else `result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]`
            # will give error when image sizes are different.

            image = cv2.resize(image, resize_factor)

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(orig_boxes) > 0:
                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh

                result_boxes.append(boxes)
                result_classes += labels

        final_classes = []
        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
            result_boxes = result_boxes.astype(np.int32)
            for idx in range(len(result_boxes)):
                if ((result_boxes[idx, 2] - result_boxes[idx, 0]) * (result_boxes[idx, 3] - result_boxes[idx, 1])) > 0:
                    final_classes.append(result_classes[idx])
            result_boxes = result_boxes[
                np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)
            ]
        return result_image/255., torch.tensor(result_boxes), \
            torch.tensor(np.array(final_classes)), area, iscrowd, dims

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        if not self.mosaic:
            image, image_resized, orig_boxes, boxes, \
                labels, area, iscrowd, dims = self.load_image_and_labels(
                index=idx
            )

        if self.train and self.mosaic:
            #while True:
            image_resized, boxes, labels, \
                area, iscrowd, dims = self.load_cutmix_image_and_boxes(
                idx, resize_factor=(self.height, self.width)
            )
                # Only needed if we don't allow training without target bounding boxes
               # if len(boxes) > 0:
               #     break
        
        # visualize_mosaic_images(boxes, labels, image_resized, self.classes)

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.use_train_aug: # Use train augmentation if argument is passed.
            train_aug = get_train_aug()
            sample = train_aug(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)
        else:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes']).to(torch.int64)

        # Fix to enable training without target bounding boxes,
        # see https://discuss.pytorch.org/t/fasterrcnn-images-with-no-objects-present-cause-an-error/117974/4
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
            
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# Prepare the final datasets and data loaders.
def create_train_dataset(
    train_dir_images, train_dir_labels, 
    resize_width, resize_height, classes,
    use_train_aug=False,
    mosaic=True,
    cache_size=0
):
    train_dataset = CustomDataset(
        train_dir_images, train_dir_labels,
        resize_width, resize_height, classes, 
        get_train_transform(),
        use_train_aug=use_train_aug,
        train=True, mosaic=mosaic,
        cache_size=cache_size
    )
    return train_dataset
def create_valid_dataset(
    valid_dir_images, valid_dir_labels, 
    resize_width, resize_height, classes, cache_size=0
):
    valid_dataset = CustomDataset(
        valid_dir_images, valid_dir_labels, 
        resize_width, resize_height, classes, 
        get_valid_transform(),
        train=False,
        cache_size=cache_size
    )
    return valid_dataset

def create_train_loader(
    train_dataset, batch_size, num_workers=0, batch_sampler=None, prefetch_factor=2
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        #pin_memory=True
    )
    return train_loader
def create_valid_loader(
    valid_dataset, batch_size, num_workers=0, batch_sampler=None, prefetch_factor=2
):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=batch_sampler,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        #pin_memory=True,
    )
    return valid_loader