import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def coco_to_voc(coco_json_path, output_annotations_dir, output_image_dir, original_image_dir, categories):
    """
    Convert COCO format JSON file to Pascal VOC format XML files and copy image files.

    Args:
        coco_json_path (str): Path to COCO JSON file (train.json or val.json).
        output_annotations_dir (str): Output directory for Pascal VOC XML files (Annotations/train or Annotations/val).
        output_image_dir (str): Output directory for Pascal VOC image files (images/train or images/val).
        original_image_dir (str): Original COCO image file directory (images/train or images/val).
        categories (list): List of categories, read from JSON file.
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    image_id_to_filename = {image['id']: image['file_name'] for image in coco_data['images']}
    image_id_to_size = {image['id']: (image['width'], image['height']) for image in coco_data['images']}
    category_id_to_name = {category['id']: category['name'] for category in categories}

    os.makedirs(output_annotations_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True) # Create image output directory

    image_filenames = []

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        filename = image_id_to_filename[image_id]
        width, height = image_id_to_size[image_id]
        category_id = annotation['category_id']
        category_name = category_id_to_name[category_id]
        bbox = annotation['bbox']  # [x_min, y_min, width, height]

        xml_filename = filename.replace(os.path.splitext(filename)[1], '.xml')
        xml_filepath = os.path.join(output_annotations_dir, xml_filename)
        output_image_filepath = os.path.join(output_image_dir, filename)
        original_image_filepath = os.path.join(original_image_dir, filename)

        if xml_filename not in image_filenames:
            image_filenames.append(xml_filename.replace('.xml', ''))
            root = ET.Element('annotation')

            ET.SubElement(root, 'folder').text = 'VOC' # You can customize the folder name
            ET.SubElement(root, 'filename').text = filename
            ET.SubElement(root, 'path').text = output_image_filepath # Modify path to the new image path

            source = ET.SubElement(root, 'source')
            ET.SubElement(source, 'database').text = 'Unknown' # You can customize the database name

            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(width)
            ET.SubElement(size, 'height').text = str(height)
            ET.SubElement(size, 'depth').text = '3' # Assume color images

            ET.SubElement(root, 'segmented').text = '0' # Typically set to 0 for object detection

            # Copy image file to the new directory
            shutil.copy2(original_image_filepath, output_image_filepath) # Use copy2 to preserve metadata

        object_elem = ET.SubElement(root, 'object')
        ET.SubElement(object_elem, 'name').text = category_name
        ET.SubElement(object_elem, 'pose').text = 'Unspecified'
        ET.SubElement(object_elem, 'truncated').text = '0'
        ET.SubElement(object_elem, 'difficult').text = '0' # You can set the difficult flag as needed
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0])) # x_min
        ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1])) # y_min
        ET.SubElement(bndbox, 'xmax').text = str(int(bbox[0] + bbox[2])) # x_max
        ET.SubElement(bndbox, 'ymax').text = str(int(bbox[1] + bbox[3])) # y_max

        # Write XML to file, write XML header only once per image
        if not os.path.exists(xml_filepath): # Avoid writing header repeatedly
            xml_string = prettify_xml(root)
            with open(xml_filepath, 'w', encoding='utf-8') as xml_file:
                xml_file.write(xml_string)

    return image_filenames


if __name__ == '__main__':
    original_dataset_root = 'First_Dataset_COCO' # Original COCO dataset root directory
    dataset_root = 'First_Dataset_PascalVOC' # New Pascal VOC dataset root directory

    categories_json_path = os.path.join(original_dataset_root, 'train.json') # Category information is read from the original COCO dataset
    train_json_path = os.path.join(original_dataset_root, 'train.json')
    val_json_path = os.path.join(original_dataset_root, 'val.json')
    original_train_image_dir = os.path.join(original_dataset_root, 'images', 'train') # Original training image directory
    original_val_image_dir = os.path.join(original_dataset_root, 'images', 'val')   # Original validation image directory
    output_annotations_dir = os.path.join(dataset_root, 'Annotations') # XML files output to Annotations directory under PascalVOC
    output_imagesets_dir = os.path.join(dataset_root, 'ImageSets', 'Main') # train.txt, val.txt output to ImageSets directory under PascalVOC
    output_image_dir = os.path.join(dataset_root, 'images') # New image root directory

    os.makedirs(dataset_root, exist_ok=True) # Create PascalVOC root directory
    os.makedirs(output_image_dir, exist_ok=True) # Create new image root directory
    os.makedirs(os.path.join(output_image_dir, 'train'), exist_ok=True) # Create train image directory
    os.makedirs(os.path.join(output_image_dir, 'val'), exist_ok=True)   # Create val image directory


    # Read category information (read from train.json, assuming category information is consistent in train.json and val.json)
    with open(categories_json_path, 'r') as f:
        categories_data = json.load(f)
        categories = categories_data['categories']

    # Convert train set
    print("Converting train set...")
    train_image_filenames = coco_to_voc(train_json_path, os.path.join(output_annotations_dir, 'train'), os.path.join(output_image_dir, 'train'), original_train_image_dir, categories)
    # Convert val set
    print("Converting val set...")
    val_image_filenames = coco_to_voc(val_json_path, os.path.join(output_annotations_dir, 'val'), os.path.join(output_image_dir, 'val'), original_val_image_dir, categories)

    os.makedirs(output_imagesets_dir, exist_ok=True)

    # Generate train.txt
    with open(os.path.join(output_imagesets_dir, 'train.txt'), 'w') as f:
        for filename in train_image_filenames:
            f.write(filename + '\n')
    print(f"train.txt file generated, containing {len(train_image_filenames)} image filenames.")

    # Generate val.txt
    with open(os.path.join(output_imagesets_dir, 'val.txt'), 'w') as f:
        for filename in val_image_filenames:
            f.write(filename + '\n')
    print(f"val.txt file generated, containing {len(val_image_filenames)} image filenames.")

    print("COCO to Pascal VOC conversion completed!")
    print(f"Pascal VOC format dataset saved in: {dataset_root}")
    print(f"XML files saved in: {output_annotations_dir}")
    print(f"ImageSets files saved in: {output_imagesets_dir}")
    print(f"Image files saved in: {output_image_dir}")