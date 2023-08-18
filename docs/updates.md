# Updates

## 2023-8-18

* Filter classes to visualize during inference using the `--classes` command line argument with space separated class indices from the dataset YAML file. 

  For example, to visualize only persons in COCO dataset, use,  `python inference.py --classes 1 <rest of the command>`

  To visualize person and car, use, `python inference.py --classes 1 3 <rest of the command>`

* Added Deep SORT Real-Time tracking to `inference_video.py` and `onnx_video_inference.py`. Using `--track` command with the usual inference command. Support for **MobileNet** Re-ID for now.

## 2023-02-02

* New DenseNet backbones.
* Mosaic augmentation updated to be Ultralytics/YOLOv5/YOLOv8 like.
* Updated augmentations regime for better training results

## 2022-10-02

* Released a Mini Darknet Nano Head model pretrained on the Pascal VOC model for 600 epochs. [Find the release details here](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline/releases/tag/Latest).

## 2022-09-09

* Can load COCO/Pascal VOC pretrained weights for transfer learning/fine tuning using the `--weights` flag and providing the path to the weights file.
* Resume training by providing the path to the previous trained weights using the `--weights` flag and `--resume-training` flag to continue from previous plots and load the optimizer state dictionary as well. Here, `--weights` will take the path to the `last_model.pth` and not `best_model.pth` or `last_model_state.pth`. The latter two models store the model weights dictionary only.
* Weights and Biases logging possible now.
* Default training resolution now 640x640. Provides slightly better results.