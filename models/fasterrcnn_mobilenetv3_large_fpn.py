import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        pretrained=pretrained
    )
    if coco_model:
        return model
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model


# import torchvision

# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# def create_model(num_classes, min_size=None, max_size=None):
    
#     # load Faster RCNN pre-trained model
#     if min_size is None and max_size is None:
#         model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
#             pretrained=True
#         )
#     else:
#         assert min_size is not None and max_size is not None, \
#             'Please provide both min and max size'
#         model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
#             pretrained=True, max_size=max_size
#         )
    
#     # get the number of input features 
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # define a new head for the detector with required number of classes
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

#     return model