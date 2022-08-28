import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        pretrained=pretrained
    )
    if coco_model: # Return the COCO pretrained model for COCO classes.
        return model
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

if __name__ == '__main__':
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")