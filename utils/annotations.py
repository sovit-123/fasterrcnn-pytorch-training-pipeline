import numpy as np
import cv2

def inference_annotations(
    outputs, detection_threshold, classes,
    colors, orig_image
):
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()
    # Filter out boxes according to `detection_threshold`.
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    draw_boxes = boxes.copy()
    # Get all the predicited class names.
    pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]

    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    
    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(draw_boxes):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        class_name = pred_classes[j]
        color = colors[classes.index(class_name)]
        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color, 
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        # For filled rectangle.
        w, h = cv2.getTextSize(
            class_name, 
            0, 
            fontScale=lw / 3, 
            thickness=tf
        )[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(orig_image, p1, p2, color, -1, cv2.LINE_AA)  
        cv2.putText(
            orig_image, 
            class_name, 
            (p1[0], p1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw/3, 
            color=(255, 255, 255), 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    return orig_image