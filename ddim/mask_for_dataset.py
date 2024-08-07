#巨慢，建议睡一觉....5s一张，共7k张，大概10h
import numpy as np
import cv2
import os

def extract_and_save_masks(dataset_path, output_path, mask_rcnn_path, confidence_threshold=0.7, mask_threshold=0.3, use_gpu=True, grabcut_iter=10):
    # Load the COCO class labels
    labelsPath = os.path.sep.join([mask_rcnn_path, "object_detection_classes_coco.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # Initialize colors for each class
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Load Mask R-CNN weights and model configuration
    weightsPath = os.path.sep.join([mask_rcnn_path, "frozen_inference_graph.pb"])
    configPath = os.path.sep.join([mask_rcnn_path, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
    print(os.listdir(dataset_path))
    # Check for GPU usage
    if use_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Process each image in the dataset
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_mask_path = os.path.join(output_path, "mask_" + image_name)
            process_and_save_mask(image_path, output_mask_path, net, LABELS, confidence_threshold, mask_threshold,
                                  grabcut_iter)

def process_and_save_mask(image_path, output_mask_path, net, LABELS, confidence_threshold, mask_threshold, grabcut_iter):
    # Load and resize the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to load image at {image_path}. Skipping.")
        return
    image = cv2.resize(image, (600, int(image.shape[0] * 600 / image.shape[1])))

    # Prepare the blob and perform a forward pass
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    final_mask = np.zeros(image.shape[:2], dtype="uint8")

    # Process each detected object
    for i in range(0, boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > confidence_threshold:
            (H, W) = image.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # Extract and resize the mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            mask = (mask > mask_threshold).astype("uint8") * 255

            # Combine mask of each object
            final_mask[startY:endY, startX:endX] = cv2.bitwise_or(final_mask[startY:endY, startX:endX], mask)

    # Apply GrabCut algorithm
    if np.any(final_mask > 0) and np.any(final_mask == 0):
        gcMask = final_mask.copy()
        gcMask[gcMask > 0] = cv2.GC_PR_FGD
        gcMask[gcMask == 0] = cv2.GC_BGD
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        cv2.grabCut(image, gcMask, None, bgModel, fgModel, iterCount=grabcut_iter, mode=cv2.GC_INIT_WITH_MASK)
        outputMask = np.where((gcMask == cv2.GC_BGD) | (gcMask == cv2.GC_PR_BGD), 0, 1)
        outputMask = (outputMask * 255).astype("uint8")
        
        # Save the final mask
        cv2.imwrite(output_mask_path, outputMask)

# Example usage
dataset_path = 'images'
output_path = 'images_mask'
mask_rcnn_path = 'mask-rcnn-coco'
extract_and_save_masks(dataset_path, output_path, mask_rcnn_path)