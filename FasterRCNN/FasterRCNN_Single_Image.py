import torch 
import torchvision
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2

CUDA = torch.cuda.is_available()
print("CUDA :: ",CUDA)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
if CUDA:
    model.cuda()

classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]  # There are 91 classes(including N/A) for the torch library's FasterCNN. This list can be found online

def preprocess(image):
    # Resize; Although the FasterRCNN is size invariant, it can work better with resizing(Depending on the data).
    # In our case we don;t need to resize
    # The boudning boxes are later brought back to the original image dimensions
    # ratio = 800.0 / min(image.shape[0], image.shape[1])
    # image = cv2.resize(image, (int(ratio * image.shape[0]), int(ratio * image.shape[1])), interpolation = cv2.)

    # Convert to RGB from BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1]).astype('float32')

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

def predict(img):
    img_data = preprocess(img)
    im = torch.as_tensor(img_data, dtype=torch.float32)/255
    if CUDA:    
        im = im.cuda()

    # start_time = time.time()
    predictions = model([im])
    # end_time = time.time()
    # print("Time Taken for a Frame: ", end_time - start_time)
    return predictions

def getBoundingBoxAndCentroid(image, boxes, labels, scores, score_threshold=0.5):
    # Resize boxes
    # ratio = 800.0 / min(image.shape[0], image.shape[1])
    # boxes /= ratio
    output = image.copy()
    
    # Showing boxes with score > threshold
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:# and label == 37:  # Sports Ball label = 37
            # Given only one 1 ball per image
            x0 = np.clip(box[0], 0, width)
            x1 = np.clip(box[2], 0, width)
            y0 = np.clip(box[1], 0, height)
            y1 = np.clip(box[3], 0, height)
            output = cv2.rectangle(output, (x0, y0), (x1, y1), (0,0,255), 1)
            # Predictions come in descending score order. Due to the one frame, one ball assumption, we break the loop after one ball detection
            # break
    return output

#################################################################################################################################
# Strings and Thresholds
#################################################################################################################################
imageFile = input("Enter Filename With Extension (Eg : demo.jpg) :: ")
imageFile_withoutextension = imageFile.split(".")[0]
arraySaveFileName = "out_" + imageFile_withoutextension
outImage = "out_"+ imageFile

# Class Confidence Threshold
# In this Image demo we detect everything, so we keep a igher thresh
score_thresh = 0.5
#################################################################################################################################
# End of Stings and Thresholds
#################################################################################################################################
print("Image File :: ", imageFile)
frame = cv2.imread(imageFile, 1) # Open Color
assert os.path.isfile(imageFile), 'Image does not exist'
height, width, _ = frame.shape

predictions = predict(frame)
output = getBoundingBoxAndCentroid(frame, predictions[0]['boxes'].cpu().detach().numpy(), predictions[0]['labels'].cpu().detach().numpy(), predictions[0]['scores'].cpu().detach().numpy(), score_thresh)

cv2.imshow("frame", output)
cv2.imwrite("output.jpg",output)
key = cv2.waitKey(5000)

print("Detection Complete")