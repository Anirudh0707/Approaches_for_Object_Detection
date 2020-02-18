import torch 
import torchvision
import numpy as np
# from PIL import Image
# import os
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
    centroid = np.array([0,0])
    
    # Showing boxes with score > threshold
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold and label == 37:  # Sports Ball label = 37
            # Given only one 1 ball per image
            x0 = np.clip(box[0], 0, width)
            x1 = np.clip(box[2], 0, width)
            y0 = np.clip(box[1], 0, height)
            y1 = np.clip(box[3], 0, height)
            output = cv2.rectangle(output, (x0, y0), (x1, y1), (0,0,255), 3)
            centroid = np.array([ (x0 + x1)//2 , (y0 + y1)//2 ])
            # Predictions come in descending score order. Due to the one frame, one ball assumption, we break the loop after one ball detection
            break
    return output, centroid

#################################################################################################################################
# Strings and Thresholds
#################################################################################################################################
videofile = input("Enter Filename With Extension :: ")
videofile_withoutextension = videofile.split(".")[0]
arraySaveFileName = "out_" + videofile_withoutextension
videoPathOut = "out_"+ videofile

# Since we're only looking at sports balls and that too only one sports ball perimeg(strongest)
# We keep a low threshold.  0.4 or 0.5 can be kept but in a few frames the ball can be missed
score_thresh = 0.2
#################################################################################################################################
# End of Stings and Thresholds
#################################################################################################################################
print("Video File :: ", videofile)
cap = cv2.VideoCapture(videofile)
assert cap.isOpened(), 'Cannot capture source'
centroid_list = []

# Values for saving frames and Normalizing Axes in the centroid plots
width = 1920; height = 1080
out = cv2.VideoWriter(videoPathOut,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920, 1080))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        predictions = predict(frame)
        output, centroid = getBoundingBoxAndCentroid(frame, predictions[0]['boxes'].cpu().detach().numpy(), predictions[0]['labels'].cpu().detach().numpy(), predictions[0]['scores'].cpu().detach().numpy(), score_thresh)
        centroid_list.append(centroid)
        
        cv2.imshow("frame", output)
        out.write(output)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break        
    else:
        break
print("Detectin Complete")
centroid_list = np.stack(centroid_list,0)
np.save(arraySaveFileName, centroid_list)
print("Saving Centroid List")
out.release()
cap.release()

array_to_find_nonzeros = np.mean(centroid_list, axis=1)
non_zero_index = np.nonzero(array_to_find_nonzeros)

plt.scatter(centroid_list[non_zero_index,0], centroid_list[non_zero_index,1])
# Due to the top left being origin (0,0) while plotting the origin will be at the bottom
# Alternatively we can perform y = Y_LIM - y ;   where Y_LIM will be number of rows. This will also essentially invert the axis
plt.xlim(0, width)
plt.ylim(0, height)
plt.title("Centroid Plot") 
plt.xlabel('X Position')
plt.ylabel('Y Position, Flipped')
plt.gca().invert_yaxis()
plt.show()