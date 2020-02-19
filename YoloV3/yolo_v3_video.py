from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
import os
from util import *
from darknet import Darknet
import random 

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(output, img):
    """
    The current implmentation only involves a Single Class per box. With a few modifications, the network can be trained on mulitlabel classification for each bbox
    Takes each of the BBox predictions and draws a rectangle around the detected objects
    """
    for i in range(len(output)):
        x = output[i].astype("int32")
        c1 = tuple(x[1:3])
        c2 = tuple(x[3:5])
        cls = x[-1]
        label = "{0}".format(classes[cls])
        color = (0,0,255) # Red
        centroid = np.array([0,0])
        if cls == 32: ## Sports Ball Label = 32
            img = cv2.rectangle(img, c1, c2,color, 3)
            centroid = np.array([ (c1[0] + c2[0])//2 , (c1[1] + c2[1])//2 ])
            # Since we assume only one ball per frame we take the strongest sports ball predictioon and draw the box. This helps eliminate other slight mismatches in case any
            break
    return img, centroid


if __name__ == '__main__':
    classes = [
        'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 
        'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
##################################################################################################################
## Srings and Tunable Variables + Thresh
##################################################################################################################
    videofile = input("Enter Filename With Extension (Eg : video.avi) (Recommended file types : avi) :: ")
    videofile_withoutextension = videofile.split(".")[0]
    arraySaveFileName = "out_" + videofile_withoutextension
    videoPathOut = "out_"+ videofile

    confidence = 0.5
    nms_thesh = 0.4
    start = 0
    num_classes = 80    
    bbox_attrs = 5 + num_classes
##################################################################################################################
    # Load the model from cofig and the weights file
    CUDA = torch.cuda.is_available()
    print("CUDA :: ", CUDA)
    print("Loading network.....")
    assert os.path.isfile("yolov3.cfg"), 'Config File does not exist'
    assert os.path.isfile("yolov3.weights"), 'Weights File don\'t does not exist. Please check the download, Link :: https://pjreddie.com/media/files/yolov3.weights'
    model = Darknet("yolov3.cfg")
    model.load_weights("yolov3.weights")
    if CUDA:
        model.cuda()
    model.eval()
    print("Network successfully loaded")

    model.net_info["height"] = 416
    inp_dim = 416
    
    # Initialize the Video reader and writer
    cap = cv2.VideoCapture(videofile)
    assert os.path.isfile(videofile), 'Video does not exist'
    assert cap.isOpened(), 'Cannot capture source'
    
    width = 1920; height = 1080
    out = cv2.VideoWriter(videoPathOut,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920, 1080))
    centroid_list = []
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Prepare the image/frame
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            # Predict the output
            with torch.no_grad():   
                output = model(Variable(img), CUDA)
            
            # Non Maximal Suppression and confidence thresholding
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            # Draw the bounding boxes and obtain the centroid
            out_im, centroid = write(output.cpu().detach().numpy(), orig_im)
            centroid_list.append(centroid)
            
            cv2.imshow("frame", orig_im)
            out.write(orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break            
        else:
            break
print("Time Taken", time.time() - start)
centroid_list = np.stack(centroid_list,0)
np.save(arraySaveFileName, centroid_list)
print("Saving Centroid List")
out.release()
cap.release()

frame_count = len(centroid_list)
array_to_find_nonzeros = np.mean(np.abs(centroid_list), axis=1)
temp = array_to_find_nonzeros[array_to_find_nonzeros==0]
non_zero_index = np.nonzero(array_to_find_nonzeros)
print("Frames :: ", frame_count)
print("Frame which had a sports ball detected :: ", frame_count - len(temp))

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

    

