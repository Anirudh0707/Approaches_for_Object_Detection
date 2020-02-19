from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
import random 
import os
import pickle as pkl

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
        x = output[i]
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = (0,0,255) # Red
        img = cv2.rectangle(img, c1, c2,color, 1)
    return img


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
    imageFile = input("Enter Filename With Extension (Eg : demo.jpg) :: ")
    imageFile_withoutextension = imageFile.split(".")[0]
    arraySaveFileName = "out_" + imageFile_withoutextension
    ImagePathOut = "out_"+ imageFile

    confidence = 0.5
    nms_thesh = 0.4
    start = 0
    num_classes = 80    
    bbox_attrs = 5 + num_classes
##################################################################################################################
    # Load the Model
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
    
    # Open the Image
    assert os.path.isfile(imageFile), 'Image does not exist'
    frame = cv2.imread(imageFile)

    # preprocess the image
    img, orig_im, dim = prep_image(frame, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1,2)                        
    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()
    
    # Predic the bboxes
    start = time.time()
    with torch.no_grad():   
        output = model(Variable(img), CUDA)
    # print("Time Taken for a frame", time.time() - start)

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
    
    # Draw bboxes
    out_im = write(output, orig_im)
    cv2.imwrite("output.jpg",out_im)
    cv2.imshow("frame", out_im)
    key = cv2.waitKey(5000)
    print("Detection Complete")