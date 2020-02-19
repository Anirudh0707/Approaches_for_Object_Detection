# Approaches_for_Object_Detection
Two sepereate modules have been implmented for the task of object detection. 
<br/>Namely:<br/>
1) Faster RCNN
2) Yolo V3

## Requirements
pytorch (torch and torchvision)<br/>
numpy<br/>
cv2<br/>
matplotlib<br/>
pickle<br/>
random<br/>
os<br/>
time<br/>

## Faster RCNN
There implmentation comprises of 2 files:
1) FasterRCNN_video.py
2) FasterRCNN_singleframe.py

### FasterRCNN_video.py
Inputs    : Video File name (video.avi)<br/>
Outputs   : Output Video (out.avi); Numpy array of the centroids; matplotlib plot(GUI) of the centroids<br/>
Algorithm : The frames of the video are preprocessed. A pre-trained Faster RCNN is used to predict the bounding box for the ball only. The centroid are also calculated from the network outwork 

### FasterRCNN_singleframe.py


## Yolo V3
