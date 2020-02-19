# Faster RCNN
The resnet based faster rcnn from pytorch is used. The model is pretrained.

This implmentation comprises of 2 files:
1) FasterRCNN_video.py
2) FasterRCNN_singleframe.py

### FasterRCNN_Video.py
Inputs    : Video File name (video.avi)<br/>
Outputs   : Output Video (out.avi); Numpy array of the centroids; matplotlib plot(GUI) of the centroids<br/>
Algorithm : The frames of the video are preprocessed. A pre-trained Faster RCNN is used to predict the bounding box for the ball only. The centroid are also calculated from the network outwork 

### FasterRCNN_Single_Image.py
Inputs    : Image File name (image.jpg)<br/>
Outputs   : Output image (out.jpg)
Algorithm : The image is preprocessed. A pre-trained Faster RCNN is used to predict the bounding box for the ball only. The centroid are also calculated from the network outwork 
