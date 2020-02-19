# Yolo V3
The darknet based faster rcnn from pytorch is used. The model is pretrained. The weights can be downloaded from [here](https://pjreddie.com/media/files/yolov3.weights) 

This implmentation comprises of 2 files:
1) FasterRCNN_video.py
2) FasterRCNN_singleframe.py

### yolo_v3_video.py
**Inputs**    : Video File name (video.avi)<br/>
**Outputs**   : Output Video (out.avi); Numpy array of the centroids; matplotlib plot(GUI) of the centroids<br/>
**Algorithm** : The frames of the video are preprocessed. A pre-trained Yolo V3 is used to predict the bounding box for the ball only. The centroid are also calculated from the network outwork 

### yolo_v3_singleframe.py
**Inputs**    : Image File name (image.jpg)<br/>
**Outputs**   : Output image (out.jpg)<br/>
**Algorithm** : The image is preprocessed. A Yolo V3 is used to predict the bounding box for the ball only. The centroid are also calculated from the network outwork 

## Credit
The implmentation and further examples of the darknet based Yolo V3 net is given in this [repository](https://github.com/ayooshkathuria/pytorch-yolo-v3)
