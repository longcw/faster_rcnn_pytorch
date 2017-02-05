# Faster RCNN with PyTorch
This is a [PyTorch](https://github.com/pytorch/pytorch)
implementation of Faster RCNN. 
This project is mainly based on [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
and [TFFRCNN](https://github.com/CharlesShang/TFFRCNN).

For details about R-CNN please refer to the [paper](https://arxiv.org/abs/1506.01497) 
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks 
by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

### Progress

- [x] forward pass for detecting
- [x] using models trained by Tensorflow
- [x] roi pooling layer implemented by python and pytorch
- [x] roi pooling layer with C extensions on CPU (only forward)
- [x] roi pooling layer on GPU (forward and backward)
- [ ] backward pass for training

### Installation and demo
1. Clone the Faster R-CNN repository
    ```bash
    git clone git@github.com:longcw/faster_rcnn_pytorch.git
    ```

2. Build the Cython modules for nms and the roi_pooling layer
    ```bash
    cd faster_rcnn_pytorch/faster_rcnn
    ./make.sh
    ```
3. Download the trained model [VGGnet_fast_rcnn_iter_70000.h5](https://drive.google.com/open?id=0B4pXCfnYmG1WOXdpYVFybWxiZFE) 
and set the model path in `demo.py`
3. Run demo `python demo.py`

