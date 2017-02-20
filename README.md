# Faster RCNN with PyTorch
This is a [PyTorch](https://github.com/pytorch/pytorch)
implementation of Faster RCNN. 
This project is mainly based on [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
and [TFFRCNN](https://github.com/CharlesShang/TFFRCNN).

For details about R-CNN please refer to the [paper](https://arxiv.org/abs/1506.01497) 
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks 
by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

### Progress

- [x] Forward for detecting
- [x] RoI Pooling layer with C extensions on CPU (only forward)
- [x] RoI Pooling layer on GPU (forward and backward)
- [x] Training on VOC2007
- [x] TensroBoard support
- [x] Evaluation

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

### Training on Pascal VOC 2007

Follow [this project (TFFRCNN)](https://github.com/CharlesShang/TFFRCNN)
to download and prepare the training, validation, test data 
and the VGG16 model pre-trained on ImageNet. 

Since the program loading the data in `faster_rcnn_pytorch/data` by default,
you can set the data path as following.
```bash
cd faster_rcnn_pytorch
mkdir data
cd data
ln -s $VOCdevkit VOCdevkit2007
```

Then you can set some hyper-parameters in `train.py` and training parameters in the `.yml` file.

Now I got a 0.661 mAP on VOC07 while the origin paper got a 0.699 mAP.
You may need to tune the loss function defined in `faster_rcnn/faster_rcnn.py` by yourself.

### Training with TensorBoard
With the aid of [Crayon](https://github.com/torrvision/crayon),
we can access the visualisation power of TensorBoard for any 
deep learning framework.

To use the TensorBoard, install Crayon (https://github.com/torrvision/crayon)
and set `use_tensorboard = True` in `faster_rcnn/train.py`.

### Evaluation
Set the path of the trained model in `test.py`.
```bash
cd faster_rcnn_pytorch
mkdir output
python test.py
```