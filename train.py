import os
import torch
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file


# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
output_dir = '/media/longc/Data/models/faster_rcnn_pytorch'

max_iters = 70000
lr = cfg.TRAIN.LEARNING_RATE
# ------------

cfg_from_file(cfg_file)

imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb

data_layer = RoIDataLayer(roidb, imdb.num_classes)

net = FasterRCNN(classes=imdb.classes)
net.rpn.features.load_from_npy_file(pretrained_model)

net.cuda()
net.train()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_loss = 0
log_interval = 50
step_cnt = 0
t = Timer()
t.tic()
for step in range(max_iters):
    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']

    # forward
    net(im_data, im_info, gt_boxes, dontcare_areas)
    loss = net.loss
    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % log_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        print('step %d, loss: %.4f, fps: %.2f' % (step, train_loss / step_cnt, fps))
        train_loss = 0
        step_cnt = 0
        t.tic()

    if step % 10000 == 0 and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))