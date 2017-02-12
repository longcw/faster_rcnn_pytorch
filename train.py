import os
import torch
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
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

# ------------

cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE

imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb

data_layer = RoIDataLayer(roidb, imdb.num_classes)

net = FasterRCNN(classes=imdb.classes)
# network.weights_normal_init(net, dev=0.01)
# net.rpn.features.load_from_npy_file(pretrained_model)
model_file = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
# model_file = '/media/longc/Data/models/faster_rcnn_pytorch/faster_rcnn_10000.h5'
network.load_net(model_file, net)
network.weights_normal_init([net.bbox_fc, net.score_fc, net.fc6, net.fc7], dev=0.01)

# net = net.rpn

net.cuda()
net.train()

params = list(net.parameters())
for p in params:
    print p.size()
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.SGD(params[-8:], lr=lr, momentum=0.9, weight_decay=0.0005)
train_all = True
# target_net = net.rpn
target_net = net
network.set_trainable(net.rpn, False)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
log_interval = 50
step_cnt = 0
t = Timer()
t.tic()
for step in range(0, max_iters+1):
    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']

    # forward
    # cls_prob, bbox_pred, rois = net(im_data, im_info, gt_boxes, dontcare_areas)

    target_net(im_data, im_info, gt_boxes, dontcare_areas)
    loss = target_net.loss

    tp += float(target_net.tp)
    tf += float(target_net.tf)
    fg += target_net.fg_cnt
    bg += target_net.bg_cnt
    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(target_net, 10.)
    optimizer.step()

    if step % log_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration
        print('step %d, loss: %.4f, fps: %.2f, tp: %.2f, tf: %.2f, fg/bg=(%d/%d)'
              % (step, train_loss / step_cnt, fps, tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt)),
        if train_all:
            print net.rpn.cross_entropy.data.cpu().numpy(), net.rpn.loss_box.data.cpu().numpy(), \
                net.cross_entropy.data.cpu().numpy(), net.loss_box.data.cpu().numpy()
        else:
            print net.rpn.cross_entropy.data.cpu().numpy(), net.rpn.loss_box.data.cpu().numpy()

        train_loss = 0
        step_cnt = 0
        tp, tf, fg, bg = 0., 0., 0, 0
        t.tic()

    if step % 10000 == 0 and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))
        # lr /= 10
        # optimizer = torch.optim.SGD(params[-8:], lr=lr, momentum=0.9, weight_decay=0.0005)
        # if step >= 20000:
        #     lr /= 3
        #     optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=0.9, weight_decay=0.0005)
        #     # optimizer = torch.optim.Adam(params[8:], lr=lr)
        #     train_all = True
        #     target_net = net
