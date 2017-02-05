import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

import network
from network import Conv2d, FC
# from roi_pool import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
from vgg16 import VGG16


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class RPN(nn.Module):
    _feat_stride = [16, ]
    anchor_scales = [8, 16, 32]

    def __init__(self):
        super(RPN, self).__init__()

        self.features = VGG16(bn=False)
        self.conv1 = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv = Conv2d(512, len(self.anchor_scales) * 3 * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * 3 * 4, 1, relu=False, same_padding=False)

    def forward(self, image):
        features, im_info = self.features(image)

        rpn_conv1 = self.conv1(features)
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.anchor_scales)*3*2)

        rpn_bbox_pred = self.bbox_conv(rpn_conv1)
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                                   'TEST', self._feat_stride, self.anchor_scales)

        return features, rois, im_info

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        self.features.load_from_npz(params)

        pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv': 'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(3, 2, 0, 1)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales):
        # if isinstance(rpn_cls_prob_reshape, tuple):
        #     rpn_cls_prob_reshape = rpn_cls_prob_reshape[0]
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales)
        x = torch.from_numpy(x)
        x = Variable(x).cuda()
        return x.view(-1, 5)


class FasterRCNN(nn.Module):
    n_classes = 21
    classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])

    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.rpn = RPN()
        self.roi_pool = RoIPool(7, 7, 1.0/16)
        self.fc6 = FC(512 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096)
        self.score_fc = FC(4096, self.n_classes, relu=False)
        self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)

    def forward(self, image):
        features, rois, im_info = self.rpn(image)
        t = Timer()
        t.tic()
        pooled_features = self.roi_pool(features, rois)
        roi_pooling_t = t.toc()
        print('roi pooling spend: {}s'.format(roi_pooling_t))

        print torch.max(pooled_features)
        print torch.min(pooled_features)
        print pooled_features.size()
        print torch.max(pooled_features, 1)[0][0]

        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.fc6(x)
        x = self.fc7(x)

        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)

        bbox_pred = self.bbox_fc(x)

        return cls_prob, bbox_pred, rois, im_info

    def load_from_npz(self, params):
        self.rpn.load_from_npz(params)

        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)

    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()
        drop = np.zeros_like(inds, dtype=np.int)
        drop[inds <= 0] = 1
        drop[scores < min_score] = 1
        keep = drop == 0
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep.reshape(-1)
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

        return pred_boxes, scores, self.classes[inds]

    def detect(self, image, thr=0.3):
        cls_prob, bbox_pred, rois, im_info = self.__call__(image)
        pred_boxes, scores, classes = \
            self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
        return pred_boxes, scores, classes

if __name__ == '__main__':
    def main():
        import os
        im_file = 'demo/004545.jpg'
        image = cv2.imread(im_file)

        detector = FasterRCNN()
        network.load_net('/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
        detector.cuda()
        print('load model successfully!')

        # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
        # print('save model succ')


        t = Timer()
        t.tic()
        dets, scores, classes = detector.detect(image, 0.3)
        runtime = t.toc()
        print('total spend: {}s'.format(runtime))

        im2show = np.copy(image)
        for i, det in enumerate(dets):
            if scores[i] < 0.3:
                continue
            det = tuple(int(x) for x in det)
            cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
            cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
        cv2.imwrite(os.path.join('demo', 'out.jpg'), im2show)

    main()
