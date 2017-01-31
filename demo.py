import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer


def test():
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


if __name__ == '__main__':
    test()