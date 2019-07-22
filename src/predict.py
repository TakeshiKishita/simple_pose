from __future__ import division
import cv2

import mxnet as mx

import gluoncv as gcv
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints

from util.video_read import VideoReader

ctx = mx.cpu()
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})

estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)

# 入力を設定（デバイス番号、URL等）
SOURCE = 0

# インスタンスの初期化
c = VideoReader(SOURCE)
c.open()
try:
    for frame in c.read():
        # 出力内容をディスプレイに表示

        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

        x, frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=512, max_size=350)
        x = x.as_in_context(ctx)
        class_IDs, scores, bounding_boxs = detector(x)

        pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                           output_shape=(128, 96), ctx=ctx)
        if len(upscale_bbox) > 0:
            predicted_heatmap = estimator(pose_input)
            pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

            frame = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                      box_thresh=0.5, keypoint_thresh=0.2)
        cv_plot_image(frame)
        cv2.waitKey(1)
except KeyboardInterrupt:
    print('pushed Ctrl-C')
finally:
    # 終了処理
    c.close()
    cv2.destroyAllWindows()
