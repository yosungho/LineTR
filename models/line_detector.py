import cv2
import math
# import pyelsed

import numpy as np
import tensorflow as tf
from .thirdparty.mlsd.utils import pred_lines


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LSD:
    default_config = {
        'n_octave': 2,
        'scale': 2,
    }

    def __init__(self, config):
        self.config = {**self.default_config, **config}
        self.lsd = cv2.line_descriptor.LSDDetector_createLSDDetector()

    def detect(self, image):
        klines_cv2 = self.lsd.detect(image, self.config['scale'], self.config['n_octave'])
        return klines_cv2

    def detect_torch(self, image):
        image = (image*255).cpu().numpy().squeeze().astype('uint8')
        klines_cv2 = self.lsd.detect(image, self.config['scale'], self.config['n_octave'])
        return klines_cv2


class MLSD:
    default_config = {
        'model_name': 'models/thirdparty/mlsd/tflite_models/M-LSD_512_large_fp32.tflite',
        'input_shape': [512, 512],
        'score_thr': 0.2,
        'dist_thr': 10
    }

    def __init__(self, config):
        self.config = {**self.default_config, **config}
        self.model_name = self.config['model_name']
    
        self.interpreter = tf.lite.Interpreter(model_path=self.model_name)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image):
        klines_mlsd = pred_lines(image, self.interpreter, self.input_details, self.output_details,
                                 input_shape=self.config['input_shape'], score_thr=self.config['score_thr'],
                                 dist_thr=self.config['dist_thr'])  # (num_lines, 4)

        klines_mlsd = klines_mlsd.reshape(-1, 2, 2)
        klines_mlsd[..., 0] = np.where(klines_mlsd[..., 0] < 0, 0, klines_mlsd[..., 0])
        klines_mlsd[..., 1] = np.where(klines_mlsd[..., 0] < 0, 0, klines_mlsd[..., 1])
        klines_mlsd[..., 0] = np.where(klines_mlsd[..., 0] >= image.shape[1], image.shape[1], klines_mlsd[..., 0])
        klines_mlsd[..., 1] = np.where(klines_mlsd[..., 1] >= image.shape[0], image.shape[0], klines_mlsd[..., 1])

        return klines_mlsd

    def detect_torch(self, image):
        image = (image * 255).cpu().numpy().astype('uint8')
        return self.detect(image)

# class ELSED():
#     default_config = {
#         'n_octave': 1,
#         'scale': 1,
#     }

#     def __init__(self, config):
#         self.config = {**self.default_config, **config}

#     def detect(self, image):
#         segments, scores = pyelsed.detect(image)

#         klines_cv2 = []
#         for (seg, score) in zip(segments, scores):
#             kline={}
#             kline['startPointX'] = seg[0]
#             kline['startPointY'] = seg[1]
#             kline['endPointX'] = seg[2]
#             kline['endPointY'] = seg[3]
#             kline['lineLength'] = math.sqrt((kline['endPointX']-kline['startPointX'])**2+(kline['endPointY']-kline['startPointY'])**2)
#             kline['octave'] = 0
#             kline['score'] = score
#             kline_cv2 = dotdict(kline)
#             klines_cv2.append(kline_cv2)

#         return klines_cv2

#     def detect_torch(self, image):
#         image = (image*255).cpu().numpy().squeeze().astype('uint8')
#         segments, scores = pyelsed.detect(image)

#         klines_cv2 = []
#         for seg in segments:
#             kline={}
#             kline['startPointX'] = seg[0]
#             kline['startPointY'] = seg[1]
#             kline['endPointX'] = seg[2]
#             kline['endPointY'] = seg[3]
#             kline['lineLength'] = math.sqrt((kline['endPointX']-kline['startPointX'])**2+(kline['endPointY']-kline['startPointY'])**2)
#             kline['octave'] = 0
#             kline_cv2 = dotdict(kline)
#             klines_cv2.append(kline_cv2)

#         return klines_cv2