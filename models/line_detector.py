import cv2
import math
# import pyelsed

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LSD():
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