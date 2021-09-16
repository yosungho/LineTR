import cv2

class LSD():
    default_config = {
        'n_octave': 2,
        'scale': 2,
    }

    def __init__(self, config, sample_descriptors):
        self.config = {**self.default_config, **config}
        self.lsd = cv2.line_descriptor.LSDDetector_createLSDDetector()

    def detect(self, image):
        klines_cv2 = self.lsd.detect(image, self.config['scale'], self.config['n_octave'])
        return klines_cv2

    def detect_torch(self, image):
        image = (image*255).cpu().numpy().squeeze().astype('uint8')
        klines_cv2 = self.lsd.detect(image, self.config['scale'], self.config['n_octave'])
        return klines_cv2
