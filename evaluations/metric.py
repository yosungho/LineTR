import numpy as np
import cv2
import torch
from .evaluate_pr import Evaluate_PR
from .matcher import nn_matcher_batches

class Result(object):
    def __init__(self, mode, args):
        self.args = args
        self.type_dataset = args.dataset_type   # homography
        self.mode = mode
        self.eval_pr = Evaluate_PR(args)
        self.set_to_worst()
        self.nn_thresh = args.nn_thresh

    def evaluate(self, pred, batch_data, loss = 0, batch_idx=0):
        self.loss = loss
        pred_ret = pred

        keys2eval = ['num_sublines0', 'num_sublines1', 'mat_assign_sublines', \
                        'sublines0', 'sublines1', 'sublines0_3d', 'sublines1_3d', \
                            'num_klines0', 'num_klines1', 'mat_assign_klines', 'klines0', 'klines1',\
                                'mat_klines2sublines0', 'mat_klines2sublines1',\
                                'intrinsic', 'cam0_T_w', 'cam1_T_w', \
                                    'keypoints0', 'keypoints1', 'keypoints0_3d', 'keypoints1_3d', \
                                        'num_keypoints0', 'num_keypoints1', 'descriptors0', 'descriptors1', 'scene_id', \
                                            'homographies', 'inv_homographies', 'image0_shape', 'image1_shape',\
                                                'image0', 'image1','assign_mat_gt',\
                                                    ]
        with torch.no_grad():
            pred = {k: v.cpu().numpy() for k, v in pred.items()}
            batch_data = {k: batch_data[k].cpu().numpy() for k in keys2eval if k in batch_data}

        # matcher
        desc0 = pred['line_desc0']      # [32, 256, 128]
        desc1 = pred['line_desc1']      # [32, 256, 128]
        score_pred = nn_matcher_batches(desc0, desc1, self.nn_thresh, is_mutual_NN=self.args.mutual_nn)
        score_pred = score_pred[:,:-1,:-1]
        score_gt = batch_data['mat_assign_sublines'][:,:-1,:-1]
        precision, recall, f1_score = self.eval_pr.get_precision_recall(score_pred, score_gt)
    
        self.precision.extend(precision)
        self.recall.extend(recall)
        self.f1_score.extend(f1_score)

        return pred_ret

    def update(self, precision, recall, f1_score, gpu_time, loss):
        # 'homography'
        self.precision = [precision]
        self.recall = [recall]
        self.f1_score = [f1_score]

        self.gpu_time = gpu_time
        self.loss = loss

    def set_to_worst(self):
        # common 
        self.precision = []
        self.recall = []
        self.f1_score = []
        
        self.gpu_time = 0
        self.loss = 0

    def set_to_worst_for_logger(self):
        # common 
        self.precision = [0]
        self.recall = [0]
        self.f1_score = [0]
        
        self.gpu_time = 0
        self.loss = 0

class AverageMeter(object):
    def __init__(self, args):
        self.args = args
        self.reset()

    def reset(self):
        self.count = 0.0

        # common 
        self.sum_precision = 0
        self.sum_recall = 0
        self.sum_f1_score = 0
        
        self.sum_gpu_time = 0
        self.sum_loss = 0

    def update(self, result, gpu_time, n=1):
        self.count += n     # n = batch_size

        # homography
        self.sum_precision += np.sum(result.precision)
        self.sum_recall += np.sum(result.recall)
        self.sum_f1_score += np.sum(result.f1_score)

        # others        
        self.sum_gpu_time += n * gpu_time
        self.sum_loss += n * result.loss

    def average(self):
        avg = Result('test', self.args)
        if self.count > 0:
            avg.update(
                self.sum_precision / self.count, 
                self.sum_recall / self.count, 
                self.sum_f1_score / self.count, 
                self.sum_gpu_time / self.count,
                self.sum_loss / self.count)
        return avg

# rgb_images[0,1]
# lines[0,1]
# pairs[matches, mis_matches, no_matches]
def save_matching_image(f_name, rgb_images, lines, pairs):
    image0 = cv2.cvtColor(rgb_images[0], cv2.COLOR_BGR2GRAY)
    image1 = cv2.cvtColor(rgb_images[1], cv2.COLOR_BGR2GRAY)

    margin = 3
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    # putText
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .3
    thickness = 1

    # line
    green = (0, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)

    color = blue
    for i, pair in enumerate(pairs['no_matches'][0]):
        kline0 = lines[0][pair]
        start_point = (int(kline0[0,0]), int(kline0[0,1]))
        end_point = (int(kline0[1,0]), int(kline0[1,1]))
        # cv2.circle(out, start_point, 3, color, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(out, end_point, 3, color, thickness, lineType=cv2.LINE_AA)
        cv2.line(out, start_point, end_point, color, thickness)

    color = blue
    for i, pair in enumerate(pairs['no_matches'][1]):
        kline1 = lines[1][pair]
        start_point = (int(kline1[0,0])+ margin + W0, int(kline1[0,1]))
        end_point = (int(kline1[1,0])+ margin + W0, int(kline1[1,1]))
        # cv2.circle(out, start_point, 3, color, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(out, end_point, 3, color, thickness, lineType=cv2.LINE_AA)
        cv2.line(out, start_point, end_point, color, thickness)

    color = red
    for i, pair in enumerate(pairs['mis_matches']):
        kline0 = lines[0][pair[0]]
        start_point = (int(kline0[0,0]), int(kline0[0,1]))
        end_point = (int(kline0[1,0]), int(kline0[1,1]))
        # cv2.circle(out, start_point, 3, color, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(out, end_point, 3, color, thickness, lineType=cv2.LINE_AA)
        cv2.line(out, start_point, end_point, color, thickness)

        kline1 = lines[1][pair[1]]
        start_point = (int(kline1[0,0])+ margin + W0, int(kline1[0,1]))
        end_point = (int(kline1[1,0])+ margin + W0, int(kline1[1,1]))
        # cv2.circle(out, start_point, 3, color, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(out, end_point, 3, color, thickness, lineType=cv2.LINE_AA)
        cv2.line(out, start_point, end_point, color, thickness)

        mid_point1 = (int((kline0[0,0]+kline0[1,0])/2), int((kline0[0,1]+kline0[1,1])/2))
        mid_point2 = (int((kline1[0,0]+kline1[1,0])/2)+ margin + W0, int((kline1[0,1]+kline1[1,1])/2))
        cv2.line(out, mid_point1, mid_point2, color, thickness)

    color = green
    for i, pair in enumerate(pairs['matches']):
        kline0 = lines[0][pair[0]]
        start_point = (int(kline0[0,0]), int(kline0[0,1]))
        end_point = (int(kline0[1,0]), int(kline0[1,1]))
        # cv2.circle(out, start_point, 3, color, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(out, end_point, 3, color, thickness, lineType=cv2.LINE_AA)
        cv2.line(out, start_point, end_point, color, thickness)

        kline1 = lines[1][pair[1]]
        start_point = (int(kline1[0,0])+ margin + W0, int(kline1[0,1]))
        end_point = (int(kline1[1,0])+ margin + W0, int(kline1[1,1]))
        # cv2.circle(out, start_point, 3, color, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(out, end_point, 3, color, thickness, lineType=cv2.LINE_AA)
        cv2.line(out, start_point, end_point, color, thickness)

        mid_point1 = (int((kline0[0,0]+kline0[1,0])/2), int((kline0[0,1]+kline0[1,1])/2))
        mid_point2 = (int((kline1[0,0]+kline1[1,0])/2)+ margin + W0, int((kline1[0,1]+kline1[1,1])/2))
        cv2.line(out, mid_point1, mid_point2, color, thickness)

    cv2.imwrite(f_name, out)
    pass