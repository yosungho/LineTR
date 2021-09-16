#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
#  Modified by Sungho Yoon (for LineTR demo)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import cv2
# import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_pl_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LineTR demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='4',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_conf_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--match_threshold', type=float, default=0.8,
        help='LineTR NN match threshold')
    parser.add_argument(
        '--auto_min_length', action='store_false',
        help='Adjust minimum line length depending on image size')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    config = {
        'auto_min_length': opt.auto_min_length,

        'superpoint': {
            'nms_radius': 4,
            'keypoint_conf_threshold': 0.005,
            'max_keypoints': 1024,
            'nn_threshold': 0.7,
        },
        'lsd': {
            'n_octave': 2, 
        },
        'linetransformer': {
            'image_shape': [480, 640],
            'max_keylines': -1,
            'min_length': 8,
            'token_distance': 16,
            'nn_threshold': 0.8,
        },
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors','klines', 'line_desc', 'mat_klines2sublines']

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data_sp = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data_sp[k] for k in keys if k in last_data_sp.keys()}
    last_data['image0'] = frame_tensor

    # klines_cv = matching.lsd.detect(frame)
    klines_cv = matching.lsd.detect_torch(frame_tensor)
    klines0 = matching.linetransformer.preprocess(klines_cv, frame_tensor.shape, last_data_sp)
    klines0 = matching.linetransformer(klines0)
    last_data = {**last_data, **{k+'0': v for k, v in klines0.items()}}

    last_frame = frame
    last_image_id = 0

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('LineTR matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('LineTR matches', (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the nn match threshold for keypoints\n'
          '\tc/v: increase/decrease the nn match threshold for keylines\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_lineTR.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches_p'][0].cpu().numpy()
        confidence = pred['matching_scores_p'][0].cpu().numpy()
        matches_p = np.where(matches > 0)
        mkpts0 = kpts0[matches_p[0]]
        mkpts1 = kpts1[matches_p[1]]
        color = None

        klines0 = last_data['klines0'][0].cpu().numpy()
        klines1 = pred['klines1'][0].cpu().numpy()
        matches = pred['matches_l'][0].cpu().numpy()
        confidence = pred['matching_scores_l'][0].cpu().numpy()
        matches_l = np.where(matches > 0)
        mklines0 = klines0[matches_l[0]]
        mklines1 = klines1[matches_l[1]]
        timer.update('forward')

        text = [
            'LineTR',
            'Keylines: {}:{}'.format(len(klines0), len(klines1)),
            'Matches: {}'.format(len(mklines0))
        ]

        kpt_conf_thresh = matching.superpoint.config['keypoint_conf_threshold']
        small_text = [
            'Keypoint Conf_Threshold: {:.4f}'.format(kpt_conf_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        out = make_pl_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, klines0, klines1, mklines0, mklines1, 
            color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

        if not opt.no_display:
            cv2.imshow('LineTR matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_LineTR.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_conf_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_conf_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_conf_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease keypoint NN threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superpoint.config['nn_threshold'] = min(max(
                    0.05, matching.superpoint.config['nn_threshold']+d), .95)
                print('\nChanged the kpoint match threshold to {:.2f}'.format(
                    matching.superpoint.config['nn_threshold']))
            elif key in ['c', 'v']:
                # Increase/decrease keyline NN threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'c' else 1)
                matching.linetransformer.config['nn_threshold'] = min(max(
                    0.05, matching.linetransformer.config['nn_threshold']+d), .95)
                print('\nChanged the kline match threshold to {:.2f}'.format(
                    matching.linetransformer.config['nn_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()

        if opt.output_dir is not None:
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vs.cleanup()
