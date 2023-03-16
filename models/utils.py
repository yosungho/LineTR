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
#  Modified by Sungho Yoon (for LineTR)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import os
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        origin_image = cv2.imread(impath)
        if origin_image is None:
            raise Exception('Error reading image %s' % impath)
        w, h = origin_image.shape[1], origin_image.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        rgb_image = cv2.resize(
            origin_image, (w_new, h_new), interpolation=self.interp)
        grayim = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        return grayim, rgb_image

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, None, False)
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, origin_image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, origin_image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, None, False)
            w, h = origin_image.shape[1], origin_image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            rgb_image = cv2.resize(origin_image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image, rgb_image = self.load_image(image_file)
        self.i = self.i + 1
        return (image, rgb_image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)

##########################################################################################3

def make_pl_matching_plot_fast(last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, 
                            klines0, klines1, mklines0, mklines1, 
                            color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = last_frame.shape
    H1, W1 = frame.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = last_frame
    out[:H1, W0+margin:] = frame
    out = np.stack([out]*3, -1)
    thickness = 2

    # line
    if klines0 is not None:
        green = (0, 255, 0)
        blue = (255, 0, 0)
        red = (0, 0, 255)
        for i, line in enumerate(klines0):
            cv2.line(out, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), 
                color=blue, thickness=1, lineType=cv2.LINE_AA)
        for i, line in enumerate(klines1):
            cv2.line(out, (int(line[0][0]) + margin + W0, int(line[0][1])), (int(line[1][0]) + margin + W0, int(line[1][1])), 
                color=blue, thickness=1, lineType=cv2.LINE_AA)
        for i, (line0, line1) in enumerate(zip(mklines0, mklines1)):
            cv2.line(out, (int(line0[0][0]), int(line0[0][1])), (int(line0[1][0]), int(line0[1][1])), 
            color=red, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.line(out, (int(line1[0][0]) + margin + W0, int(line1[0][1])), (int(line1[1][0]) + margin + W0, int(line1[1][1])), 
            color=red, thickness=thickness, lineType=cv2.LINE_AA)

            mid0 = (line0[0] + line0[1])/2
            mid1 = (line1[0] + line1[1])/2
            cv2.line(out, (int(mid0[0]), int(mid0[1])), (int(mid1[0]) + margin + W0, int(mid1[1])), 
            color=green, thickness=1, lineType=cv2.LINE_AA)
    
    # point
    if show_keypoints: 
        green = (0, 255, 0)
        blue = (255, 0, 0)
        red = (0, 0, 255)
        magenta = (255,255,0)
        white = (255, 255, 255)
        
        for i, (kpt0, kpt1) in enumerate(zip(mkpts0,mkpts1)):
            point0 = (int(kpt0[0]), int(kpt0[1]))
            point1 = (int(kpt1[0])+ margin + W0, int(kpt1[1]))
            cv2.circle(out, point0, 3, white, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, point1, 3, white, -1, lineType=cv2.LINE_AA)
            cv2.line(out, point0, point1, magenta, thickness=1)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


##################################################################################################
from evaluations.metric import Result
import csv
import shutil

def get_folder_name(args):
    current_time = time.strftime('%Y-%m-%d@%H-%M')
    prefix = "mode={}.".format(args.mode)

    return os.path.join(args.backup_path,
        prefix + 'input={}.lr={}.bs={}.wd={}.time={}'.
        format('linetr', args.lr, args.batch_size, args.wd, \
            current_time))

def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(".", "..", "..", ".git*", "*pycache*",
                                       "*build", "*.fuse*", "*_drive_*")
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)
    shutil.copytree('.', backup_directory, ignore=ignore_hidden)


def save_checkpoint(state, is_best, epoch, output_directory):
    import torch
    checkpoint_filename = os.path.join(output_directory,
                                       'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(
            output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)
            
class logger:
    def __init__(self, args, tensorboard_writer=None, prepare=True):
        self.args = args
        self.writer = tensorboard_writer
        output_directory = get_folder_name(args)
        self.output_directory = output_directory
        self.print_freq = 10
        self.rank_metric = args.rank_metric
        
        self.best_result = Result('test', args)
        self.best_result.set_to_worst_for_logger()
        
        if not prepare:
            return
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # # backup the source code
        # # if args.resume == '' or args.resume == False:
        # print("=> creating source code backup ...")
        # backup_directory = os.path.join(output_directory, "code_backup")
        # self.backup_directory = backup_directory
        # backup_source_code(backup_directory)
        # print("=> finished creating source code backup.")
        
        ## csv
        self.train_csv = os.path.join(output_directory, 'train.csv')
        self.val_csv = os.path.join(output_directory, 'val.csv')
        self.best_txt = os.path.join(output_directory, 'best.txt')
        self.fieldnames = [
            'epoch', 'precision', 'recall', 'f1_score', 'loss'
        ]
        # create new csv files with only header
        with open(self.train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
        with open(self.val_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
        

    def conditional_print(self, args, split, i, epoch, lr, n_set, blk_avg_meter, avg_meter, loss):
        if (i + 1) % self.print_freq == 0:
            avg = avg_meter.average()
            blk_avg = blk_avg_meter.average()

            print('=> output: {}'.format(self.output_directory))
            print(
                '{split} Epoch: {0} [{1}/{2}]\tlr={lr}\n\t'
                'Loss={blk_avg.loss:.1f}  '
                'Precision={blk_avg.precision[0]:.2f}({average.precision[0]:.2f})  '
                'Recall={blk_avg.recall[0]:.2f}({average.recall[0]:.2f})  '
                'F1_score={blk_avg.f1_score[0]:.2f}({average.f1_score[0]:.2f})  '
                # 'MScore={blk_avg.mscore:.2f}({average.mscore:.2f}) '
                't_infer={blk_avg.gpu_time:.3f}({average.gpu_time:.3f})\n'
                .format(epoch,
                        i + 1,
                        n_set,
                        lr=lr,
                        blk_avg=blk_avg,
                        average=avg,
                        split=split.capitalize(), 
                        loss=loss))
            blk_avg_meter.reset()

    def conditional_save_info(self, args, average_meter, epoch):
        avg = average_meter.average()
        if args.mode == "train":
            csvfile_name = self.train_csv
        elif args.mode == "val":
            csvfile_name = self.val_csv
        else:
            raise ValueError("wrong mode provided to logger")
        
        with open(csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if args.dataset_type == 'homography':
                writer.writerow({
                    'epoch': epoch,
                    'precision': avg.precision[0],
                    'recall': avg.recall[0],
                    'f1_score': avg.f1_score[0],
                    'loss': avg.loss
                })
            else:
                writer.writerow({
                    'epoch': epoch,
                    'precision': avg.precision[0],
                    'recall': avg.recall[0],
                    'f1_score': avg.f1_score[0],
                    'loss': avg.loss
                })
        return avg
    
    def save_single_txt(self, filename, result, epoch):
        with open(filename, 'w') as txtfile:
            txtfile.write(
                ("rank_metric={}\n" + "epoch={}\n" + "precision={:.3f}\n" +
                 "recall={:.3f}\n" + "f1_score={:.3f}\n"
                  + "t_gpu={:.4f}\n" + "loss={:.3f}\n")
                                        .format(self.rank_metric, epoch,
                                        result.precision[0], result.recall[0], result.f1_score[0], 
                                        result.gpu_time, result.loss))

    def save_best_txt(self, result, epoch):
        self.save_single_txt(self.best_txt, result, epoch)

    def get_ranking_error(self, result):
        return getattr(result, self.rank_metric)

    def rank_conditional_save_best(self, mode, result, epoch):
        score = self.get_ranking_error(result)
        best_score = self.get_ranking_error(self.best_result)

        is_best = np.max(score) > np.max(best_score)
        if is_best and (mode in ['val']):
            self.old_best_result = self.best_result
            self.best_result = result
            self.save_best_txt(result, epoch)
        return is_best
    
    def _get_image_name(self, mode, epoch, is_best=False):
        if mode == 'test':
            return self.output_directory + '/image_test.png'
        if mode == 'val':
            if is_best:
                return self.output_directory + '/image_val_best.png'
            else:
                return self.output_directory + '/image_val_' + str(
                    epoch) + '.png'

    def save_image_as_best(self, mode, epoch):
        if mode == 'val':
            filename = self._get_image_name(mode, epoch, is_best=True)
            # self.writer
            # make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, klines0, klines1, r2d2kpts0, r2d2kpts1,
            #            color, text, path, show_keypoints=False,
            #            fast_viz=False, opencv_display=False,
            #            opencv_title='matches', small_text=[]):

            # vis_utils.save_image(self.img_merge, filename)
    
    def conditional_summarize(self, args, mode, avg, is_best):
        print("\n*\nSummary of ", mode, "round")
        print(''
            'Precision={average.precision[0]:.3f}\n'
            'Recall={average.recall[0]:.3f}\n'
            'F1_score={average.f1_score[0]:.3f}\n'
            't_GPU={average.gpu_time:.3f}\n'
            'loss={average.loss:.3f}\n'.format(average=avg))
        if is_best and mode == "val":
            print("New best model by %s (was %.3f)" %
                (self.rank_metric,
                self.get_ranking_error(self.old_best_result)[0]))
        elif mode == "val":
            print("(best %s is %.3f)" %
                (self.rank_metric,
                self.get_ranking_error(self.best_result)[0]))
            print("*\n")
##################################################################################################