from pathlib import Path
import argparse
import torch
import numpy as np
import cv2
import random
from models.matching import Matching

torch.set_grad_enabled(False)

def read_image(path, device, resize):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if len(resize)==2 and resize[0]!=-1:
        image = cv2.resize(image.astype('float32'), (resize[0], resize[1]))
    image_torch = torch.from_numpy(image/255.).float()[None, None].to(device)
    return image, image_torch
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Line matching with LineTR',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_pairs', type=str, default='assets/input_pairs.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='assets/outputs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If -1, do not resize')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--auto_min_length', action='store_false',
        help='set minimum line length to be 1/30 of maximum image size.')
    opt = parser.parse_args()

    # Load the SuperPoint and Line-Transformer models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running with \"{}\"'.format(device))

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))

    config = {
        'auto_min_length': opt.auto_min_length,

        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
            'nn_threshold': 0.7,
        },
        'lsd': {
            'n_octave': 2, 
        },
        'linetransformer': {
            'max_keylines': -1,
            'min_length': 16,
            'token_distance': 8,
            'nn_threshold': 0.8,
        },
    }

    matching = Matching(config).eval().to(device)

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem

        # Load the image pair.
        image0, image_torch0 = read_image(input_dir / name0, device, opt.resize)
        image1, image_torch1 = read_image(input_dir / name1, device, opt.resize)

        # Perform the matching.
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        pred_matches = matching({'image0': image_torch0, 'image1': image_torch1})
        pred = {k: v[0].cpu().numpy() for k, v in pred_matches.items() if torch.is_tensor(v[0])}
        pred = {**pred, **{k: v[0] for k, v in pred_matches.items() if not torch.is_tensor(v[0])}}

        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches_kpt, confidence_kpt = pred['matches_p'], pred['matching_scores_p']
        kls0, kls1 = pred['klines0'], pred['klines1']
        matches_kls, confidence_kls = pred['matches_l'], pred['matching_scores_l']

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                        'matches_p': matches_kpt, 'match_confidence_p': confidence_kpt,
                        'keylines0': kls0, 'keylines1': kls1,
                        'matches_l': matches_kls, 'match_confidence_l': confidence_kls,}
        np.savez(str(matches_path), **out_matches)

        # Keep the matching keylines.
        matches_l = np.where(matches_kls > 0)
        mklines0 = kls0[matches_l[0]]
        mklines1 = kls1[matches_l[1]]

        # Keep the matching keypoints.
        matches_p = np.where(matches_kpt > 0)
        mkpts0 = kpts0[matches_p[0]]
        mkpts1 = kpts1[matches_p[1]]

        # visualization
        margin = 10
        H0, W0 = image0.shape
        H1, W1 = image1.shape
        H, W = max(H0, H1), W0 + W1 + margin

        out = 255*np.ones((H, W), np.uint8)
        out[:H0, :W0] = image0
        out[:H1, W0+margin:] = image1
        out = np.stack([out]*3, -1)

        thickness = 2
        matches_l = np.where(matches_kls>0)
        mklines0 = kls0[matches_l[0]]
        mklines1 = kls1[matches_l[1]]

        for i, (kline0, kline1) in enumerate(zip(mklines0,mklines1)):
            rnd_color = (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256))

            start_point = (int(kline0[0,0]), int(kline0[0,1]))
            end_point = (int(kline0[1,0]), int(kline0[1,1]))
            cv2.line(out, start_point, end_point, rnd_color, thickness, cv2.LINE_AA)
            cv2.circle(out, start_point, 3, rnd_color, thickness, cv2.LINE_AA)
            cv2.circle(out, end_point, 3, rnd_color, thickness, cv2.LINE_AA)
            mid_point0 = (int((start_point[0]+end_point[0])/2), int((start_point[1]+end_point[1])/2))

            start_point = (int(kline1[0,0])+ margin + W0, int(kline1[0,1]))
            end_point = (int(kline1[1,0])+ margin + W0, int(kline1[1,1]))
            cv2.line(out, start_point, end_point, rnd_color, thickness, cv2.LINE_AA)
            cv2.circle(out, start_point, 3, rnd_color, thickness, cv2.LINE_AA)
            cv2.circle(out, end_point, 3, rnd_color, thickness, cv2.LINE_AA)            
            mid_point1 = (int((start_point[0]+end_point[0])/2), int((start_point[1]+end_point[1])/2))
            
            # cv2.line(out, mid_point0, mid_point1, rnd_color, 1, cv2.LINE_AA)

        # for i, (kpt0, kpt1) in enumerate(zip(mkpts0,mkpts1)):
        #     rnd_color = (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256))

        #     point0 = (int(kpt0[0]), int(kpt0[1]))
        #     point1 = (int(kpt1[0])+ margin + W0, int(kpt1[1]))
        #     cv2.circle(out, point0, 3, rnd_color, thickness, cv2.LINE_AA)
        #     cv2.circle(out, point1, 3, rnd_color, thickness, cv2.LINE_AA)

        #     # cv2.line(out, point0, point1, rnd_color, 1, cv2.LINE_AA)

        img_path = output_dir / '{}_{}_matches.png'.format(stem0, stem1)
        cv2.imwrite(str(img_path), out)