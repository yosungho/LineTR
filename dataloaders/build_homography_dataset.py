import os
import numpy as np
import cv2
import yaml
import h5py
from pathlib import Path
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist
from tqdm import tqdm

from utils.util_lines import find_line_matches, calculate_line_overlaps, conv_fixed_size

import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from models.superpoint import SuperPoint
from models.line_detector import LSD 
from models.line_process import preprocess, line_tokenizer

import multiprocessing as mp

def seed_everything(seed=1004):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def remove_out_edge(pred, valid_mask):
    kpt = pred['keypoints'][0]
    desc = pred['descriptors'][0]
    scores = pred['scores'][0]

    # remove boundary keypoints
    inlier_idx = []
    for ifeat in range(len(kpt)):
        if valid_mask[0][int(kpt[ifeat,1]),int(kpt[ifeat,0])] > 0.99: # (== 1.) inside of the image
            inlier_idx.append(ifeat)

    pred['keypoints'][0] = kpt[inlier_idx,:]
    pred['descriptors'][0] = desc[:,inlier_idx]
    pred['scores'] = list(pred['scores'])
    pred['scores'][0] = scores[inlier_idx]

    return pred 

class Homography_Dataset(Dataset):
    def __init__(self, conf, device):
            
        super().__init__()
        seed_everything()

        self.conf = conf
        self.device = device #conf['data']['device']
        self.is_photometric_aug = conf['augmentation']['photometric']['enable']
        self.max_sublines = conf['feature']['linetr']['max_sublines']
        self.thred_reprojected_l = conf['feature']['linetr']['thred_reprojected']
        self.thred_angdiff = conf['feature']['linetr']['thred_angdiff']
        self.min_overlap_ratio = conf['feature']['linetr']['min_overlap_ratio']
        
        self.image_path = Path(conf['data']['image_path'])
        self.output_path = Path(conf['data']['output_path'])
        self.resize = conf['data']['resize']
        
        self.superpoint = SuperPoint(conf['feature']['superpoint']).to(self.device).eval()
        self.linedetector = LSD(conf['feature']['linetr'])
        
        self.image_paths = sorted(Path(self.image_path).glob(conf['data']['image_type']))

        self.init_var()

    def init_var(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        from dataloaders.utils.homographies import sample_homography_np as sample_homography
        from dataloaders.utils.utils import compute_valid_mask
        from dataloaders.utils.photometric import ImgAugTransform, customizedTransform
        # from .utils.utils import inv_warp_image, inv_warp_image_batch, warp_points
        
        self.sample_homography = sample_homography
        # self.inv_warp_image = inv_warp_image
        # self.inv_warp_image_batch = inv_warp_image_batch
        self.compute_valid_mask = compute_valid_mask
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform
        # self.warp_points = warp_points
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, iter, idx):
        def read_data(path, device, resize):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            if resize[0] != -1:
                image = cv2.resize(image.astype('float32'), (resize[0], resize[1]))
                gray = cv2.resize(gray.astype('uint8'), (resize[0], resize[1]))
            gray_torch = torch.from_numpy(gray/255.).float()[None, None].to(device)
            
            return image, gray, gray_torch
        
        def imgPhotometric(img_o, num_batch):
            """
            :param img:
                numpy (H, W)
            :return:
            """
            augmentation = self.ImgAugTransform(**self.conf['augmentation'])
            cusAug = self.customizedTransform()
            img_o = img_o[:,:,np.newaxis]

            img_batch = np.empty((num_batch,1,img_o.shape[0],img_o.shape[1]))
            for i in range(num_batch):
                img = augmentation(img_o)
                img = cusAug(img, **self.conf['augmentation'])

                img_batch[i] = img.squeeze()
            return img_batch
        
        def scale_homography(H, shape, shift=(-1,-1)):
            height, width = shape[0], shape[1]
            trans = np.array([[2./width, 0., shift[0]], [0., 2./height, shift[1]], [0., 0., 1.]])
            H_tf = np.linalg.inv(trans) @ H @ trans
            return H_tf
        
        ##################################################################################################3
        file_path = os.path.join(self.output_path, f'data_{idx}_{iter}.h5')
        if os.path.exists(file_path):
            return 

        ## load an image
        image0_path = self.image_paths[idx]
        image0, gray0, _ = read_data(image0_path, self.device, self.resize)

        ## load another image for backgrounding
        idx_bg = np.random.randint(0, self.__len__())
        background_path = self.image_paths[idx_bg]
        image_bg, gray_bg, _ = read_data(background_path, self.device, self.resize)
        
        ###
        homoAdapt_iter = 1
        homographies = np.stack([self.sample_homography(np.array([2, 2]), shift=-1,
                **self.conf['augmentation']['homographic']['params'])
                for i in range(homoAdapt_iter)])
        inv_homographies = np.stack([np.linalg.inv(h_mat) for h_mat in homographies])
        
        homographies = scale_homography(homographies, gray0.shape, shift=(-1,-1))
        inv_homographies = scale_homography(inv_homographies, gray0.shape, shift=(-1,-1))
        
        # applying random photometric distortions to real images
        if (self.is_photometric_aug):
            num_pairs=1
            gray1 = gray0.copy()
            gray0 = (imgPhotometric(gray0/255., num_pairs)[0][0]*255).astype('uint8')      # gray image with photometric distortion
            gray1 = (imgPhotometric(gray1/255., num_pairs)[0][0]*255).astype('uint8')      # gray image with photometric distortion
        else:
            gray0 = gray0
            gray1 = gray0.copy()
            
        
        height, width = image_shape = gray0.shape
        gray1 = gray0.copy()
        gray1 = cv2.warpPerspective(gray1, inv_homographies.squeeze(), (width, height))
        warped_img = cv2.warpPerspective(image0, inv_homographies.squeeze(), (width, height))
        
        image0_torch = torch.from_numpy(gray0/255.).float()[None, None].to(self.device)
        image1_torch = torch.from_numpy(gray1/255.).float()[None, None].to(self.device)
        
        # cv2.imwrite("debug/gray0_"+str(idx)+".png", gray0)
        # cv2.imwrite("debug/gray1_"+str(idx)+".png", gray1)

#####################################################################################################################

        htrans = inv_homographies[0]
        
        valid_mask0 = np.ones_like(gray0)    # 1: valid, 0: invalid
        valid_mask1 = np.ones_like(gray1)
        mask = (warped_img==0)[:,:,0] * (warped_img==0)[:,:,1] * (warped_img==0)[:,:,2]
        valid_mask1[mask] = 0
        kernel = np.ones((5, 5), np.uint8)
        valid_mask1 = cv2.erode(valid_mask1, kernel)
        
        #################################################################################3
        ## feature extraction
        ## point 
        with torch.no_grad():
            pred0 = self.superpoint({'image': image0_torch})
            pred1 = self.superpoint({'image': image1_torch})
            
        ## line
        klns0_cv = self.linedetector.detect(gray0)
        klns1_cv = self.linedetector.detect(gray1)

        try:
            klines0 = preprocess(klns0_cv, image_shape, pred0, mask=valid_mask0, conf=self.conf['feature']['linetr'])   ## TODO: torch vs. np. 잘 정리하기, tokenizer 다시 정리
            klines1 = preprocess(klns1_cv, image_shape, pred1, mask=valid_mask1, conf=self.conf['feature']['linetr'])
        except:
            return
        ###################################################################################
        
        ## insert pseudo lines    
        klines0 = conv_fixed_size(klines0, self.conf, func_token=line_tokenizer, pred_sp=pred0)
        klines1 = conv_fixed_size(klines1, self.conf, func_token=line_tokenizer, pred_sp=pred1)
        
        #############################################################################        
        
        klns0 = klines0['sublines'].reshape(-1, 2, 2).cpu().numpy()
        klns1 = klines1['sublines'].reshape(-1, 2, 2).cpu().numpy()
              
        try:
            klns0_projected = cv2.perspectiveTransform(klns0.reshape((1, -1, 2)), htrans)[0, :, :] #h_mat_inv
            klns0_projected = klns0_projected.reshape(-1,2,2)
            
            klns1_projected = cv2.perspectiveTransform(klns1.reshape((1, -1, 2)), np.linalg.inv(htrans))[0, :, :] #h_mat_inv
            klns1_projected = klns1_projected.reshape(-1,2,2)
        except:
            return
        
        mat_matches0 = find_line_matches(klns0, klns1_projected, self.thred_reprojected_l, self.thred_angdiff)
        mat_matches1 = find_line_matches(klns1, klns0_projected, self.thred_reprojected_l, self.thred_angdiff)
        mat_matches1 = mat_matches1.T
        b_matches = (np.logical_and(mat_matches0 > 0, mat_matches1 > 0))
        
        lmatches0 = np.array(np.where(b_matches)).T
        lmatches1 = np.zeros_like(lmatches0)
        lmatches1[:,0], lmatches1[:,1] = lmatches0[:,1], lmatches0[:,0]
        mat_overlap0, overlaps0 = calculate_line_overlaps(klns0, klns1_projected, lmatches0)
        mat_overlap1, overlaps1  = calculate_line_overlaps(klns1, klns0_projected, lmatches1)
        mat_overlap1 = mat_overlap1.T
        mat_assign_sublines = np.where(mat_overlap0 > mat_overlap1, mat_overlap0, mat_overlap1)
        lmatches = np.array(np.where(mat_assign_sublines > self.min_overlap_ratio)).T
        
        lmatches_ret = np.ones((int(self.max_sublines*1.5),2))*-1
        lmatches_ret[:len(lmatches)] = lmatches
        
        
        ## visualization #######################################################################################
        if self.conf['data']['visualize'] == True:
            green = (0, 255, 0)
            red = (0, 0, 255)
            dbg0 = cv2.cvtColor(gray0.copy(), cv2.COLOR_GRAY2RGB)
            dbg1 = cv2.cvtColor(gray1.copy(), cv2.COLOR_GRAY2RGB)
            out = np.hstack((dbg0, dbg1))

            for (s, ss) in zip(klns0.reshape(-1,4)[lmatches[:,0]].astype(np.int32), klns1.reshape(-1,4)[lmatches[:,1]].astype(np.int32)):
                cv2.line(out, (s[0], s[1]), (s[2], s[3]), green, 1, cv2.LINE_AA)
                cv2.line(out, (ss[0]+640, ss[1]), (ss[2]+640, ss[3]), green, 1, cv2.LINE_AA)
                mid_point0 = (int((s[0]+s[2])/2), int((s[1]+s[3])/2))
                mid_point1 = (int((ss[0]+ss[2])/2+640), int((ss[1]+ss[3])/2))
                cv2.line(out, mid_point0, mid_point1, red, 1, cv2.LINE_AA)
            cv2.imwrite(str(self.output_path / Path('line'+str(idx)+'_'+str(iter)+'.jpg')), out)
        
        #######################################################################################################        
        keys_l = ['klines', 'sublines', 'angle_sublines','pnt_sublines', 'desc_sublines', \
            'score_sublines', 'resp_sublines', 'mask_sublines', 'num_klns', 'mat_klines2sublines', 'num_slns']
        # klines, resp, angle, pnt, desc, score, mask
        klines0 = {k+'0':v[0].cpu().numpy() for k,v in klines0.items() if k in keys_l}
        klines1 = {k+'1':v[0].cpu().numpy() for k,v in klines1.items() if k in keys_l}
        
        ret = {}
        ret = {**ret, **klines0, **klines1}
        ret = {**ret, **{'lmatches':lmatches_ret}}

        with h5py.File(str(file_path), 'a') as fd:
            try:
                for k, v in ret.items():
                    fd.create_dataset(k, data=v)
            except OSError as error:
                    raise error
                

def main():
    conf_path = Path('dataloaders/confs/homography.yaml')
    with open(conf_path, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    
    n_images = len(list(Path(conf['data']['image_path']).glob(conf['data']['image_type'])))
    
    choose_worker = conf['data']['choose_worker']
    nWorkers = conf['data']['nWorkers']
    list_of_range_list = [
        (k, x.tolist()) for k, x in enumerate(np.array_split(range(0, n_images), nWorkers))
    ]
    range_list = list_of_range_list[choose_worker]
    device_num = range_list[0]
    device = torch.device('cuda:'+str(device_num) if torch.cuda.is_available() else "cpu")
    dataset = Homography_Dataset(conf, device)
    
    for iter in range(conf['data']['n_iters']):
        for idx in tqdm(range_list[1]):
            try:
                print(idx)
                dataset.__getitem__(iter, idx)
            except KeyboardInterrupt:
                print("pressed control + c")
                return

if __name__ == '__main__':
    main()
        
        