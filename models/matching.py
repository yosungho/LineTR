import torch

from .superpoint import SuperPoint
from .line_detector import LSD #, ELSED
from .line_transformer import LineTransformer, get_dist_matrix
from .nn_matcher import nn_matcher, nn_matcher_distmat

class Matching(torch.nn.Module):
    """ Image Matching with SuperPoint & LineTR """
    def __init__(self, config={}):
        super().__init__()
        self.auto_min_length = config['auto_min_length']
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.lsd = LSD(config.get('lsd', {}))
        # self.elsed = ELSED(config.get('elsed', {}))
        self.linetransformer = LineTransformer(config.get('linetransformer', {}))

    def forward(self, data):
        pred = {}
        if 'keypoints0' not in data:
            pred_sp0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred_sp0.items()}}

        if 'keypoints1' not in data:
            pred_sp1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred_sp1.items()}}        
        
        if 'klines0' not in data:
            image_shape = data['image0'].shape
            if self.auto_min_length:
                self.linetransformer.config['min_length'] = max(16, max(image_shape)/40)
                self.linetransformer.config['token_distance'] = max(8, max(image_shape)/80)

            klines_cv = self.lsd.detect_torch(data['image0'])
            # klines_cv = self.elsed.detect_torch(data['image0'])
            
            if not 'valid_mask0' in data.keys():
                data['valid_mask0'] = torch.ones_like(data['image0']).to(data['image0'])
            valid_mask0 = data['valid_mask0']
            klines0 = self.linetransformer.preprocess(klines_cv, image_shape, pred_sp0, valid_mask0)
            klines0 = self.linetransformer(klines0)
            pred = {**pred, **{k+'0': v for k, v in klines0.items()}}
    
        if 'klines1' not in data:
            image_shape = data['image1'].shape
            if self.auto_min_length:
                self.linetransformer.config['min_length'] = max(16, max(image_shape)/40)
                self.linetransformer.config['token_distance'] = max(8, max(image_shape)/80)
            
            # data['image1'] = torch.zeros_like(data['image1']).to(data['image1'])
            klines_cv = self.lsd.detect_torch(data['image1'])
            n_lines = len(klines_cv)
            
            if not 'valid_mask1' in data.keys():
                data['valid_mask1'] = torch.ones_like(data['image1']).to(data['image1'])
            # klines_cv = self.elsed.detect_torch(data['image1'])
            valid_mask1 = data['valid_mask1']
            klines1 = self.linetransformer.preprocess(klines_cv, image_shape, pred_sp1, valid_mask1)
            klines1 = self.linetransformer(klines1)
            pred = {**pred, **{k+'1': v for k, v in klines1.items()}}
            
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        ## Feature Point Matching using Nearest Neighbor
        desc0_pnt = data['descriptors0'][0].cpu().numpy()
        desc1_pnt = data['descriptors1'][0].cpu().numpy()
        match_mat_pnt, dist_mat_pnt = nn_matcher(desc0_pnt, desc1_pnt, self.superpoint.config['nn_threshold'], is_mutual_NN=True)
        
        pred['matches_p'] = torch.from_numpy(match_mat_pnt)
        pred['matching_scores_p'] = torch.from_numpy(dist_mat_pnt)

        ## Feature Line Matching using Nearest Neighbor
        desc_slines0 = data['line_desc0'].cpu().numpy()
        desc_slines1 = data['line_desc1'].cpu().numpy()
        distance_sublines = get_dist_matrix(desc_slines0, desc_slines1)[0]
        distance_matrix = self.linetransformer.subline2keyline(distance_sublines, data['mat_klines2sublines0'][0], data['mat_klines2sublines1'][0])
        match_mat = nn_matcher_distmat(distance_matrix, self.linetransformer.config['nn_threshold'], is_mutual_NN=True)

        pred['matches_l'] = torch.from_numpy(match_mat)
        pred['matching_scores_l'] = torch.from_numpy(distance_matrix)
       
        return pred
