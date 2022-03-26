import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

loss_names = ['l1', 'l2']

class matching_loss(nn.Module):
    def __init__(self):
        super(matching_loss, self).__init__()
        self.epsilon = 1e-10
        self.lamda = 0.5

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        match_mask = (target[:,:-1,:-1] > 0).detach()
        P_bar = pred[:,:-1,:-1][match_mask]

        unmatch_mask0 = (target[:,:-1,-1] > 0).detach()
        P_bar0 = pred[:,:-1,-1][unmatch_mask0]

        unmatch_mask1 = (target[:,-1,:-1] > 0).detach()
        P_bar1 = pred[:,-1,:-1][unmatch_mask1]
    
        pos_loss = - torch.sum(torch.log(P_bar + self.epsilon))
        neg_loss0 = - torch.sum(torch.log(P_bar0 + self.epsilon))
        neg_loss1 = - torch.sum(torch.log(P_bar1 + self.epsilon))

        # self.loss = self.lamda*pos_loss + (1-self.lamda)*(neg_loss0+neg_loss1)
        self.loss = pos_loss + (neg_loss0+neg_loss1)

        return self.loss

### triplet loss
class descriptor_loss(nn.Module):
    def __init__(self):
        super(descriptor_loss, self).__init__()
        self.margin = 0.5
        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # def compute_distances(self, pred, target):

    #     overlap_gt = target['mat_assign_sublines'][:,:-1,:-1]
    #     mat_match = torch.where(overlap_gt > 0, 1, 0)
    #     mat_unmatch = torch.where(overlap_gt <= 0, 1, 0)

    #     desc0 = pred['line_desc0']      # [32, 256, 128]
    #     desc1 = pred['line_desc1']      # [32, 256, 128]
    #     scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
    #     mat_dist = 2-2*scores   # # [32, 256, 256]

    #     dists_pos = mat_dist * mat_match
    #     max_dist = 10000.0 #2.0
    #     dists_neg = mat_dist + max_dist * mat_match


    #     return dists_pos, mat_match, dists_neg, mat_unmatch
    
    def compute_distances(self, pred, target):
        desc0 = pred['line_desc0']      # [32, 256, 128]
        desc1 = pred['line_desc1']      # [32, 256, 128]
        # mat_score = pred['score_matrix_line']
        
        overlap_gt = target['mat_assign_sublines'][:,:-1,:-1]
        mat_match = torch.where(overlap_gt > 0.3, 1, 0)
        mat_unmatch = torch.where(overlap_gt <= 0, 1, 0)

        b, d, nd0 = desc0.shape
        nd1 = desc1.shape[2]

        # distance = torch.zeros((b,nd0,nd1)).to(overlap_gt)
        # for b_idx in np.arange(b):
        #     desc0_ = desc0[b_idx].T
        #     desc1_ = desc1[b_idx].T

        #     sum_sq0 = torch.sum(torch.square(desc0_), axis=1)[:,None]  # (1, 16, 128) -> (1, 128)
        #     sum_sq1 = torch.sum(torch.square(desc1_), axis=1)[:,None].T

        #     dmat = desc0_ @ desc1_.T    # [128, 128]
        #     distance[b_idx] = (sum_sq0 + sum_sq1 - 2.0 * dmat).clip(min=0)

        distance = torch.einsum('bdn,bdm->bnm', desc0, desc1)
        distance = 2-2*distance

        dists_pos = distance * mat_match    # [8, 128, 128]
        dists_pos = torch.cat((dists_pos, dists_pos.transpose(1,2)), dim=1)  # [16, 256, 128]
        dists_pos = torch.amax(dists_pos, dim=2).reshape(-1,1)   # [16, 256] -> [4096]
        idx_pos = torch.where(dists_pos>0.0)
        dists_pos_final = dists_pos[idx_pos]   # [1959]

        hardest_pos = torch.amax(dists_pos_final)

        ## negative distance를 구한다. 
        max_dist = 10000.0
        dists_neg = (distance * mat_unmatch) + (max_dist * (1-mat_unmatch))   # [16, 128, 128]
        dists_neg = torch.cat((dists_neg, dists_neg.transpose(1,2)), 1)  # [16, 256, 128]
        dists_neg = dists_neg.reshape(-1,desc0.shape[2]) # [4096, 128]
        dists_neg_candi = dists_neg[idx_pos[0],:]

        margin = 0.5
        # masks = torch.logical_and(dists_neg_candi < dists_pos_final.reshape(-1,1)+margin, dists_neg_candi > dists_pos_final.reshape(-1,1))
        masks = torch.logical_and(dists_neg_candi < dists_pos_final.reshape(-1,1)+margin, dists_neg_candi > dists_pos_final.reshape(-1,1))
        # masks = torch.logical_and(dists_neg_candi < margin, dists_neg_candi > 0)

        dists_neg_list = []
        valid_idx = []
        for i, (dist_neg, mask) in enumerate(zip(dists_neg_candi, masks)):
            neg = dist_neg[mask]
            if len(neg) == 0:
                continue
            else:
                valid_idx.append(i)

            # idx = torch.randint(len(neg),(1,))
            idx = torch.argmin(neg)
            dists_neg_list.append(neg[idx])
        dists_neg_final = torch.stack(dists_neg_list)
        dists_pos_final = dists_pos_final[valid_idx]

        # choices = torch.where(mask, dists_neg_candi, 0)
        # rand_idx = torch.randint(len(choices[0]),(len(dists_neg_candi),))
        # dists_neg_final = dists_neg_candi[choices[0][rand_idx], choices[1][rand_idx]]

        return dists_pos_final, dists_neg_final

    def compute_distances_hardnet(self, pred, target):
        def find_pos_neg(distance, mat_match, mat_unmatch):
            dists_pos = distance * mat_match    # [8, 128, 128]
            # dists_pos = torch.cat((dists_pos, dists_pos.transpose(1,2)), dim=1)  # [16, 256, 128]
            dists_pos = torch.max(dists_pos, dim=2)
            idx_anchor = torch.where(dists_pos.values > 0.0) # torch.Size([3, 512])
            # idx_pos = torch.where(dists_max>0.0)
            dists_pos_final = dists_pos.values[idx_anchor]   # [1959]
            idx_pos = dists_pos.indices[idx_anchor]   # [1959]

            # hardest_pos = torch.amax(dists_pos_final)

            ## negative distance를 구한다. 
            max_dist = 10000.0
            dists_neg = (distance * mat_unmatch) + (max_dist * (1-mat_unmatch))   # [16, 128, 128]

            dists_neg_list = []
            for i_bat, i_anc, i_pos in zip(idx_anchor[0], idx_anchor[1], idx_pos):
                min_anc = torch.amin(dists_neg[i_bat,i_anc])
                min_pos = torch.amin(dists_neg[i_bat,:,i_pos])
                min_neg = torch.min(min_anc, min_pos)
                dists_neg_list.append(min_neg)
            dists_neg_final = torch.stack(dists_neg_list)
            return dists_pos_final, dists_neg_final

        desc0 = pred['line_desc0']      # [32, 256, 128]
        desc1 = pred['line_desc1']      # [32, 256, 128]
        # mat_score = pred['score_matrix_line']
        
        overlap_gt = target['mat_assign_sublines'][:,:-1,:-1]
        mat_match = torch.where(overlap_gt > 0.3, 1, 0)
        mat_unmatch = torch.where(overlap_gt <= 0, 1, 0)

        b, d, nd0 = desc0.shape
        nd1 = desc1.shape[2]

        distance = torch.einsum('bdn,bdm->bnm', desc0, desc1)
        distance = 2-2*distance

        dists_pos0, dists_neg0 = find_pos_neg(distance, mat_match, mat_unmatch)
        dists_pos1, dists_neg1 = find_pos_neg(distance.transpose(1,2), mat_match.transpose(1,2), mat_unmatch.transpose(1,2))
        
        dists_pos = torch.cat((dists_pos0,dists_pos1))
        dists_neg = torch.cat((dists_neg0,dists_neg1))

        return dists_pos, dists_neg

    def forward(self, pred, target):
        
        # dists_pos, dists_neg = self.compute_distances_hardnet(pred, target)
        dists_pos, dists_neg = self.compute_distances(pred, target)

        # anchor, positive, negative = self.extract_triplet(pred, target)
        # t_loss = self.triplet_loss(anchor, positive, negative)

        batch_size = 1 #pred['klines0'].shape[0]
        hardest_positive = torch.amax(dists_pos) / batch_size
        hardest_negative = torch.amin(dists_neg) / batch_size
        # t_loss1 = F.relu(hardest_positive - dists_neg + 0.5)
        # t_loss2 = F.relu(dists_pos - hardest_negative + 0.5)
        # t_loss = F.relu(- hardest_negative + 0.5)
        # t_loss = torch.mean(torch.cat((t_loss2.squeeze(),t_loss2))) 
        t_loss = F.relu(dists_pos - dists_neg + 1.0)
        # t_loss = t_loss[t_loss>0] # 효과 없는거 같음.
        t_loss = t_loss.mean() / batch_size
        
        return t_loss, hardest_positive, hardest_negative


# class descriptor_loss(nn.Module):
#     def __init__(self):
#         super(descriptor_loss, self).__init__()
#         self.angular_margin = 0.5
#         self.scale = 64
    
#     def compute_distances(self, pred, target):
#         desc0 = pred['line_desc0']      # [32, 256, 128]
#         desc1 = pred['line_desc1']      # [32, 256, 128]
#         mat_score = pred['score_matrix_line']
        
#         overlap_gt = target['mat_assign_sublines'][:,:-1,:-1]
#         mat_match = torch.where(overlap_gt > 0.5, 1, 0)
#         mat_unmatch = torch.where(overlap_gt <= 0, 1, 0)

#         b, d, nd0 = desc0.shape
#         nd1 = desc1.shape[2]

#         distance = torch.zeros((b,nd0,nd1)).to(overlap_gt)
#         for b_idx in np.arange(b):
#             desc0_ = desc0[b_idx].T
#             desc1_ = desc1[b_idx].T

#             sum_sq0 = torch.sum(torch.square(desc0_), axis=1)[:,None]  # (1, 16, 128) -> (1, 128)
#             sum_sq1 = torch.sum(torch.square(desc1_), axis=1)[:,None].T

#             dmat = desc0_ @ desc1_.T    # [128, 128]
#             distance[b_idx] = (sum_sq0 + sum_sq1 - 2.0 * dmat).clip(min=0)

#         dists_pos = distance * mat_match    # [8, 128, 128]
#         dists_pos = torch.cat((dists_pos, dists_pos.transpose(1,2)), dim=1)  # [16, 256, 128]
#         dists_pos = torch.amax(dists_pos, dim=2).reshape(-1,1)   # [16, 256] -> [4096]
#         idx_pos = torch.where(dists_pos>0.0)
#         dists_pos_final = dists_pos[idx_pos]   # [1959]

#         hardest_pos = torch.amax(dists_pos_final)

#         ## negative distance를 구한다. 
#         max_dist = 10000.0
#         dists_neg = (distance * mat_unmatch) + (max_dist * (1-mat_unmatch))   # [16, 128, 128]
#         dists_neg = torch.cat((dists_neg, dists_neg.transpose(1,2)), 1)  # [16, 256, 128]
#         dists_neg = dists_neg.reshape(-1,128) # [4096, 128]
#         dists_neg_candi = dists_neg[idx_pos[0],:]

#         margin = 0.5
#         masks = torch.logical_and(dists_neg_candi < dists_pos_final.reshape(-1,1)+margin, dists_neg_candi > dists_pos_final.reshape(-1,1))
#         # masks = torch.logical_and(dists_neg_candi < margin, dists_neg_candi > 0)

#         dists_neg_list = []
#         valid_idx = []
#         for i, (dist_neg, mask) in enumerate(zip(dists_neg_candi, masks)):
#             neg = dist_neg[mask]
#             if len(neg) == 0:
#                 continue
#             else:
#                 valid_idx.append(i)

#             # idx = torch.randint(len(neg),(1,))
#             idx = torch.argmin(neg)
#             dists_neg_list.append(neg[idx])
#         dists_neg_final = torch.stack(dists_neg_list)
#         dists_pos_final = dists_pos_final[valid_idx]

#         return dists_pos_final, dists_neg_final

#     def extract_triplet(self, pred, target):
#         desc0 = pred['line_desc0'].transpose(1,2)      # [32, 256, 128] - >[32, 128, 256]
#         desc1 = pred['line_desc1'].transpose(1,2)      # [32, 256, 128] - >[32, 128, 256]
#         mat_score = pred['score_matrix_line']
#         match_gt = target['mat_assign_sublines'][:,:-1,:-1]
#         # mat_match = torch.where(overlap_gt > 0.0, 1, 0)
#         # mat_unmatch = torch.where(overlap_gt <= 0, 1, 0)

#         b, d, nd0 = desc0.shape
#         nd1 = desc1.shape[2]

#         anchor = torch.cat((desc0, desc1), dim = 1).reshape(-1,256)
#         matches = torch.cat((match_gt, match_gt.transpose(1,2)), dim=1).reshape(-1,nd1)
#         pos_index = torch.max(overlap_gt, dim=1)


#         return anchor, positive, negative

#     def forward(self, pred, target):
        
#         dists_pos, dists_neg = self.compute_distances(pred, target)

#         # anchor, positive, negative = self.extract_triplet(pred, target)
#         # t_loss = self.triplet_loss(anchor, positive, negative)

#         hardest_positive = torch.amax(dists_pos)
#         hardest_negative = torch.amin(dists_neg)
#         # t_loss1 = F.relu(hardest_positive - dists_neg + 0.5)
#         # t_loss2 = F.relu(dists_pos - hardest_negative + 0.5)
#         # t_loss = F.relu(- hardest_negative + 0.5)
#         # t_loss = torch.mean(torch.cat((t_loss2.squeeze(),t_loss2))) 
#         t_loss = F.relu(dists_pos - dists_neg + 1.)
#         t_loss = t_loss.mean()

#         return t_loss, dists_pos.mean(), dists_neg.mean()

# class descriptor_loss(nn.Module):
#     def __init__(self):
#         super(descriptor_loss, self).__init__()
#         self.margin = 0.5

#     def forward(self, pred, target):




# ### triplet loss
# class descriptor_loss(nn.Module):
#     def __init__(self):
#         super(descriptor_loss, self).__init__()
#         self.margin_neg = 0.3
#         self.lamda_d = 250
#         self.margin = 1

#     def forward(self, pred, target):
#         # assert pred.dim() == target.dim(), "inconsistent dimensions"

#         b,desc_dim, nd0 = pred['line_desc0'].shape

#         overlap_gt = target['mat_assign_sublines'][:,:-1,:-1]
#         mat_match = torch.where(overlap_gt > 0, 1, 0)
#         mat_unmatch = torch.where(overlap_gt <= 0, 1, 0)

#         desc0 = pred['line_desc0']      # [32, 256, 128]
#         desc1 = pred['line_desc1']      # [32, 256, 128]
#         scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
#         mat_dist = 2-2*scores   # # [32, 256, 256]


#         pos_dist = mat_dist * mat_match
#         neg_dist = mat_dist * mat_unmatch + (desc_dim * mat_match)
        
#         neg_dist0 = torch.amin(neg_dist, dim=1)
#         neg_dist1 = torch.amin(neg_dist, dim=2)
#         neg_dist_ = torch.amin(torch.cat([neg_dist0,neg_dist1], dim=1), dim=1)

#         zero = torch.tensor(0.).to(desc0)

#         loss_desc = torch.tensor(0.).to(desc0)
#         for i in range(b):
#             loss_desc += torch.max(zero, self.margin + pos_dist[i].sum() - neg_dist_[i])

#         # # hinge loss
#         # positive_dist = torch.max(overlap_gt - scores, torch.tensor(0.).to(scores))
#         # positive_dist = mat_match * positive_dist 
#         # # positive_dist[positive_dist < 0] = 0
#         # # negative_dist = torch.max(scores, torch.tensor(0.).to(scores))
#         # negative_dist = torch.max(scores - self.margin_neg, torch.tensor(0.).to(scores))
#         # negative_dist = mat_unmatch * negative_dist 

#         # loss_desc = (self.lamda_d * mat_match * positive_dist) + (mat_unmatch * negative_dist)
#         # # loss_desc = loss_desc * mask_valid
#         # # loss_desc = loss_desc.sum() / (mat_match+mat_unmatch).sum()
#         # loss_desc = loss_desc.sum()

#         return loss_desc
