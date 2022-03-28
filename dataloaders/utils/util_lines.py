import numpy as np
import torch
import math

def calc_distance_point_line(point, lines):
    x0 = point[0]
    y0 = point[1]
    x1 = lines[0][0]
    y1 = lines[0][1]
    x2 = lines[1][0]
    y2 = lines[1][1]

    return np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

def calc_distance_point_point(point0, point1):
    x0 = point0[0]
    y0 = point0[1]
    x1 = point1[0]
    y1 = point1[1]

    return np.sqrt((x1-x0)**2 + (y1-y0)**2)

# def find_line_matches(lines0, lines1, thres_reprojected, thres_angdiff):
#     nlines0 = len(lines0)
#     nlines1 = len(lines1)
#     mat_line_match = np.zeros((nlines0, nlines1))

#     # lines0 기준으로 overlap score 측정
#     for i0, line0 in enumerate(lines0):
#         for i1, line1 in enumerate(lines1):
            
#             # calculate a reprojection error 
#             dist0 = calc_distance_point_line(line1[0], line0)
#             dist1 = calc_distance_point_line(line1[1], line0)

#             # calculate angle difference
#             angle0 = np.degrees(np.arctan2((line0[1,0] - line0[0,0]), (line0[1,1] - line0[0,1])))
#             angle1 = np.degrees(np.arctan2((line1[1,0] - line1[0,0]), (line1[1,1] - line1[0,1])))

#             ang_diff = np.abs(angle1 - angle0)
#             ang_diff = ang_diff % 180

#             # out?
#             len0 = calc_distance_point_point(line0[0], line0[1])
#             len1 = calc_distance_point_point(line1[0], line1[1])
#             dist_s0s1 = calc_distance_point_point(line0[0], line1[0])
#             dist_e0s1 = calc_distance_point_point(line0[1], line1[0])
#             dist_s0e1 = calc_distance_point_point(line0[0], line1[1])
#             dist_e0e1 = calc_distance_point_point(line0[1], line1[1])

#             is_sp1_on_line0 = (dist_s0s1 < len0) and (dist_e0s1 < len0)
#             is_ep1_on_line0 = (dist_s0e1 < len0) and (dist_e0e1 < len0)

#             is_overlap = True
#             if (not is_sp1_on_line0) and (not is_ep1_on_line0):
#                 dists = [dist_s0s1, dist_e0s1, dist_s0e1, dist_e0e1]
#                 if np.max(dists) > len0 + len1:
#                     is_overlap = False

#             # line0(ref)과 line1이 일직선 상에 있다면
#             if dist0 < thres_reprojected and dist1 < thres_reprojected \
#                 and ang_diff < thres_angdiff and is_overlap:
#                 mat_line_match[i0, i1] = 1

#     return mat_line_match

def find_line_matches(lines0, lines1, thres_reprojected, thres_angdiff):
    nlines0 = len(lines0)
    nlines1 = len(lines1)
    mat_line_match = np.zeros((nlines0, nlines1))

    # lines0 기준으로 overlap score 측정
    for i0, line0 in enumerate(lines0):
        for i1, line1 in enumerate(lines1):
            
            # calculate a reprojection error 
            dist0 = calc_distance_point_line(line1[0], line0)
            dist1 = calc_distance_point_line(line1[1], line0)
            
            if dist0 > thres_reprojected and dist1 > thres_reprojected:
                continue

            # calculate angle difference
            angle0 = np.degrees(np.arctan2((line0[1,0] - line0[0,0]), (line0[1,1] - line0[0,1])))
            angle1 = np.degrees(np.arctan2((line1[1,0] - line1[0,0]), (line1[1,1] - line1[0,1])))

            ang_diff = np.abs(angle1 - angle0)
            ang_diff = ang_diff % 180
            
            if ang_diff > thres_angdiff:
                continue
            
            # out?
            len0 = calc_distance_point_point(line0[0], line0[1])
            len1 = calc_distance_point_point(line1[0], line1[1])
            dist_s0s1 = calc_distance_point_point(line0[0], line1[0])
            dist_e0s1 = calc_distance_point_point(line0[1], line1[0])
            dist_s0e1 = calc_distance_point_point(line0[0], line1[1])
            dist_e0e1 = calc_distance_point_point(line0[1], line1[1])

            is_sp1_on_line0 = (dist_s0s1 < len0) and (dist_e0s1 < len0)
            is_ep1_on_line0 = (dist_s0e1 < len0) and (dist_e0e1 < len0)

            is_overlap = True
            if (not is_sp1_on_line0) and (not is_ep1_on_line0):
                dists = [dist_s0s1, dist_e0s1, dist_s0e1, dist_e0e1]
                if np.max(dists) > len0 + len1:
                    is_overlap = False

            # line0(ref)과 line1이 일직선 상에 있다면
            if is_overlap:
                mat_line_match[i0, i1] = 1

    return mat_line_match

def calculate_line_overlaps(lines0, lines1, matches_sublines):
    def calc_overlap(line0, line1):     # ref_line: line0
        len0 = calc_distance_point_point(line0[0], line0[1])
        len1 = calc_distance_point_point(line1[0], line1[1])

        # line1의 point들이 line0안에 있나 없나 검사-
        # line1의 sp기준으로 sp가 line0 안에 포함되어 있는지 확인 (sp1과 sp0, ep0의 길이가 line0의 길이보다 길면 sp1은 밖에 존재)
        dist_s0s1 = calc_distance_point_point(line0[0], line1[0])
        dist_e0s1 = calc_distance_point_point(line0[1], line1[0])
        # is_sp1_out = (dist_s0s1 > len0) or (dist_e0s1 > len0)
        is_sp1_on_line0 = (dist_s0s1 < len0) and (dist_e0s1 < len0)

        # line1의 ep기준으로 sp가 line0 안에 포함되어 있는지 확인 (sp1과 sp0, ep0의 길이가 line0의 길이보다 길면 sp1은 밖에 존재)
        dist_s0e1 = calc_distance_point_point(line0[0], line1[1])
        dist_e0e1 = calc_distance_point_point(line0[1], line1[1])
        # is_ep1_out = (dist_s0e1 > len0) or (dist_e0e1 > len0)
        is_ep1_on_line0 = (dist_s0e1 < len0) and (dist_e0e1 < len0)
        
        if (is_sp1_on_line0) and (is_ep1_on_line0): # line1의 두 점이 모두 line0 안에 있는 경우
            ratio_line0 = len1/len0
        elif (is_sp1_on_line0) and (not is_ep1_on_line0):   # sp1이 line0 안에 있는 경우 -> ep1이 밖에 있는 경우
            if dist_s0e1 > dist_e0e1:   # 
                ratio_line0 = dist_e0s1 / len0
            else:
                ratio_line0 = dist_s0s1 / len0
        elif (is_ep1_on_line0) and (not is_sp1_on_line0):   # ep1이 line0 안에 있는 경우 -> sp1이 밖에 있는 경우
            if dist_s0s1 > dist_e0s1:
                ratio_line0 = dist_e0e1 / len0
            else:
                ratio_line0 = dist_s0e1 / len0
        elif (not is_sp1_on_line0) and (not is_ep1_on_line0):
            dists = [dist_s0s1, dist_e0s1, dist_s0e1, dist_e0e1]
            if np.max(dists) <= len0 + len1:
                ratio_line0 = 1.
            elif np.max(dists) > len0 + len1:
                ratio_line0 = 0.                # 안겹침.. 한쪽 PROJECTION만 유효할 때, 이쪽으로 들어올 수 있음.
                # print("somethings wrong!!")

        return ratio_line0

    nlines0 = len(lines0)
    nlines1 = len(lines1)
    n_matches = len(matches_sublines)

    mat_overlap = np.zeros((nlines0, nlines1))
    overlaps = np.zeros(n_matches)

    for i, match_pair in enumerate(matches_sublines):
        idx0 = match_pair[0]
        idx1 = match_pair[1]

        ratio0 = calc_overlap(lines0[idx0], lines1[idx1]) # line0기준으로 overlap score
        mat_overlap[idx0, idx1] = ratio0
        overlaps[i] = ratio0
    
    return mat_overlap, overlaps

def make_pseudo_lines(n_left):
    mask_length = 15
    min_length = 16
    max_rand_length = 160
    sp_area = [640-min_length-mask_length,480-min_length-mask_length]
    img_size = np.array([640,480])
    
    # np.random.seed(np.random.randint(1004))
    pseudo_lines = np.zeros((n_left,2,2))
    pseudo_lines[:,0] = np.array(sp_area).T * np.random.rand(n_left, 2)

    pseudo_angle = 2*np.pi * np.random.rand(n_left)
    dist_range = (min_length, max_rand_length)
    pseudo_len = (dist_range[1]-dist_range[0]) * np.random.rand(n_left) + dist_range[0]
    pseudo_lines[:,1,0] = pseudo_lines[:,0,0] + pseudo_len*np.cos(pseudo_angle)
    pseudo_lines[:,1,1] = pseudo_lines[:,0,1] + pseudo_len*np.sin(pseudo_angle)
    pseudo_lines[:,:,0] = np.clip(pseudo_lines[:,:,0], mask_length, img_size[0]-mask_length)
    pseudo_lines[:,:,1] = np.clip(pseudo_lines[:,:,1], mask_length, img_size[1]-mask_length)

    # validation check
    for i, line in enumerate(pseudo_lines):
        sp = line[0]
        ep = line[1]
        length = np.sqrt(np.sum((ep-sp)**2))

        # if np.isnan(sp[0]) or np.isnan(ep[0]) or np.isnan(sp[1]) or np.isnan(ep[1]):
        #     print('here')

        if length < min_length:
            end = False
            new_line = np.zeros((2,2))
            count=0
            while end == False:
                count += 1
                new_line[0] = np.array(sp_area).T * np.random.rand(1, 2)
                pseudo_angle = 2*np.pi * np.random.rand(1)
                dist_range = (min_length, max_rand_length)
                pseudo_len = (dist_range[1]-dist_range[0]) * np.random.rand(1) + dist_range[0]
                new_line[1,0] = new_line[0,0] + pseudo_len*np.cos(pseudo_angle)
                new_line[1,1] = new_line[0,1] + pseudo_len*np.sin(pseudo_angle)
                new_line[:,0] = np.clip(new_line[:,0], mask_length, img_size[0]-mask_length)
                new_line[:,1] = np.clip(new_line[:,1], mask_length, img_size[1]-mask_length)

                sp = new_line[0]
                ep = new_line[1]

                # if np.isnan(sp[0]) or np.isnan(ep[0]) or np.isnan(sp[1]) or np.isnan(ep[1]):
                #     print('here')

                length = np.sqrt(np.sum((ep-sp)**2))
                if length > min_length:
                    end = True
                    pseudo_lines[i] = new_line
                    # print('exit', count, pseudo_lines[i], new_line)
                    break
            # print('almost')
    # print('done')
    return pseudo_lines

def conv_fixed_size(dic_line1, dic_line2, ids1, ids2, mat_assign_sublines, sublines1_3d, sublines2_3d, total_num_subline):
    # 매칭을 최대가 되도록 남기고, 나머지는 subline으로 나뉘는 긴 라인 순으로 채워 넣음.
    # TODO 1: Match된 라인들이 최대한 남기고
    # TODO 2: Match 안된 라인들은 (1) 원래 라인들로 채우고, (2) 수가 부족하면 랜덤으로 생성.
    # input: dic_line1, dic_line2, ids1, ids2, mat_assign_sublines, self.total_num_subline
    # output: sublines1, sublines2, mat_assign_sublines, klines1, klines2, mat_klines2sublines1, mat_klines2sublines2, 
    # klines1 = dic_line1['keylines']             # (346, 2, 2)
    # klines2 = dic_line2['keylines']             # (433, 2, 2)
    # mat_klines2sublines1 = dic_line1['mat_klines2sublines']            # (346, 364)
    # mat_klines2sublines2 = dic_line2['mat_klines2sublines']            # (433, 451)
    # sublines1 = dic_line1['sublines']           # (364, 2, 2) -> (184, 2, 2)
    # sublines2 = dic_line2['sublines']           # (451, 2, 2) -> (166, 2, 2)
    # mat_assign_sublines                         # (184, 166)  -> should be (128, 128)
    # ids1, ids2                                  # dic_line1['sublines'] 관점에서의 id
    
    # n_klines1 = len(klines1)
    # n_klines2 = len(klines2)
    # n_sublines1 = len(sublines1)
    # n_sublines2 = len(sublines2)

    matched_pairs = np.array(np.where(mat_assign_sublines > 0)).T

    num_sublines1 = len(dic_line1['sublines'])
    num_sublines2 = len(dic_line2['sublines'])
    sublines1_3d_ = np.ones((num_sublines1, 2, 3))*-1.
    sublines2_3d_ = np.ones((num_sublines2, 2, 3))*-1.
    mat_sublines1_T_sublines2 = np.zeros((num_sublines1, num_sublines2))
    for pair in matched_pairs:
        mat_sublines1_T_sublines2[ids1[pair[0]],ids2[pair[1]]] = mat_assign_sublines[pair[0], pair[1]]

    sublines1_3d_[ids1] = sublines1_3d
    sublines2_3d_[ids2] = sublines2_3d

    ## 
    matched_pairs = np.array(np.where(mat_sublines1_T_sublines2 > 0)).T
    index_matched1 = np.array(sorted(list(set(matched_pairs[:,0]))))
    index_matched2 = np.array(sorted(list(set(matched_pairs[:,1]))))

    # 매칭이 많으면, 그 중 128개만 고름 
    if len(index_matched1) > total_num_subline:
        index_sublines1 = np.sort(index_matched1[:total_num_subline])
        n_sublines1 = total_num_subline
        n_pseudoline1 = 0

    else:   # 매칭이 128보다 적으면, unmatch된 것 중에서 앞쪽부터 나머지를 고름 #  
        n_remain = total_num_subline - len(index_matched1)
        index_unmatched1 = np.delete(np.arange(num_sublines1), index_matched1)[:n_remain]
        index_sublines1 = np.sort(np.concatenate((index_matched1, index_unmatched1)))
        n_sublines1 = len(index_sublines1)
        n_pseudoline1 = total_num_subline - len(index_sublines1)

    if len(index_matched2) > total_num_subline:
        index_sublines2 = np.sort(index_matched2[:total_num_subline])
        n_sublines2 = total_num_subline
        n_pseudoline2 = 0

    else:   
        n_remain = total_num_subline - len(index_matched2)
        index_unmatched2 = np.delete(np.arange(num_sublines2), index_matched2)[:n_remain]
        index_sublines2 = np.sort(np.concatenate((index_matched2, index_unmatched2)))
        n_sublines2 = len(index_sublines2)
        n_pseudoline2 = total_num_subline - len(index_sublines2)

    ## within new index
    mat_klines2sublines1 = dic_line1['mat_klines2sublines'][:,index_sublines1]
    index_klines1 = ~np.all(mat_klines2sublines1 == 0, axis=1)
    mat_klines2sublines1 = mat_klines2sublines1[index_klines1]
    n_klines1 = mat_klines2sublines1.shape[0]
    mat_klines2sublines1 = np.hstack((mat_klines2sublines1, np.zeros((n_klines1, n_pseudoline1))))
    mat_klines2sublines1 = np.vstack((mat_klines2sublines1, np.zeros((total_num_subline-n_klines1, total_num_subline))))
       
    mat_klines2sublines2 = dic_line2['mat_klines2sublines'][:,index_sublines2]
    index_klines2 = ~np.all(mat_klines2sublines2 == 0, axis=1)
    mat_klines2sublines2 = mat_klines2sublines2[index_klines2]
    n_klines2 = mat_klines2sublines2.shape[0]
    mat_klines2sublines2 = np.hstack((mat_klines2sublines2, np.zeros((n_klines2, n_pseudoline2))))
    mat_klines2sublines2 = np.vstack((mat_klines2sublines2, np.zeros((total_num_subline-n_klines2, total_num_subline))))

    # n_unmat = 0
    mat_sublines1_T_sublines2_ = np.zeros((total_num_subline, total_num_subline))
    for pair in matched_pairs:
        if (pair[0] == index_sublines1).any() == False:
            # n_unmat+=1
            continue
        if (pair[1] == index_sublines2).any() == False:
            # n_unmat+=1
            continue
        pair0 = np.where(pair[0] == index_sublines1)[0][0]
        pair1 = np.where(pair[1] == index_sublines2)[0][0]
        mat_sublines1_T_sublines2_[pair0, pair1] = mat_sublines1_T_sublines2[pair[0], pair[1]]

    # print('n_unmat:', n_unmat)

    klines1 = dic_line1['keylines'][index_klines1]
    klines2 = dic_line2['keylines'][index_klines2]
    # num_klines1, num_klines2 = len(klines1), len(klines2)
    sublines1 = dic_line1['sublines'][index_sublines1]
    sublines2 = dic_line2['sublines'][index_sublines2]
    sublines1_3d_ = sublines1_3d_[index_sublines1]
    sublines2_3d_ = sublines2_3d_[index_sublines2]

    sublines1_3d__ = np.ones((total_num_subline, 2, 3))*-1.
    sublines2_3d__ = np.ones((total_num_subline, 2, 3))*-1.
    sublines1_3d__[:n_sublines1] = sublines1_3d_
    sublines2_3d__[:n_sublines2] = sublines2_3d_

    klines1_ = np.zeros((total_num_subline,2,2))
    klines1_[:n_klines1] = klines1
    klines2_ = np.zeros((total_num_subline,2,2))
    klines2_[:n_klines2] = klines2

    if n_pseudoline1 > 0:
        sublines1 = np.vstack((sublines1, make_pseudo_lines(n_pseudoline1)))
    if n_pseudoline2 > 0:
        sublines2 = np.vstack((sublines2, make_pseudo_lines(n_pseudoline2)))

    ## fixed size
    matches_sublines = np.array(np.where(mat_sublines1_T_sublines2_>0)).T
    mat_sublines1_T_sublines2_ret = np.zeros((total_num_subline+1,total_num_subline+1))
    mat_sublines1_T_sublines2_ret[:total_num_subline, :total_num_subline] = mat_sublines1_T_sublines2_
    unmatches_line0 = np.array(range(total_num_subline+1))
    unmatches_line0 = np.delete(unmatches_line0, matches_sublines[:,0])[None]
    unmatches_line1 = np.array(range(total_num_subline+1))
    unmatches_line1 = np.delete(unmatches_line1, matches_sublines[:,1])[None]
    mat_sublines1_T_sublines2_ret[unmatches_line0,-1] = 1
    mat_sublines1_T_sublines2_ret[-1,unmatches_line1] = 1
    mat_sublines1_T_sublines2_ret[-1,-1] = 1
    # mat_assign_sublines_ret[matches_sublines[:,0], -1] = 1 - mat_sublines1_T_sublines2_[matches_sublines[:,0],matches_sublines[:,1]]
    # mat_assign_sublines_ret[-1,matches_sublines[:,1]] = 1 - mat_sublines1_T_sublines2_[matches_sublines[:,0],matches_sublines[:,1]]

    mat_klines1_T_klines2 = mat_klines2sublines1@mat_sublines1_T_sublines2_@mat_klines2sublines2.T
    num_klines1, num_klines2 = mat_klines1_T_klines2.shape

    mat_klines1_T_klines2_ret = np.zeros((total_num_subline+1, total_num_subline+1))
    mat_klines1_T_klines2_ret[:num_klines1, :num_klines2] = mat_klines1_T_klines2
    matches_klines = np.where(mat_klines1_T_klines2 > 0)
    unmatches_line0 = np.array(range(total_num_subline+1))
    unmatches_line0 = np.delete(unmatches_line0, matches_klines[0])[None]
    unmatches_line1 = np.array(range(total_num_subline+1))
    unmatches_line1 = np.delete(unmatches_line1, matches_klines[1])[None]
    mat_klines1_T_klines2_ret[unmatches_line0,-1] = 1
    mat_klines1_T_klines2_ret[-1,unmatches_line1] = 1
    mat_klines1_T_klines2_ret[-1,-1] = 1


    # print('here')
    # # TODO 2: Match 안된 라인들은 (1) 원래 라인들로 채우고, (2) 수가 부족하면 랜덤으로 생성.
    # if n_matches < total_num_subline:
    #     # n_sublines1 = sublines1.shape[0]
    #     sublines1_fixed = np.zeros((total_num_subline,2,2))
    #     sublines1_fixed[:n_sublines1] = sublines1[ids1]
    #     mat_klines2sublines1_fixed = np.zeros((klines1.shape[0], total_num_subline))
    #     mat_klines2sublines1_fixed[:,:n_sublines1] = mat_klines2sublines1

    #     n_sublines2 = sublines2.shape[0]
    #     sublines2_fixed = np.zeros((total_num_subline,2,2))
    #     sublines2_fixed[:n_sublines2] = sublines2
    #     mat_klines2sublines2_fixed = np.zeros((klines2.shape[0], total_num_subline))
    #     mat_klines2sublines2_fixed[:,:n_sublines2] = mat_klines2sublines2

    #     n_left = total_num_subline - n_matches
    #     ## 원래 라인들로 채우고
    #     ids1_unmatch = np.arange(dic_line1['sublines'].shape[0])
    #     ids1_unmatch = np.delete(ids1_unmatch, np.array(ids1))
    #     n_unmatch1 = len(ids1_unmatch)
    #     if n_unmatch1 >= n_left:
    #         sublines1_fixed[n_sublines1:] = dic_line1['sublines'][ids1_unmatch[:n_left]]
    #     ## 수가 부족하면 랜덤으로 생성
    #     else:
    #         sublines1_fixed[n_sublines1:n_sublines1+n_unmatch1] = dic_line1['sublines'][ids1_unmatch]
    #         sublines1_fixed[n_sublines1+n_unmatch1:] = make_pseudo_lines(n_left - n_unmatch1)

    #     ids2_unmatch = np.arange(dic_line2['sublines'].shape[0])
    #     ids2_unmatch = np.delete(ids2_unmatch, np.array(ids2))
    #     n_unmatch2 = len(ids2_unmatch)
    #     if n_unmatch2 >= n_left:
    #         sublines2_fixed[n_sublines2:] = dic_line2['sublines'][ids2_unmatch[:n_left]]
    #     ## 수가 부족하면 랜덤으로 생성
    #     else:
    #         sublines1_fixed[n_sublines2:n_sublines2+n_unmatch2] = dic_line2['sublines'][ids2_unmatch]
    #         sublines1_fixed[n_sublines2+n_unmatch2:] = make_pseudo_lines(n_left - n_unmatch2)

    #     # ## 수가 부족하면 랜덤으로 생성
    #     # sublines1_fixed[n_sublines1:] = make_pseudo_lines(n_left)
    #     # sublines2_fixed[n_sublines2:] = make_pseudo_lines(n_left)

    #     overlap_scores = mat_assign_sublines[matched_pairs]
    #     mat_assign_sublines_fixed = np.zeros((total_num_subline,total_num_subline))
    #     mat_assign_sublines_fixed[:n_matches,:n_matches] = np.diag(overlap_scores)
    
    # else:
    #     sublines1_fixed = sublines1
    #     sublines2_fixed = sublines2
    #     mat_klines2sublines1_fixed = mat_klines2sublines1
    #     mat_klines2sublines2_fixed = mat_klines2sublines2
    #     n_sublines1 = sublines1.shape[0]
    #     n_sublines2 = sublines2.shape[0]

    #     overlap_scores = mat_assign_sublines[matched_pairs][:total_num_subline]
    #     mat_assign_sublines_fixed = np.diag(overlap_scores)


    # b_matches = (mat_assign_sublines > 0)
    # matches_sublines = np.array(np.where(b_matches)).T

    # mat_assign_sublines_ = np.zeros((self.total_num_subline+1,self.total_num_subline+1))
    # mat_assign_sublines_[:self.total_num_subline, :self.total_num_subline] = mat_assign_sublines
    # unmatches_line0 = np.array(range(self.total_num_subline+1))
    # unmatches_line0 = np.delete(unmatches_line0, matches_sublines[:,0])[None]
    # unmatches_line1 = np.array(range(self.total_num_subline+1))
    # unmatches_line1 = np.delete(unmatches_line1, matches_sublines[:,1])[None]
    # mat_assign_sublines_[unmatches_line0,-1] = 1
    # mat_assign_sublines_[-1,unmatches_line1] = 1
    # mat_assign_sublines_[matches_sublines[:,0], -1] = 1 - mat_assign_sublines[matches_sublines[:,0],matches_sublines[:,1]]
    # mat_assign_sublines_[-1,matches_sublines[:,1]] = 1 - mat_assign_sublines[matches_sublines[:,0],matches_sublines[:,1]]


    # num_klines1 = mat_klines2sublines1.shape[0]
    # num_klines2 = mat_klines2sublines2.shape[0]
    # mat_assign_klines = mat_klines2sublines1 @ mat_assign_sublines @ mat_klines2sublines2.T
    # mat_assign_klines_ = np.zeros((self.total_num_subline+1, self.total_num_subline+1))
    # mat_assign_klines_[:num_klines1, :num_klines2] = mat_assign_klines
    # matches_klines = np.where(mat_assign_klines > 0)
    # unmatches_line0 = np.array(range(self.total_num_subline+1))
    # unmatches_line0 = np.delete(unmatches_line0, matches_klines[0])[None]
    # unmatches_line1 = np.array(range(self.total_num_subline+1))
    # unmatches_line1 = np.delete(unmatches_line1, matches_klines[1])[None]
    # mat_assign_klines_[unmatches_line0,-1] = 1
    # mat_assign_klines_[-1,unmatches_line1] = 1

    # mat_klines2sublines1_ = np.zeros((self.total_num_subline, self.total_num_subline))
    # mat_klines2sublines2_ = np.zeros((self.total_num_subline, self.total_num_subline))
    # mat_klines2sublines1_[:num_klines1,:] = mat_klines2sublines1
    # mat_klines2sublines2_[:num_klines2,:] = mat_klines2sublines2

    return klines1_, klines2_, sublines1, sublines2, sublines1_3d__, sublines2_3d__, mat_klines2sublines1, mat_klines2sublines2, mat_sublines1_T_sublines2_ret, n_sublines1, n_sublines2, mat_klines1_T_klines2_ret, n_klines1, n_klines2

def calc_line_traits(ksublines, max_num_pixels_in_sentence):
    # sublines의 개수가 128 (total_num_subline) 보다 작을 때... 
    # ksublines = ksublines
    ksublines_sp = ksublines[:,0]
    ksublines_ep = ksublines[:,1]
    len_sublines = np.sqrt(np.sum((ksublines_ep - ksublines_sp)**2, axis=1))
    resp_sublines = len_sublines / max_num_pixels_in_sentence
    angle_sublines = np.arctan2((ksublines_ep[:,0]-ksublines_sp[:,0]), (ksublines_ep[:,1]-ksublines_sp[:,1]))
    for i, angle in enumerate(angle_sublines):
        if angle < 0:
            angle_sublines[i] += np.pi
    angle_sublines = np.array([np.cos(2*angle_sublines), np.sin(2*angle_sublines)]).T

    return len_sublines, resp_sublines, angle_sublines

def remove_duplicated_matches(matches, assign_mat, img_pairs):
    l0 = list(matches[0])                               # matches: keypoints0 & 1중에서 서로 매치되는 인덱스의 모음
    dupset0 = set([x for x in l0 if l0.count(x) > 1])   # 다수 keypoints0 경우

    l1 = list(matches[1])                               # matches: keypoints0 & 1중에서 서로 매치되는 인덱스의 모음
    dupset1 = set([x for x in l1 if l1.count(x) > 1])   # 다수 keypoints1 경우

    # case0: keypoints0 w/ 다수 keypoints1
    match_idx_to_remove = np.array([], dtype='int')
    for match_idx0 in dupset0:     # dupset0: {616, 694, 569, 603, 412}
        
        # match pairs에서 몇번째 위치?
        loc_in_matches = np.where(matches[0] == match_idx0)[0]     
        match_idx10 = matches[1][loc_in_matches]

        # scores 비교 -> 가장 높지 않은 score에 해당하는 것들은 지울 수 있게 저장
        max_score = torch.max(img_pairs['scores1'][match_idx10])
        idx_tmp = (max_score != img_pairs['scores1'][match_idx10])
        if not any(idx_tmp):   
            idx_to_remove = loc_in_matches[0]   # score가 모두 같다면 앞에 index를 선정
        else:
            idx_to_remove = loc_in_matches[idx_tmp]
        match_idx_to_remove = np.append(match_idx_to_remove, idx_to_remove)

    # case1: 다수 keypoints0 w/ keypoints1
    for match_idx1 in dupset1:     # dupset0: {616, 694, 569, 603, 412}
        
        # match pairs에서 몇번째 위치?
        loc_in_matches = np.where(matches[1] == match_idx1)[0]     
        match_idx01 = matches[0][loc_in_matches]

        # scores 비교 -> 가장 높지 않은 score에 해당하는 것들은 지울 수 있게 저장
        max_score = torch.max(img_pairs['scores0'][match_idx01])
        idx_tmp = (max_score != img_pairs['scores0'][match_idx01])
        if not any(idx_tmp):
            idx_to_remove = loc_in_matches[0]   # score가 모두 같다면 앞에 index를 선정
        else:
            idx_to_remove = loc_in_matches[idx_tmp]
        match_idx_to_remove = np.append(match_idx_to_remove, idx_to_remove)

    # print(match_idx_to_remove)
    matches2 = np.delete(matches, match_idx_to_remove, axis=1)

    for iremove in match_idx_to_remove:
        x = matches[0][int(iremove)]
        y = matches[1][int(iremove)]
        assign_mat[x,y] = 0

    # # sanity check
    # ll0 = list(matches2[0])                               # matches: keypoints0 & 1중에서 서로 매치되는 인덱스의 모음
    # dupset0_ = set([x for x in ll0 if ll0.count(x) > 1])   # 다수 keypoints0 경우

    # ll1 = list(matches2[1])                               # matches: keypoints0 & 1중에서 서로 매치되는 인덱스의 모음
    # dupset1_ = set([x for x in ll1 if ll1.count(x) > 1])   # 다수 keypoints1 경우
    # if len(dupset0_) != 0 or len(dupset0_) != 0:
    #     print("something wroing")
    #     print("something wroing")
    #     print("something wroing")

    return matches2, assign_mat

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

def make_pseudo_lines(n_left, image_shape):
    mask_length = 15
    min_length = 16
    max_rand_length = 160
    sp_area = [image_shape[0]-min_length-mask_length,image_shape[1]-min_length-mask_length]
    img_size = np.array(image_shape)
    
    np.random.seed(np.random.randint(1004))
    pseudo_lines = np.zeros((n_left,2,2))
    pseudo_lines[:,0] = np.array(sp_area).T * np.random.rand(n_left, 2)

    np.random.seed(np.random.randint(1005))
    pseudo_angle = 2*np.pi * np.random.rand(n_left)
    dist_range = (min_length, max_rand_length)
    pseudo_len = (dist_range[1]-dist_range[0]) * np.random.rand(n_left) + dist_range[0]
    pseudo_lines[:,1,0] = pseudo_lines[:,0,0] + pseudo_len*np.cos(pseudo_angle)
    pseudo_lines[:,1,1] = pseudo_lines[:,0,1] + pseudo_len*np.sin(pseudo_angle)
    pseudo_lines[:,:,0] = np.clip(pseudo_lines[:,:,0], mask_length, img_size[0]-mask_length)
    pseudo_lines[:,:,1] = np.clip(pseudo_lines[:,:,1], mask_length, img_size[1]-mask_length)

    # validation check
    for i, line in enumerate(pseudo_lines):
        sp = line[0]
        ep = line[1]
        length = np.sqrt(np.sum((ep-sp)**2))

        # if np.isnan(sp[0]) or np.isnan(ep[0]) or np.isnan(sp[1]) or np.isnan(ep[1]):
        #     print('here')

        if length < min_length:
            end = False
            new_line = np.zeros((2,2))
            count=0
            while end == False:
                count += 1
                new_line[0] = np.array(sp_area).T * np.random.rand(1, 2)
                pseudo_angle = 2*np.pi * np.random.rand(1)
                dist_range = (min_length, max_rand_length)
                pseudo_len = (dist_range[1]-dist_range[0]) * np.random.rand(1) + dist_range[0]
                new_line[1,0] = new_line[0,0] + pseudo_len*np.cos(pseudo_angle)
                new_line[1,1] = new_line[0,1] + pseudo_len*np.sin(pseudo_angle)
                new_line[:,0] = np.clip(new_line[:,0], mask_length, img_size[0]-mask_length)
                new_line[:,1] = np.clip(new_line[:,1], mask_length, img_size[1]-mask_length)

                sp = new_line[0]
                ep = new_line[1]

                # if np.isnan(sp[0]) or np.isnan(ep[0]) or np.isnan(sp[1]) or np.isnan(ep[1]):
                #     print('here')

                length = np.sqrt(np.sum((ep-sp)**2))
                if length > min_length:
                    end = True
                    pseudo_lines[i] = new_line
                    # print('exit', count, pseudo_lines[i], new_line)
                    break
            # print('almost')
    # print('done')
    return pseudo_lines

def get_angles(lines):
    line_exists = (len(lines) > 0)
    if not line_exists:
        angles = []
        return angles

    sp = lines[:,0]
    ep = lines[:,1]
    angles = np.arctan2((ep[:,0]-sp[:,0]), (ep[:,1]-sp[:,1]))
    for i, angle in enumerate(angles):
        if angle < 0:
            angles[i] += np.pi
    angles = np.asarray([np.cos(2*angles), np.sin(2*angles)]).T
    return angles

def find_line_attributes(klines, min_length, max_sublines):
    klines_sp, klines_ep, length, angle = [], [], [], []

    for line in klines:
        sp_x, sp_y = line[0]
        ep_x, ep_y = line[1]
        kline_sp = []
        if sp_x < ep_x:
            kline_sp = [sp_x, sp_y]
            kline_ep = [ep_x, ep_y]
        else:
            kline_sp = [ep_x, ep_y]
            kline_ep = [sp_x, sp_y]
        
        linelength = math.sqrt((kline_ep[0]-kline_sp[0])**2 +(kline_ep[1]-kline_sp[1])**2)
        if linelength < min_length:
            continue
        
        klines_sp.append(kline_sp)
        klines_ep.append(kline_ep)
        length.append(linelength)
        
    klines_sp = np.asarray(klines_sp)
    klines_ep = np.asarray(klines_ep)
    klines = np.stack((klines_sp, klines_ep), axis=1)
    length = np.asarray(length)

    # re-ordering by line length
    index = np.argsort(length)
    index = index[::-1]
    klines = klines[index[:max_sublines]]
    length = length[index[:max_sublines]]

    angles = get_angles(klines)
    return {'klines':klines, 'length_klines':length, 'angles': angles}

def conv_fixed_size(object, conf, func_token=None, pred_sp=None):
    
    if 'keypoints' in object.keys():
        ## keypoints
        resize = conf['data']['resize']
        max_kpts = conf['feature']['superpoint']['max_keypoints']
        kpts = object['keypoints'][0]
        desc = object['descriptors'][0]
        scores = object['scores'][0]
        dense_desc = object['dense_descriptor']
        device = kpts.device
        num_kpts = len(kpts)
        
        pseudo_pts = torch.tensor(resize).T * torch.rand(max_kpts-num_kpts, 2) # make a uniformly random tensor 
        pseudo_pts = pseudo_pts.to(device)        
        pseudo_desc = sample_descriptors(pseudo_pts[None], dense_desc, 8)[0].reshape(256,len(pseudo_pts))
        pseudo_scores = torch.zeros(len(pseudo_pts))
        
        object['keypoints'][0] = torch.cat((kpts, pseudo_pts.to(device)), 0)
        object['descriptors'][0] = torch.cat((desc, pseudo_desc.to(device)), 1)
        object['scores'][0] = torch.cat((scores, pseudo_scores.to(device)), 0)
        object['num_kpts'] = torch.tensor([num_kpts])

        return object

    elif 'klines' in object.keys():
        klines = object
        klines_np = {k: v[0].cpu().numpy() for k, v in klines.items()}
        line_tokenizer = func_token
        token_distance = conf['feature']['linetr']['token_distance']
        max_tokens = conf['feature']['linetr']['max_tokens']
        image_shape = conf['data']['resize']
        num_klns = len(klines_np['klines'])
        num_slns = len(klines_np['sublines'])
        max_slns = conf['feature']['linetr']['max_sublines']
        min_length = conf['feature']['linetr']['min_length']
        
        num_add_lns = max_slns-num_slns
        if num_add_lns > 0:
            pseudo_lines = make_pseudo_lines(num_add_lns, image_shape)
            
            # klines needs dict_keys(['klines', 'length_klines', 'angles'])
            pseudo_lines = find_line_attributes(pseudo_lines, min_length, max_slns)
            pseudo_lines = line_tokenizer(pseudo_lines, token_distance, max_tokens, pred_sp, image_shape)
            
            new_klns = torch.zeros((1,max_slns,2,2))
            new_klns[0,:num_klns] = klines['klines']
            new_klns[0,num_klns:num_klns+num_add_lns] = pseudo_lines['klines']
            klines['klines'] = new_klns
            
            new_data = torch.zeros((1,max_slns))
            new_data[0,:num_klns] = klines['length_klines']
            new_data[0,num_klns:num_klns+num_add_lns] = pseudo_lines['length_klines']
            klines['length_klines'] = new_data
            
            new_data = torch.zeros((1,max_slns,2))
            new_data[0,:num_klns] = klines['angles']
            new_data[0,num_klns:num_klns+num_add_lns] = pseudo_lines['angles']
            klines['angles'] = new_data
            
            key_to_store = ['sublines', 'pnt_sublines', 'mask_sublines', 'resp_sublines', 'angle_sublines', 'desc_sublines', 'score_sublines',]
            klines = {**klines, **{k:torch.cat((klines[k], pseudo_lines[k]), dim=1) for k,v in klines.items() if k in key_to_store}}
            
            new_k2s = torch.eye(max_slns)
            new_k2s[:num_klns, :num_slns] = klines['mat_klines2sublines'][0]
            klines['mat_klines2sublines'] = new_k2s[None]
            object = klines
        else:
            new_klns = torch.zeros((1,max_slns,2,2))
            new_klns[0,:num_klns] = klines['klines']
            klines['klines'] = new_klns
            
            new_data = torch.zeros((1,max_slns))
            new_data[0,:num_klns] = klines['length_klines']
            klines['length_klines'] = new_data
            
            new_data = torch.zeros((1,max_slns,2))
            new_data[0,:num_klns] = klines['angles']
            klines['angles'] = new_data

            key_to_store = ['sublines', 'pnt_sublines', 'mask_sublines', 'resp_sublines', 'angle_sublines', 'desc_sublines', 'score_sublines',]
            klines = {**klines, **{k:v[:,:max_slns] for k,v in klines.items() if k in key_to_store}}
            
            new_k2s = torch.eye(max_slns)
            if max_slns < num_klns:
                new_k2s[:max_slns,:max_slns] = klines['mat_klines2sublines'][0,:max_slns,:max_slns]
                klines['mat_klines2sublines'] = new_k2s[None]
            else:
                new_k2s[:num_klns,:max_slns] = klines['mat_klines2sublines'][0,:num_klns,:max_slns]
                klines['mat_klines2sublines'] = new_k2s[None]

            object = klines
        
        object['num_klns'] = torch.tensor([num_klns])
        object['num_slns'] = torch.tensor([num_slns])
    
        return object
