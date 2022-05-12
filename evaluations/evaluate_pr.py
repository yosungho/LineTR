import numpy as np

class Evaluate_PR():
    def __init__(self, args):
        self.args = args
        self.thresholds = {}
        self.best_thred = {}
        self.eps=0.00001
        
    def calc_TFPN(self, score, score_gt):
        score_gt = np.where(score_gt >0, 1, 0)

        n_pos = len(np.where(np.sum(score_gt, axis=1)>0)[0])
        n_neg = len(np.where(np.sum(score_gt, axis=1)==0)[0])

        TP = len(np.where(np.sum(score * score_gt, axis=1)>0)[0])
        TN = len(np.where(np.sum((1-score) * (1-score_gt), axis=1)==score_gt.shape[0])[0])

        FN = n_pos - TP
        FP = n_neg - TN 

        return TP, FP, FN, TN

    def get_precision_recall(self, score, score_gt):
        precisions, recalls, fscores = [],[],[]
        for i in range(score.shape[0]):
            TP, FP, FN, TN = self.calc_TFPN(score[i], score_gt[i])
            precision = np.average(np.nan_to_num(TP / (TP+FP+self.eps)))*100.
            recall = np.average(np.nan_to_num(TP / (TP+FN+self.eps)))*100.
            fscore = np.nan_to_num(2*precision*recall / (precision + recall))
            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)
            
        return precisions, recalls, fscores

if __name__ == '__main__':
    pass
