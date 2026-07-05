import torch.nn as nn
import torch.nn.functional as F

class ReconsLoss(nn.Module):
    def __init__(self):
        super(ReconsLoss, self).__init__()

    def forward(self, pred_label, truth_label, para):
        norm, pos_weight = para[0], para[1]
        loss = norm * F.binary_cross_entropy_with_logits(pred_label, 
                            truth_label, pos_weight=truth_label * pos_weight)

        return loss
