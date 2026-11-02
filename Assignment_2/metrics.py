##########################################################################
#   Segmentation metrics from an accumulated confusion matrix            #
#   mIoU, mean Dice, pixel accuracy, pet-vs-background IoU               #
##########################################################################

import numpy as np
import torch

import config


class SegMetrics:
    # Accumulates a NUM_CLASSES x NUM_CLASSES confusion matrix (rows = ground truth, columns = prediction)
    # over the whole split, ignoring IGNORE_INDEX pixels, so metrics are dataset-level, not per-image averages

    def __init__(self, num_classes = config.NUM_CLASSES, ignore_index = config.IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype = torch.int64)

    @torch.no_grad()
    def update(self, preds, targets):
        valid = targets != self.ignore_index
        index = targets[valid].to(torch.int64) * self.num_classes + preds[valid].to(torch.int64)
        counts = torch.bincount(index.cpu(), minlength = self.num_classes ** 2)
        self.confusion += counts.view(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.confusion.double()
        tp = cm.diag()
        gt, pred = cm.sum(1), cm.sum(0)
        present = gt > 0                                             # Classes absent from the split are skipped
        iou = tp / (gt + pred - tp).clamp(min = 1)
        dice = 2 * tp / (gt + pred).clamp(min = 1)
        # Pet vs background: merge the 37 breed classes into one "pet" class
        fg_tp = cm[1:, 1:].sum()
        fg_iou = fg_tp / (cm[1:, :].sum() + cm[:, 1:].sum() - fg_tp).clamp(min = 1)
        return {
            "miou": iou[present].mean().item(),
            "mdice": dice[present].mean().item(),
            "pixel_acc": (tp.sum() / cm.sum().clamp(min = 1)).item(),
            "pet_iou": fg_iou.item(),
            "per_class_iou": iou.tolist(),
        }


def species_confusion(confusion, species_of_breed):
    # Collapse the 38x38 pixel confusion into background / cat / dog, to separate
    # "wrong breed, right species" from "wrong species" errors (analysed in A2.2)
    groups = np.array([0] + [species_of_breed[b] for b in range(1, len(species_of_breed) + 1)])
    out = np.zeros((3, 3), dtype = np.int64)
    cm = np.asarray(confusion)
    for i in range(3):
        for j in range(3):
            out[i, j] = cm[np.ix_(groups == i, groups == j)].sum()
    return out
