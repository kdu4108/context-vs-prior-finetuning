import torch
import numpy as np

def iou(A, B, k):
    topk_A = A.flatten().abs().topk(k)
    topk_B = B.flatten().abs().topk(k)
    union_topk = set(torch.cat([topk_A.indices, topk_B.indices]).tolist())
    intersection_topk = set(topk_A.indices.tolist()) & set(topk_B.indices.tolist())
    len(union_topk), len(intersection_topk)
    # print("K:", topk, " - IoU: ", len(intersection_topk) / len(union_topk))
    return len(intersection_topk) / len(union_topk)
    
def iou_range(A, B, k_min=1, k_max=100):
    ious = []
    for k in range(k_min, k_max):
        ious.append(iou(A, B, k))
    return ious


def iou_auc(A, B, k_min=1, k_max=100):
    ious = iou_range(A, B, k_min, k_max)
    return np.trapz(ious)

def cosine(A,B):
    # normalise A and B to range -1, 1
    return torch.nn.functional.cosine_similarity(A.flatten(), B.flatten(), dim=0).item()
