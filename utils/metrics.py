import numpy as np
from scipy.spatial import KDTree
import trimesh
import torch


eps = 1e-10


def IOU(pred, gt):
    '''Expect x and y to be in shape [batch, num_pts, 1]'''
    inter = torch.logical_and(pred, gt)
    union = torch.logical_or(pred, gt)
    iou = inter.sum([1]) / (union.sum([1]) + eps)

    return iou.mean()

def chamfer_and_score(pts_gt, mesh_pred, tau=1e-2):
    """
    Compute a symmetric chamfer distance, i.e. the sum of both chamfers.
    pts_gt: numpy.array of ground-truth points.
    pts_pred: numpy.array of sampled points from the extracted surface.
    """
    num_sample = pts_gt.shape[0]
    pts_pred = trimesh.sample.sample_surface(mesh_pred, num_sample)[0].astype(np.float32)

    # gt2pred
    pts_pred_kdtree = KDTree(pts_pred)
    gt2pred_chamfer, gt2pred_vertex_ids = pts_pred_kdtree.query(pts_gt)

    # pred2gt
    pts_gt_kdtree = KDTree(pts_gt)
    pred2gt_chamfer, pred2gt_vertex_ids = pts_gt_kdtree.query(pts_pred)

    # point to surface distance
    pt2surf = gt2pred_chamfer.mean()

    # symmetric chamfer distance
    sym_chamfer = (gt2pred_chamfer.mean() + pred2gt_chamfer.mean()) / 2

    # Fscore
    prec_tau = (pred2gt_chamfer <= tau).mean()
    recall_tau = (gt2pred_chamfer <= tau).mean()
    f_score = (2 * prec_tau * recall_tau) / max(prec_tau + recall_tau, eps)

    return pt2surf*1000, sym_chamfer*1000, recall_tau*100, f_score*100