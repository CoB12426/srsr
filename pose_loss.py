from ultralytics import YOLO
import cv2
import pandas as pd
import os
import cv2
import torch

model = YOLO('./YOLO-POSE/yolo11x-pose.pt')
img = '/home/aki/Downloads/men.jpg'

results = model.predict(img)
for result in results:
    keypoints = result.keypoints
    visibilities = keypoints[:, :, 2]

print('vivi: ', visibilities)

#Object Keypoint Similarityの計算
def keypoint_similarity(gt_kpts, pred_kpts, sigmas, areas):
    EPSILON = torch.finfo(torch.float32).eps
    dist_sq = (gt_kpts[:, None, :, 0] - pred_kpts[..., 0])**2 + (gt_kpts[:, None, :, 1] - pred_kpts[..., 1])**2
    vis_mask = gt_kpts[..., 2].int() > 0
    k = 2*sigmas
    denom = 2 * (k**2) * (areas[:, None, None] + EPSILON)
    exp_term = dist_sq / denom
    oks = (torch.exp(-exp_term) * vis_mask[:, None, :]).sum(-1) / (vis_mask[:, None, :].sum(-1) + EPSILON)
    return oks
    '''
    gts_kpts:キーポイントのグラウンドトゥルース
    pred_kpts:予測されたキーポイント
    sigmas:キーポイントの標準偏差
    areas:グラウンドトゥルースの面積
    '''

