import numpy as np
import cv2
import torch


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def draw_affordance(rgb, affordance, alpha=0.5):
    # normalize affordance
    aff_min = affordance.min()
    aff_max = affordance.max()
    affordance = (affordance - aff_min) / (aff_max - aff_min)

    heatmap_img = cv2.applyColorMap(
        (affordance * 255.0).astype(np.uint8), cv2.COLORMAP_HOT
    )[:, :, ::-1]
    heatmap_img = cv2.cvtColor(heatmap_img, code=cv2.COLOR_BGR2RGB)
    vis = cv2.addWeighted(heatmap_img, alpha, rgb, 1 - alpha, 0)
    weight_img = (affordance * 255.0).astype(np.uint8)
    return vis, weight_img


def draw_affordance_center(rgb, affordance, alpha=0.5):
    # normalize affordance
    aff_min = affordance.min()
    aff_max = affordance.max()
    max_idx = np.where(affordance == aff_max)
    affordance = affordance == aff_max
    affordance = affordance * aff_max

    aff_center = torch.LongTensor([max_idx[1][0], max_idx[0][0]])
    aff_map = torch.zeros((rgb.shape[0], rgb.shape[1]))
    draw_gaussian(aff_map.numpy(), aff_center, radius=20)
    aff_map = aff_map.numpy()

    aff_map_gt = aff_map * 255.0
    aff_map_gt = aff_map_gt.numpy().astype(np.uint8)

    heatmap_img = cv2.applyColorMap(aff_map_gt, cv2.COLORMAP_HOT)[:, :, ::-1]
    heatmap_img = cv2.cvtColor(heatmap_img, code=cv2.COLOR_BGR2RGB)
    vis = cv2.addWeighted(heatmap_img, alpha, rgb, 1 - alpha, 0)
    return vis
