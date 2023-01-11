import cv2 as cv
import numpy as np


def split_points(tentative_matches, kps0, kps1):
    src_pts = np.float32([kps0[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_pts = np.float32([kps1[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    kps0 = [kps0[m.queryIdx] for m in tentative_matches]
    kps1 = [kps1[m.trainIdx] for m in tentative_matches]
    return src_pts, dst_pts, kps0, kps1


def get_tentatives(kpts0, desc0, kpts1, desc1, ratio_threshold):
    matcher = cv.BFMatcher(crossCheck=False)
    knn_matches = matcher.knnMatch(desc0, desc1, k=2)
    matches2 = matcher.match(desc1, desc0)

    tentative_matches = []
    for m, n in knn_matches:
        if matches2[m.trainIdx].trainIdx != m.queryIdx:
            continue

        if m.distance < ratio_threshold * n.distance:
            tentative_matches.append(m)

    src, dst, kpts0, kpts1 = split_points(tentative_matches, kpts0, kpts1)
    return src, dst


def get_visible_part_mean_absolute_reprojection_error(img1, img2, H_gt, H):
    '''We reproject the image 1 mask to image2 and back to get the visible part mask.
    Then we average the reprojection absolute error over that area'''
    h, w = img1.shape[:2]
    mask1 = np.ones((h, w))
    mask1in2 = cv.warpPerspective(mask1, H_gt, img2.shape[:2][::-1])
    mask1inback = cv.warpPerspective(mask1in2, np.linalg.inv(H_gt), img1.shape[:2][::-1]) > 0
    xi = np.arange(w)
    yi = np.arange(h)
    xg, yg = np.meshgrid(xi, yi)
    coords = np.concatenate([xg.reshape(*xg.shape, 1), yg.reshape(*yg.shape, 1)], axis=-1)
    xy_rep_gt = cv.perspectiveTransform(coords.reshape(-1, 1, 2).astype(np.float32), H_gt.astype(np.float32)).squeeze(1)
    xy_rep_estimated = cv.perspectiveTransform(coords.reshape(-1, 1, 2).astype(np.float32),
                                               H.astype(np.float32)).squeeze(1)
    error = np.sqrt(((xy_rep_gt - xy_rep_estimated) ** 2).sum(axis=1)).reshape(xg.shape) * mask1inback
    mean_error = error.sum() / mask1inback.sum()
    return mean_error
