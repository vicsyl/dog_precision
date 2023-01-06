import torch

from kornia.geometry.transform import warp_perspective


def get_visible_part_mean_absolute_reprojection_error_torch(img1_t, img2_t, H_gt_t, H_t, device):
    """We reproject the image 1 mask to image2 and back to get the visible part mask.
    Then we average the reprojection absolute error over that area
    """
    h, w = img1_t.shape[2:]
    mask1_t = torch.ones((1, 1, h, w), device=device)

    H_gt_t = H_gt_t[None]
    mask1in2_t = warp_perspective(mask1_t, H_gt_t, img2_t.shape[2:][::-1])
    mask1inback_t = warp_perspective(mask1in2_t, torch.linalg.inv(H_gt_t), img1_t.shape[2:][::-1]) > 0

    xi_t = torch.arange(w, device=device)
    yi_t = torch.arange(h, device=device)
    xg_t, yg_t = torch.meshgrid(xi_t, yi_t, indexing='xy')

    coords_t = torch.cat(
        [xg_t.reshape(*xg_t.shape, 1), yg_t.reshape(*yg_t.shape, 1), torch.ones(*yg_t.shape, 1, device=device)], dim=2
    )

    def get_xy_rep(H_loc):
        xy_rep_t = H_loc.to(torch.float32) @ coords_t.reshape(-1, 3, 1).to(torch.float32)
        xy_rep_t /= xy_rep_t[:, 2:3]
        xy_rep_t = xy_rep_t[:, :2]
        return xy_rep_t

    xy_rep_gt_t = get_xy_rep(H_gt_t)
    xy_rep_est_t = get_xy_rep(H_t)
    error_t = torch.sqrt(((xy_rep_gt_t - xy_rep_est_t) ** 2).sum(axis=1)).reshape(xg_t.shape) * mask1inback_t[0, 0].T
    mean_error_t = error_t.sum() / mask1inback_t.sum()

    return mean_error_t.detach().cpu().item()


