from typing import Tuple

import torch
import torch.nn as nn
from kornia.filters.sobel import spatial_gradient3d
from kornia.geometry.subpix.nms import nms3d
from kornia.utils import create_meshgrid3d
from kornia.utils._compat import torch_version_geq
from kornia.utils.helpers import safe_solve_with_mask


def conv_quad_interp3d(
    input: torch.Tensor, strict_maxima_bonus: float = 10.0, eps: float = 1e-7,
    scatter_fix=True, swap_xy_fix=True, final_adjustment=0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the single iteration of quadratic interpolation of the extremum (max or min).

    Args:
        input: the given heatmap with shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
        strict_maxima_bonus: pixels, which are strict maxima will score (1 + strict_maxima_bonus) * value.
          This is needed for mimic behavior of strict NMS in classic local features
        eps: parameter to control the hessian matrix ill-condition number.

    Returns:
        the location and value per each 3x3x3 window which contains strict extremum, similar to one done is SIFT.
        :math:`(N, C, 3, D_{out}, H_{out}, W_{out})`, :math:`(N, C, D_{out}, H_{out}, W_{out})`,

        where

         .. math::
             D_{out} = \left\lfloor\frac{D_{in}  + 2 \times \text{padding}[0] -
             (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

         .. math::
             H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[1] -
             (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

         .. math::
             W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[2] -
             (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples:
        >>> input = torch.randn(20, 16, 3, 50, 32)
        >>> nms_coords, nms_val = conv_quad_interp3d(input, 1.0)
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 5:
        raise ValueError(f"Invalid input shape, we expect BxCxDxHxW. Got: {input.shape}")

    B, CH, D, H, W = input.shape
    grid_global: torch.Tensor = create_meshgrid3d(D, H, W, False, device=input.device).permute(0, 4, 1, 2, 3)
    grid_global = grid_global.to(input.dtype)

    # to determine the location we are solving system of linear equations Ax = b, where b is 1st order gradient
    # and A is Hessian matrix
    b: torch.Tensor = spatial_gradient3d(input, order=1, mode='diff')  #
    b = b.permute(0, 1, 3, 4, 5, 2).reshape(-1, 3, 1)
    A: torch.Tensor = spatial_gradient3d(input, order=2, mode='diff')
    A = A.permute(0, 1, 3, 4, 5, 2).reshape(-1, 6)
    dxx = A[..., 0]
    dyy = A[..., 1]
    dss = A[..., 2]
    dxy = 0.25 * A[..., 3]  # normalization to match OpenCV implementation
    dys = 0.25 * A[..., 4]  # normalization to match OpenCV implementation
    dxs = 0.25 * A[..., 5]  # normalization to match OpenCV implementation

    Hes = torch.stack([dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss], dim=-1).view(-1, 3, 3)
    if not torch_version_geq(1, 10):
        # The following is needed to avoid singular cases
        Hes += torch.rand(Hes[0].size(), device=Hes.device).abs()[None] * eps

    nms_mask: torch.Tensor = nms3d(input, (3, 3, 3), True)
    x_solved: torch.Tensor = torch.zeros_like(b)
    x_solved_masked, _, solved_correctly = safe_solve_with_mask(b[nms_mask.view(-1)], Hes[nms_mask.view(-1)])

    #  Kill those points, where we cannot solve
    new_nms_mask = nms_mask.masked_scatter(nms_mask, solved_correctly)

    if scatter_fix:
        x_solved[torch.where(new_nms_mask.view(-1, 1, 1))[0]] = x_solved_masked[solved_correctly]
    else:
        x_solved.masked_scatter_(new_nms_mask.view(-1, 1, 1), x_solved_masked[solved_correctly])

    dx: torch.Tensor = -x_solved

    # Ignore ones, which are far from window center
    mask1 = dx.abs().max(dim=1, keepdim=True)[0] > 0.7
    dx.masked_fill_(mask1.expand_as(dx), 0)
    dy: torch.Tensor = 0.5 * torch.bmm(b.permute(0, 2, 1), dx)
    y_max = input + dy.view(B, CH, D, H, W)
    if strict_maxima_bonus > 0:
        y_max += strict_maxima_bonus * new_nms_mask.to(input.dtype)

    dx_res: torch.Tensor = dx.flip(1).reshape(B, CH, D, H, W, 3).permute(0, 1, 5, 2, 3, 4)
    if swap_xy_fix:
        dx_res[:, :, [1, 2]] = dx_res[:, :, [2, 1]]
    # +0.5 -> #matches (MAE consistently 1.0)
    # -0.5 -> #precision (MAE 1.0/0.0 - not double image/double image)
    # 0 -> (0.76 - 1.0 / 0.5 - not double image/double image)
    dx_res[:, :, 1:3] += final_adjustment

    coords_max: torch.Tensor = grid_global.repeat(B, 1, 1, 1, 1).unsqueeze(1)
    coords_max = coords_max + dx_res
    return coords_max, y_max


class FixedConvQuadInterp3d(nn.Module):
    r"""Calculate soft argmax 3d per window.

    See :func:`~kornia.geometry.subpix.conv_quad_interp3d` for details.
    """

    def __init__(self, strict_maxima_bonus: float = 10.0, eps: float = 1e-7, scatter_fix=True, swap_xy_fix=True,
                 final_adjustment=0.0) -> None:
        super().__init__()
        self.strict_maxima_bonus = strict_maxima_bonus
        self.eps = eps
        self.scatter_fix = scatter_fix
        self.swap_xy_fix = swap_xy_fix
        self.final_adjustment = final_adjustment
        return

    def __str__(self):
        return f"sf={self.scatter_fix},sw={self.swap_xy_fix},adj={self.final_adjustment}"

    def __repr__(self) -> str:
        return self.__str__()

    def forward(self, x: torch.Tensor):  # type: ignore
        return conv_quad_interp3d(x, self.strict_maxima_bonus, self.eps, self.scatter_fix, self.swap_xy_fix,
                                  final_adjustment=self.final_adjustment)
