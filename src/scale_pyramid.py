import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from kornia.filters import gaussian_blur2d

from typing import List, Tuple
from utils import upscale_double


class FixedDogScalePyramid(nn.Module):
    r"""Create an scale pyramid of image, usually used for local feature detection.

    Images are consequently smoothed with Gaussian blur and downscaled.

    Args:
        n_levels: number of the levels in octave.
        init_sigma: initial blur level.
        min_size: the minimum size of the octave in pixels.
        double_image: add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this.

    Returns:
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output 1st: :math:`[(B, C, NL, H, W), (B, C, NL, H/2, W/2), ...]`
        - Output 2nd: :math:`[(B, NL), (B, NL), (B, NL), ...]`
        - Output 3rd: :math:`[(B, NL), (B, NL), (B, NL), ...]`

    Examples:
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = ScalePyramid(3, 15)(input)
    """

    def __init__(self,
                 n_levels: int = 3,
                 init_sigma: float = 1.6,
                 min_size: int = 15,
                 double_image: bool = False,
                 fix_upscaling=False):
        super().__init__()
        # 3 extra levels are needed for DoG nms.
        self.n_levels = n_levels
        self.extra_levels: int = 3
        self.init_sigma = init_sigma
        self.min_size = min_size
        self.border = min_size // 2 - 1
        self.sigma_step = 2 ** (1.0 / float(self.n_levels))
        self.double_image = double_image
        self.fix_upscaling = fix_upscaling

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(n_levels='
            + str(self.n_levels)
            + ', '
            + 'init_sigma='
            + str(self.init_sigma)
            + ', '
            + 'min_size='
            + str(self.min_size)
            + ', '
            + 'extra_levels='
            + str(self.extra_levels)
            + ', '
            + 'border='
            + str(self.border)
            + ', '
            + 'sigma_step='
            + str(self.sigma_step)
            + ', '
            + 'double_image='
            + str(self.double_image)
            + ')'
        )

    def gaussian_blur2d(self, x, ksize, sigma):
        return gaussian_blur2d(x, (ksize, ksize), (sigma, sigma), separable=True)

    def get_kernel_size(self, sigma: float):
        ksize = int(2.0 * 4.0 * sigma + 1.0)

        #  matches OpenCV, but may cause padding problem for small images
        #  PyTorch does not allow to pad more than original size.
        #  Therefore there is a hack in forward function

        if ksize % 2 == 0:
            ksize += 1
        return ksize

    def get_first_level(self, input):
        pixel_distance = 1.0
        cur_sigma = 0.5
        # Same as in OpenCV up to interpolation difference
        if self.double_image:
            if self.fix_upscaling:
                x = upscale_double(input)
            else:
                x = F.interpolate(input, scale_factor=2.0, mode='bilinear', align_corners=False)
            pixel_distance = 0.5
            cur_sigma *= 2.0
        else:
            x = input
        if self.init_sigma > cur_sigma:
            sigma = max(math.sqrt(self.init_sigma**2 - cur_sigma**2), 0.01)
            ksize = self.get_kernel_size(sigma)
            cur_level = self.gaussian_blur2d(x, ksize, sigma)
            cur_sigma = self.init_sigma
        else:
            cur_level = x
        return cur_level, cur_sigma, pixel_distance

    def forward(self, x: torch.Tensor) -> Tuple[List, List, List]:  # type: ignore
        bs, _, _, _ = x.size()
        cur_level, cur_sigma, pixel_distance = self.get_first_level(x)

        sigmas = [cur_sigma * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device).to(x.dtype)]
        pixel_dists = [pixel_distance * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device).to(x.dtype)]
        pyr = [[cur_level]]
        oct_idx = 0
        while True:
            cur_level = pyr[-1][0]
            for level_idx in range(1, self.n_levels + self.extra_levels):
                sigma = cur_sigma * math.sqrt(self.sigma_step**2 - 1.0)
                ksize = self.get_kernel_size(sigma)

                # Hack, because PyTorch does not allow to pad more than original size.
                # But for the huge sigmas, one needs huge kernel and padding...
                ksize = min(ksize, min(cur_level.size(2), cur_level.size(3)))
                if ksize % 2 == 0:
                    ksize += 1

                cur_level = self.gaussian_blur2d(cur_level, ksize, sigma)
                cur_sigma *= self.sigma_step
                pyr[-1].append(cur_level)
                sigmas[-1][:, level_idx] = cur_sigma
                pixel_dists[-1][:, level_idx] = pixel_distance
            _pyr = pyr[-1][-self.extra_levels]
            nextOctaveFirstLevel = _pyr[..., ::2, ::2]
            pixel_distance *= 2.0
            cur_sigma = self.init_sigma
            if min(nextOctaveFirstLevel.size(2), nextOctaveFirstLevel.size(3)) <= self.min_size:
                break
            pyr.append([nextOctaveFirstLevel])
            sigmas.append(cur_sigma * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device))
            pixel_dists.append(pixel_distance * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device))
            oct_idx += 1
        for i in range(len(pyr)):
            pyr[i] = torch.stack(pyr[i], dim=2)  # type: ignore
        return pyr, sigmas, pixel_dists
