from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.feature.laf import laf_is_inside_image
from kornia.feature.orientation import PassLAF
from kornia.feature.responses import BlobHessian
from kornia.geometry.subpix import ConvSoftArgmax3d
from kornia.geometry.transform import ScalePyramid
from kornia.testing import KORNIA_CHECK_LAF


class FixedScaleSpaceDetector(nn.Module):
    r"""Module for differentiable local feature detection, as close as possible to classical local feature detectors
    like Harris, Hessian-Affine or SIFT (DoG).

    It has 5 modules inside: scale pyramid generator, response ("cornerness") function,
    soft nms function, affine shape estimator and patch orientation estimator.
    Each of those modules could be replaced with learned custom one, as long, as
    they respect output shape.

    Args:
        num_features: Number of features to detect. In order to keep everything batchable,
          output would always have num_features output, even for completely homogeneous images.
        mr_size: multiplier for local feature scale compared to the detection scale.
          6.0 is matching OpenCV 12.0 convention for SIFT.
        scale_pyr_module: generates scale pyramid. See :class:`~kornia.geometry.ScalePyramid` for details.
          Default: ScalePyramid(3, 1.6, 10).
        resp_module: calculates ``'cornerness'`` of the pixel.
        nms_module: outputs per-patch coordinates of the response maxima.
          See :class:`~kornia.geometry.ConvSoftArgmax3d` for details.
        ori_module: for local feature orientation estimation. Default:class:`~kornia.feature.PassLAF`,
           which does nothing. See :class:`~kornia.feature.LAFOrienter` for details.
        aff_module: for local feature affine shape estimation. Default: :class:`~kornia.feature.PassLAF`,
            which does nothing. See :class:`~kornia.feature.LAFAffineShapeEstimator` for details.
        minima_are_also_good: if True, then both response function minima and maxima are detected
            Useful for symmetric response functions like DoG or Hessian. Default is False
    """

    def __init__(
            self,
            num_features: int = 500,
            mr_size: float = 6.0,
            scale_pyr_module: nn.Module = ScalePyramid(3, 1.6, 15),
            resp_module: nn.Module = BlobHessian(),
            nms_module: nn.Module = ConvSoftArgmax3d(
                (3, 3, 3), (1, 1, 1), (1, 1, 1), normalized_coordinates=False, output_value=True
            ),
            ori_module: nn.Module = PassLAF(),
            aff_module: nn.Module = PassLAF(),
            minima_are_also_good: bool = False,
            scale_space_response=False,
    ):
        super().__init__()
        self.mr_size = mr_size
        self.num_features = num_features
        self.scale_pyr = scale_pyr_module
        self.resp = resp_module
        self.nms = nms_module
        self.ori = ori_module
        self.aff = aff_module
        self.minima_are_also_good = minima_are_also_good
        # scale_space_response should be True if the response function works on scale space
        # like Difference-of-Gaussians
        self.scale_space_response = scale_space_response

    def __repr__(self):
        return (
                self.__class__.__name__ + '('
                                          'num_features='
                + str(self.num_features)
                + ', '
                + 'mr_size='
                + str(self.mr_size)
                + ', '
                + 'scale_pyr='
                + self.scale_pyr.__repr__()
                + ', '
                + 'resp='
                + self.resp.__repr__()
                + ', '
                + 'nms='
                + self.nms.__repr__()
                + ', '
                + 'ori='
                + self.ori.__repr__()
                + ', '
                + 'aff='
                + self.aff.__repr__()
                + ')'
        )

    def detect(
            self, img: torch.Tensor, num_feats: int, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev: torch.device = img.device
        dtype: torch.dtype = img.dtype
        sp, sigmas, _ = self.scale_pyr(img)

        all_responses: List[torch.Tensor] = []
        all_lafs: List[torch.Tensor] = []

        px_size = 0.5 if self.scale_pyr.double_image else 1.0

        for oct_idx, octave in enumerate(sp):
            sigmas_oct = sigmas[oct_idx]
            B, CH, L, H, W = octave.size()
            # Run response function
            if self.scale_space_response:
                oct_resp = self.resp(octave, sigmas_oct.view(-1))
            else:
                oct_resp = self.resp(octave.permute(0, 2, 1, 3, 4).reshape(B * L, CH, H, W), sigmas_oct.view(-1)).view(
                    B, L, CH, H, W
                )
                # We want nms for scale responses, so reorder to (B, CH, L, H, W)
                oct_resp = oct_resp.permute(0, 2, 1, 3, 4)
                # 3rd extra level is required for DoG only
                if self.scale_pyr.extra_levels % 2 != 0:  # type: ignore
                    oct_resp = oct_resp[:, :, :-1]

            if mask is not None:
                assert False
                oct_mask: torch.Tensor = _create_octave_mask(mask, oct_resp.shape)
                oct_resp = oct_mask * oct_resp

            coord_max, response_max = self.nms(oct_resp)

            if self.minima_are_also_good:
                coord_min, response_min = self.nms(-oct_resp)
                take_min_mask = (response_min > response_max).to(response_max.dtype)
                response_max = response_min * take_min_mask + (1 - take_min_mask) * response_max
                coord_max = coord_min * take_min_mask.unsqueeze(2) + (1 - take_min_mask.unsqueeze(2)) * coord_max

            # Now, lets crop out some small responses
            responses_flatten = response_max.reshape(response_max.size(0), -1)  # [B, N]
            max_coords_flatten = coord_max.reshape(response_max.size(0), 3, -1).permute(0, 2, 1)  # [B, N, 3]

            if responses_flatten.size(1) > num_feats:
                resp_flat_best, idxs = torch.topk(responses_flatten, k=num_feats, dim=1)
                max_coords_best = torch.gather(max_coords_flatten, 1, idxs.unsqueeze(-1).repeat(1, 1, 3))
            else:
                resp_flat_best = responses_flatten
                max_coords_best = max_coords_flatten

            B, N = resp_flat_best.size()

            # Converts scale level index from ConvSoftArgmax3d to the actual scale, using the sigmas
            max_coords_best = _scale_index_to_scale(
                max_coords_best, sigmas_oct, self.scale_pyr.n_levels  # type: ignore
            )

            # Create local affine frames (LAFs)
            rotmat = torch.eye(2, dtype=dtype, device=dev).view(1, 1, 2, 2)
            current_lafs = torch.cat(
                [
                    self.mr_size * max_coords_best[:, :, 0].view(B, N, 1, 1) * rotmat,
                    max_coords_best[:, :, 1:3].view(B, N, 2, 1),
                ],
                dim=3,
            )

            # Zero response lafs, which touch the boundary
            good_mask = laf_is_inside_image(current_lafs, octave[:, 0])
            good_mask[:] = True
            resp_flat_best = resp_flat_best * good_mask.to(dev, dtype)

            # Normalize LAFs
            current_lafs = normalize_laf(current_lafs, octave[:, 0],
                                         px_size)  # We don`t need # of scale levels, only shape

            all_responses.append(resp_flat_best)

            all_lafs.append(current_lafs)
            px_size *= 2

        # Sort and keep best n
        responses: torch.Tensor = torch.cat(all_responses, dim=1)
        lafs: torch.Tensor = torch.cat(all_lafs, dim=1)
        responses, idxs = torch.topk(responses, k=num_feats, dim=1)
        lafs = torch.gather(lafs, 1, idxs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 3))
        denom = denormalize_laf(lafs, img)

        return responses, denom

    def forward(  # type: ignore
            self, img: torch.Tensor, mask: Optional[torch.Tensor] = None  # type: ignore
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Three stage local feature detection. First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.

        Args:
            img: image to extract features with shape [BxCxHxW]
            mask: a mask with weights where to apply the response function. The shape must be the same as
              the input image.

        Returns:
            lafs: shape [BxNx2x3]. Detected local affine frames.
            responses: shape [BxNx1]. Response function values for corresponding lafs
        """
        responses, lafs = self.detect(img, self.num_features, mask)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        return lafs, responses


def denormalize_laf(LAF: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """De-normalize LAFs from scale to image scale.

        B,N,H,W = images.size()
        MIN_SIZE = min(H,W)
        [a11 a21 x]
        [a21 a22 y]
        becomes
        [a11*MIN_SIZE a21*MIN_SIZE x*W]
        [a21*MIN_SIZE a22*MIN_SIZE y*H]

    Args:
        LAF:
        images: images, LAFs are detected in.

    Returns:
        the denormalized lafs.

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    KORNIA_CHECK_LAF(LAF)
    _, _, h, w = images.size()
    wf = float(w)
    hf = float(h)
    min_size = min(hf, wf)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype).to(LAF.device) * min_size
    coef[0, 0, 0, 2] = 1.0
    coef[0, 0, 1, 2] = 1.0
    ret = coef.expand_as(LAF) * LAF
    return ret


def normalize_laf(LAF: torch.Tensor, images: torch.Tensor, pixel_size: float) -> torch.Tensor:
    """Normalize LAFs to [0,1] scale from pixel scale. See below:
        B,N,H,W = images.size()
        MIN_SIZE = min(H,W)
        [a11 a21 x]
        [a21 a22 y]
        becomes:
        [a11/MIN_SIZE a21/MIN_SIZE x/W]
        [a21/MIN_SIZE a22/MIN_SIZE y/H]

    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in

    Returns:
        LAF: (torch.Tensor).

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    KORNIA_CHECK_LAF(LAF)
    _, _, h, w = images.size()
    wf: float = float(w)
    hf: float = float(h)
    min_size = min(hf, wf)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype).to(LAF.device) / min_size
    coef[0, 0, 0, 2] = pixel_size
    coef[0, 0, 1, 2] = pixel_size
    ret = coef.expand_as(LAF) * LAF
    return ret


def _scale_index_to_scale(max_coords: torch.Tensor, sigmas: torch.Tensor, num_levels: int) -> torch.Tensor:
    r"""Auxiliary function for ScaleSpaceDetector. Converts scale level index from ConvSoftArgmax3d to the actual
    scale, using the sigmas from the ScalePyramid output.

    Args:
        max_coords: tensor [BxNx3].
        sigmas: tensor [BxNxD], D >= 1

    Returns:
        tensor [BxNx3].
    """
    # depth (scale) in coord_max is represented as (float) index, not the scale yet.
    # we will interpolate the scale using pytorch.grid_sample function
    # Because grid_sample is for 4d input only, we will create fake 2nd dimension
    # ToDo: replace with 3d input, when grid_sample will start to support it

    # Reshape for grid shape
    B, N, _ = max_coords.shape
    scale_coords = max_coords[:, :, 0].contiguous().view(-1, 1, 1, 1)
    # Replace the scale_x_y
    out = torch.cat(
        [sigmas[0, 0] * torch.pow(2.0, scale_coords / float(num_levels)).view(B, N, 1), max_coords[:, :, 1:]], dim=2
    )
    return out


def _create_octave_mask(mask: torch.Tensor, octave_shape: List[int]) -> torch.Tensor:
    r"""Downsample a mask based on the given octave shape."""
    mask_shape = octave_shape[-2:]
    mask_octave = F.interpolate(mask, mask_shape, mode='bilinear', align_corners=False)  # type: ignore
    return mask_octave.unsqueeze(1)
