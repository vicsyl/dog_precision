import warnings
from enum import Enum
from warnings import catch_warnings

import cv2 as cv
import torch
from kornia.feature import BlobDoG
from kornia.feature import LAFAffineShapeEstimator, LAFOrienter
from kornia.feature import ScaleSpaceDetector
from kornia.feature.integrated import LocalFeature, LAFDescriptor, SIFTDescriptor
from kornia.feature.responses import BlobHessian
from kornia.feature.responses import CornerHarris
from kornia.geometry import ConvQuadInterp3d
from kornia.geometry import ScalePyramid
from kornia.geometry.transform import warp_perspective
from kornia.utils import image_to_tensor

from conv_quad_interp3d import FixedConvQuadInterp3d
from scale_pyramid import FixedDogScalePyramid
from scale_space_detector import FixedScaleSpaceDetector


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_sift_descriptor_with_hessian(version, double_image=True, device=get_device()):

    sp = get_scale_pyramid(version, double_image=double_image, min_size=15)
    nms_module = get_conv_quad_interp3d(version, eps=2e-4)
    resp = BlobHessian()
    with warnings.catch_warnings(record=True):
        aff_module = LAFAffineShapeEstimator(patch_size=19)

    if version == Version.ORIGINAL:
        hessian_affine_local_detector = ScaleSpaceDetector(num_features=8000,
                                                          resp_module=resp,
                                                          nms_module=nms_module,
                                                          mr_size=6.0,
                                                          scale_pyr_module=sp,
                                                          aff_module=aff_module,
                                                          ori_module=LAFOrienter(patch_size=19),
                                                          minima_are_also_good=True)
    else:
        hessian_affine_local_detector = FixedScaleSpaceDetector(num_features=8000,
                                                               resp_module=resp,
                                                               nms_module=nms_module,
                                                               mr_size=6.0,
                                                               scale_pyr_module=sp,
                                                               aff_module=aff_module,
                                                               ori_module=LAFOrienter(patch_size=19),
                                                               minima_are_also_good=True)
    return NumpyKorniaSiftDescriptor(local_feature=CustomSIFTFeature(hessian_affine_local_detector, device=device), device=device)


def get_sift_descriptor_with_harris(version, double_image=True, device=get_device()):

    sp = get_scale_pyramid(version, double_image=double_image, min_size=15)
    nms_module = get_conv_quad_interp3d(version, eps=2e-4)
    resp = CornerHarris(0.05)

    with catch_warnings(record=True):
        aff_module = LAFAffineShapeEstimator(patch_size=19)

    if version == Version.ORIGINAL:
        harris_affine_local_detector = ScaleSpaceDetector(num_features=8000,
                                                          resp_module=resp,
                                                          nms_module=nms_module,
                                                          mr_size=6.0,
                                                          scale_pyr_module=sp,
                                                          aff_module=aff_module,
                                                          ori_module=LAFOrienter(patch_size=19),
                                                          minima_are_also_good=False)

    else:
        harris_affine_local_detector = FixedScaleSpaceDetector(num_features=8000,
                                                               resp_module=resp,
                                                               nms_module=nms_module,
                                                               mr_size=6.0,
                                                               scale_pyr_module=sp,
                                                               aff_module=aff_module,
                                                               ori_module=LAFOrienter(patch_size=19),
                                                               minima_are_also_good=False)

    return NumpyKorniaSiftDescriptor(local_feature=CustomSIFTFeature(harris_affine_local_detector, device=device), device=device)


def get_sift_descriptor(version, device=get_device()):
    sp = get_scale_pyramid(version, double_image=True, min_size=32)
    nms_module = get_conv_quad_interp3d(version)

    if version == Version.ORIGINAL:
        detector = ScaleSpaceDetector(
            num_features=8000,
            resp_module=BlobDoG(),
            nms_module=nms_module,
            scale_pyr_module=sp,
            ori_module=LAFOrienter(19),
            scale_space_response=True,
            minima_are_also_good=True,
            mr_size=6.0,
        )
    else:
        detector = FixedScaleSpaceDetector(
            num_features=8000,
            resp_module=BlobDoG(),
            nms_module=nms_module,
            scale_pyr_module=sp,
            ori_module=LAFOrienter(19),
            scale_space_response=True,
            minima_are_also_good=True,
            mr_size=6.0,
        )

    return NumpyKorniaSiftDescriptor(local_feature=CustomSIFTFeature(detector, device=device), device=device)


class NumpyKorniaSiftDescriptor:

    def __init__(self, local_feature, device=torch.device('cpu')):
        self.local_feature = local_feature
        self.device = device

    @staticmethod
    def cv_kpt_from_laffs_responses(laffs, responses):
        kpts = []
        for i, response in enumerate(responses[0]):
            yx = laffs[0, i, :, 2]
            kp = cv.KeyPoint(yx[0].item(), yx[1].item(), response.item(), angle=0)
            kpts.append(kp)
        return kpts

    def detectAndCompute(self, img, mask):
        assert mask is None, "not implemented with non-trivial mask"
        if len(img.shape) == 2:
            img = img[:, :, None]
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_t = (image_to_tensor(img, False).float() / 255.).to(device=self.device)
        laffs, responses, descs = self.local_feature(img_t, mask=None)
        kpts = self.cv_kpt_from_laffs_responses(laffs, responses)
        descs = descs[0].cpu().detach().numpy()
        return kpts, descs

    def __call__(self, *args, **kwargs):
        return self.local_feature(*args, **kwargs)


class CustomSIFTFeature(LocalFeature):
    """ See kornia.feature.integrated.SIFTFeature """

    def __init__(
            self,
            scale_space_detector,
            rootsift: bool = True,
            device: torch.device = torch.device('cpu'),
    ):
        patch_size: int = 41
        detector = scale_space_detector.to(device)
        descriptor = LAFDescriptor(
            SIFTDescriptor(patch_size=patch_size, rootsift=rootsift), patch_size=patch_size, grayscale_descriptor=True
        ).to(device)
        super().__init__(detector, descriptor)


class Version(Enum):
    FIXED = 1
    NOT_FIXED = 2
    ORIGINAL = 3


def get_scale_pyramid(version, double_image=True, min_size=15):
    if version == Version.FIXED:
        return FixedDogScalePyramid(3, 1.6, min_size=min_size, double_image=double_image, fix_upscaling=True)
    elif version == Version.NOT_FIXED:
        return FixedDogScalePyramid(3, 1.6, min_size=min_size, double_image=double_image, fix_upscaling=False)
    elif version == Version.ORIGINAL:
        return ScalePyramid(3, 1.6, min_size=min_size, double_image=double_image)
    else:
        raise ValueError(f"unexpected version: {version}")


def get_conv_quad_interp3d(version, eps=1e-7):
    if version == Version.FIXED:
        return FixedConvQuadInterp3d(10,
                                     eps=eps,
                                     scatter_fix=True,
                                     swap_xy_fix=True)
    elif version == Version.NOT_FIXED:
        return FixedConvQuadInterp3d(10,
                                     eps=eps,
                                     scatter_fix=False,
                                     swap_xy_fix=False)
    elif version == Version.ORIGINAL:
        return ConvQuadInterp3d(10, eps=eps)
    else:
        raise ValueError(f"unexpected version: {version}")


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
