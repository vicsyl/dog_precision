import torch


def upscale_double(x):

    """
    :param x:
    :return: x upscaled 2-fold, even indices maps to original indices,
    other indices bilinearly
    """

    # x : [B, CH, H, W]
    # img_t = torch.from_numpy(img).permute(2, 0, 1)[None]
    B, CH, H, W = x.shape

    upscaled = torch.zeros((B, CH, H * 2, W * 2)).to(device=x.device)
    upscaled[:, :, ::2, ::2] = x
    # sh = list(img_t.shape)
    # sh[-2] += 1
    # sh[-1] += 1

    upscaled[:, :, ::2, 1::2][..., :-1] = (upscaled[:, :, ::2, ::2][..., :-1] + upscaled[:, :, ::2, 2::2]) / 2
    upscaled[:, :, ::2, -1] = upscaled[:, :, ::2, -2]
    upscaled[:, :, 1::2, :][..., :-1, :] = (upscaled[:, :, ::2, :][..., :-1, :] + upscaled[:, :, 2::2, :]) / 2
    upscaled[:, :, -1, :] = upscaled[:, :, -2, :]

    return upscaled
