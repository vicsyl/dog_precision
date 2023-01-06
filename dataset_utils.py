import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from kornia.utils import image_to_tensor


def get_integer_scale(scale, h, w):

    gcd = math.gcd(w, h)
    real_scale_gcd = round(gcd * scale)
    real_scale = real_scale_gcd / gcd

    if real_scale == 0.0 or math.fabs(real_scale - scale) > 0.02:
        raise Exception("scale {} cannot be effectively realized for w, h = {}, {} in integer domain".format(scale, w, h))

    return real_scale


def read_imgs(file_paths, show=False):
    imgs = []
    for i, file in enumerate(file_paths):
        img = Image.open(file)
        img = np.array(img)
        imgs.append(img)
        if show:
            plt.figure()
            plt.imshow(img)
            plt.title(i + 1)
            plt.show()
            plt.close()
    return imgs


def rotation_gt_Hs(img):
    ret = [np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, img.shape[1] - 1], [0.0, 0.0, 1.0]]),
           np.array([[-1.0, 0.0, img.shape[1] - 1], [0.0, -1.0, img.shape[0] - 1], [0.0, 0.0, 1.0]]),
           np.array([[0.0, -1.0, img.shape[0] - 1], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])]
    return ret


def scale_img(img, scale):

    h, w = img.shape[:2]
    scale_o = scale
    # the scale needs to be (slightly) changed so that the aspect ratio is
    scale = get_integer_scale(scale, h, w)
    print(f"scale: {scale_o} => {scale}")

    H_gt = np.array([
        [scale, 0., 0.5 * (scale - 1)],
        [0., scale, 0.5 * (scale - 1)],
        [0., 0., 1.],
    ])

    dsize = (round(w * scale), round(h * scale))
    pil = Image.fromarray(img)
    pil_scaled = pil.resize(size=dsize, resample=Image.Resampling.LANCZOS)
    np_scaled = np.array(pil_scaled)

    return H_gt, np_scaled


def np_show(img, title=None):
    plt.figure()
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()
    plt.close()


def Hs_imgs_for_bark():
    # see https://www.robots.ox.ac.uk/~vgg/data/affine/

    Hs_bark = [
        [[0.7022029025774007, 0.4313737491020563, -127.94661199701689],
         [-0.42757325092889575, 0.6997834349758094, 201.26193857481698],
         [4.083733373964227E-6, 1.5076445750988132E-5, 1.0]],

        [[-0.48367041358997964, -0.2472935325077872, 870.2215120216712],
         [0.29085746679198893, -0.45733473891783305, 396.1604918833091],
         [-3.578663704630333E-6, 6.880007548843957E-5, 1.0]],

        [[-0.20381418476462312, 0.3510201271914591, 247.1085214229702],
         [-0.3499531830464912, -0.1975486500576974, 466.54576370699766],
         [-1.5735788289619667E-5, 1.0242951905091244E-5, 1.0]],

        [[0.30558415717792214, 0.12841186681168829, 200.94588793078017],
         [-0.12861248979242065, 0.3067557133397112, 133.77000196887894],
         [2.782320090398499E-6, 5.770764104061954E-6, 1.0]],

        [[-0.23047631546234373, -0.10655686701035443, 583.3200507850402],
         [0.11269946585180685, -0.20718914340861153, 355.2381263740649],
         [-3.580280012615393E-5, 3.2283960511548054E-5, 1.0]],
    ]

    Hs_bark = np.array(Hs_bark)
    files_bark = [f"imgs/bark/img{i + 1}.ppm" for i in range(6)]
    imgs_bark = read_imgs(files_bark, show=False)
    return Hs_bark, imgs_bark


def Hs_imgs_for_boat():
    # see https://www.robots.ox.ac.uk/~vgg/data/affine/

    Hs_boat = [
        [[8.5828552e-01, 2.1564369e-01, 9.9101418e+00],
         [-2.1158440e-01, 8.5876360e-01, 1.3047838e+02],
         [2.0702435e-06, 1.2886110e-06, 1.0000000e+00]],

        [[5.6887079e-01, 4.6997572e-01, 2.5515642e+01],
         [-4.6783159e-01, 5.6548769e-01, 3.4819925e+02],
         [6.4697420e-06, -1.1704138e-06, 1.0000000e+00]],

        [[1.0016637e-01, 5.2319717e-01, 2.0587932e+02],
         [-5.2345249e-01, 8.7390786e-02, 5.3454522e+02],
         [9.4931475e-06, -9.8296917e-06, 1.0000000e+00]],

        [[4.2310823e-01, -6.0670438e-02, 2.6635003e+02],
         [6.2730152e-02, 4.1652096e-01, 1.7460201e+02],
         [1.5812849e-05, -1.4368783e-05, 1.0000000e+00]],

        [[2.9992872e-01, 2.2821975e-01, 2.2930182e+02],
         [-2.3832758e-01, 2.4564042e-01, 3.6767399e+02],
         [9.9064973e-05, -5.8498673e-05, 1.0000000e+00]]
    ]
    Hs_boat = np.array(Hs_boat)
    files_boat = [f"imgs/boat/img{i + 1}.pgm" for i in range(6)]
    imgs_boat = read_imgs(files_boat, show=False)
    return Hs_boat, imgs_boat


def Hs_imgs_for_rotation_for_file(file, show=False):

    img = Image.open(file)
    img = np.array(img)

    if show:
        np_show(img, "original")

    Hs_gt = rotation_gt_Hs(img)
    imgs = [img] + [np.rot90(img, rotation_index, [0, 1]).copy() for rotation_index in range(1, 4)]
    return Hs_gt, imgs


def Hs_imgs_for_scaling_for_file(file, scales, crop_h2=False):

    img = Image.open(file)
    img = np.array(img)
    # this is done so that the img dimensions have a big gcd (i.e. gcd(512 - 2, 765) == 255),
    # and by extension so that the aspect ratio can be exactly kept when scaled (see `get_integer_scale`)
    if crop_h2:
        img = img[:img.shape[0] - 2]

    h_i_tuples = [scale_img(img, scale) for scale in scales]
    Hs_gt = [e[0] for e in h_i_tuples]
    imgs_r = [e[1] for e in h_i_tuples]
    imgs = [img] + imgs_r
    return Hs_gt, imgs


def Hs_imgs_for_scaling():
    files_bark = [f"imgs/bark/img{i + 1}.ppm" for i in range(6)]
    scales = [scale_int / 10 for scale_int in range(2, 10)]
    Hs, imgs = Hs_imgs_for_scaling_for_file(files_bark[0], scales, crop_h2=True)
    return Hs, imgs, scales


def Hs_imgs_for_rotation():
    files_bark = [f"imgs/bark/img{i + 1}.ppm" for i in range(6)]
    return Hs_imgs_for_rotation_for_file(files_bark[0], show=False)


def transform_to_torch(Hs, imgs, device):
    Hs_t = torch.from_numpy(np.array(Hs)).to(dtype=torch.float32)
    imgs_t_np = [np.array(ImageOps.grayscale(Image.fromarray(img))) for img in imgs]
    imgs_t = [(image_to_tensor(img, False).float() / 255.0).to(device=device) for img in imgs_t_np]
    return Hs_t, imgs_t


def Hs_imgs_for_scaling_torch(device=torch.device("cpu")):
    Hs, imgs, scales = Hs_imgs_for_scaling()
    Hs_t, imgs_t = transform_to_torch(Hs, imgs, device)
    return Hs_t, imgs_t, scales


def Hs_imgs_for_rotation_torch(device=torch.device("cpu")):

    Hs, imgs = Hs_imgs_for_rotation()
    Hs_t, imgs_t = transform_to_torch(Hs, imgs, device)
    return Hs_t, imgs_t


def Hs_imgs_for_boat_torch(device=torch.device("cpu")):
    Hs, imgs = Hs_imgs_for_boat()
    Hs_t, imgs_t = transform_to_torch(Hs, imgs, device)
    return Hs_t, imgs_t


def Hs_imgs_for_bark_torch(device=torch.device("cpu")):
    Hs, imgs = Hs_imgs_for_bark()
    Hs_t, imgs_t = transform_to_torch(Hs, imgs, device)
    return Hs_t, imgs_t
