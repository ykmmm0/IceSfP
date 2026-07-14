import os
import imageio
import numpy as np
from PIL import Image


def rgb_loader(RGB_PATH):
    rgb_arr = imageio.imread(RGB_PATH)
    return rgb_arr


def mask_loader(MASK_PATH):
    mask_arr = imageio.imread(MASK_PATH)
    return mask_arr


def rgb2grey(RGB_ARR):
    return np.dot(RGB_ARR[..., :3], [0.33333333, 0.33333333, 0.33333333]).astype(np.uint8)


def normal_to_rgb(normals_to_convert):
    camera_normal_rgb = (normals_to_convert + 1) * 127.5
    return camera_normal_rgb.astype(np.uint8)


def scaledimg2floatimg(input_arr):
    return input_arr.astype(np.float32) / 255.0


def png_saver(PNG_PATH, ndarr):
    if ndarr.ndim == 3 and ndarr.shape[2] == 1:
        ndarr = ndarr[:, :, 0]
    imageio.imwrite(PNG_PATH, ndarr.astype(np.uint8))


def exr_loader(EXR_PATH, ndim=3):
    img = imageio.imread(EXR_PATH, format='EXR-FI').astype(np.float32)
    if ndim == 1:
        return img[:, :, 0] if img.ndim == 3 else img
    return img


def exr_saver(EXR_PATH, ndarr, ndim=3):
    arr_to_save = ndarr
    if ndim == 3:
        if ndarr.ndim == 3 and ndarr.shape[0] == 3:
            arr_to_save = ndarr.transpose(1, 2, 0)
        elif ndarr.ndim == 2:
            arr_to_save = np.stack([ndarr] * 3, axis=-1)
    imageio.imwrite(EXR_PATH, arr_to_save.astype(np.float32), format='EXR-FI')
if __name__ == '__main__':
    png_arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    png_saver('test.png', png_arr)
    exr_arr = np.random.rand(3, 128, 128).astype(np.float32)
    exr_saver('test.exr', exr_arr)
    loaded_exr = exr_loader('test.exr')
    print('EXR shape:', loaded_exr.shape)
