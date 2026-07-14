import os
import glob
import sys
import time
import numpy
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
import API.utils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class IcesfpDataset(Dataset):

    def __init__(self, label_dir, dolp_dir, aolp_dir, s0_dir, i0_dir, i45_dir, i90_dir, i135_dir, synthesis_normals_dir, mask_dir=None, c_dir=None, transform=None, input_only=None):
        self.dolp_dir = dolp_dir
        self.aolp_dir = aolp_dir
        self.synthesis_normals_dir = synthesis_normals_dir
        self.labels_dir = label_dir
        self.masks_dir = mask_dir
        self.c_dir = c_dir if c_dir is not None else os.path.join(os.path.dirname(s0_dir), 'C')
        self.transform = transform
        self.input_only = input_only
        self.s0_dir = s0_dir
        self.i0_dir = i0_dir
        self.i45_dir = i45_dir
        self.i90_dir = i90_dir
        self.i135_dir = i135_dir
        self._datalist_dolp = []
        self._datalist_aolp = []
        self._datalist_s0 = []
        self._datalist_i0 = []
        self._datalist_i45 = []
        self._datalist_i90 = []
        self._datalist_i135 = []
        self._datalist_input_normal_0 = []
        self._datalist_input_normal_1 = []
        self._datalist_input_normal_2 = []
        self._datalist_input_normal_3 = []
        self._datalist_input_normal_4 = []
        self._datalist_input_normal_5 = []
        self._datalist_label = []
        self._datalist_mask = []
        self._datalist_c = []
        self._create_lists_filenames()

    def __len__(self):
        return len(self._datalist_label)

    def __getitem__(self, index):
        dolpPath = self._datalist_dolp[index]
        aolpPath = self._datalist_aolp[index]
        s0_path = self._datalist_s0[index]
        i0_path = self._datalist_i0[index]
        i45_path = self._datalist_i45[index]
        i90_path = self._datalist_i90[index]
        i135_path = self._datalist_i135[index]
        norm_0_path = self._datalist_input_normal_0[index]
        norm_1_path = self._datalist_input_normal_1[index]
        norm_2_path = self._datalist_input_normal_2[index]
        norm_3_path = self._datalist_input_normal_3[index]
        norm_4_path = self._datalist_input_normal_4[index]
        norm_5_path = self._datalist_input_normal_5[index]
        if self.masks_dir is not None:
            mask_path = self._datalist_mask[index]
        label_path = self._datalist_label[index]
        c_path = self._datalist_c[index]
        dolpImg = API.utils.rgb_loader(dolpPath)
        aolpImg = API.utils.rgb_loader(aolpPath)
        s0_img = API.utils.rgb_loader(s0_path)
        i0_img = API.utils.rgb_loader(i0_path)
        i45_img = API.utils.rgb_loader(i45_path)
        i90_img = API.utils.rgb_loader(i90_path)
        i135_img = API.utils.rgb_loader(i135_path)
        norm_0 = API.utils.rgb_loader(norm_0_path)
        norm_1 = API.utils.rgb_loader(norm_1_path)
        norm_2 = API.utils.rgb_loader(norm_2_path)
        norm_3 = API.utils.rgb_loader(norm_3_path)
        norm_4 = API.utils.rgb_loader(norm_4_path)
        norm_5 = API.utils.rgb_loader(norm_5_path)
        c_img = API.utils.mask_loader(c_path)
        mask_img = API.utils.mask_loader(mask_path)
        label_img = API.utils.rgb_loader(label_path).transpose(2, 0, 1)
        if mask_img.ndim == 3 and mask_img.shape[2] == 3:
            mask_img = mask_img[:, :, 0]
        height = label_img.shape[1]
        width = label_img.shape[2]
        target_size = (height, width)
        dolpImg = np.array(Image.fromarray(dolpImg).resize((width, height), Image.BILINEAR))
        aolpImg = np.array(Image.fromarray(aolpImg).resize((width, height), Image.BILINEAR))
        s0_img = np.array(Image.fromarray(s0_img).resize((width, height), Image.BILINEAR))
        i0_img = np.array(Image.fromarray(i0_img).resize((width, height), Image.BILINEAR))
        i45_img = np.array(Image.fromarray(i45_img).resize((width, height), Image.BILINEAR))
        i90_img = np.array(Image.fromarray(i90_img).resize((width, height), Image.BILINEAR))
        i135_img = np.array(Image.fromarray(i135_img).resize((width, height), Image.BILINEAR))
        norm_0 = np.array(Image.fromarray(norm_0).resize((width, height), Image.BILINEAR))
        norm_1 = np.array(Image.fromarray(norm_1).resize((width, height), Image.BILINEAR))
        norm_2 = np.array(Image.fromarray(norm_2).resize((width, height), Image.BILINEAR))
        norm_3 = np.array(Image.fromarray(norm_3).resize((width, height), Image.BILINEAR))
        norm_4 = np.array(Image.fromarray(norm_4).resize((width, height), Image.BILINEAR))
        norm_5 = np.array(Image.fromarray(norm_5).resize((width, height), Image.BILINEAR))
        mask_img = np.array(Image.fromarray(mask_img).resize((width, height), Image.NEAREST))
        c_img = np.array(Image.fromarray(c_img).resize((width, height), Image.BILINEAR))
        input_img_arr = np.zeros([25, height, width], dtype=np.uint8)
        input_img_arr[0, :, :] = dolpImg
        input_img_arr[1, :, :] = aolpImg
        input_img_arr[2, :, :] = s0_img
        input_img_arr[3, :, :] = i0_img
        input_img_arr[4, :, :] = i45_img
        input_img_arr[5, :, :] = i90_img
        input_img_arr[6, :, :] = i135_img
        input_img_arr[7:10, :, :] = norm_0.transpose(2, 0, 1)
        input_img_arr[10:13, :, :] = norm_1.transpose(2, 0, 1)
        input_img_arr[13:16, :, :] = norm_2.transpose(2, 0, 1)
        input_img_arr[16:19, :, :] = norm_3.transpose(2, 0, 1)
        input_img_arr[19:22, :, :] = norm_4.transpose(2, 0, 1)
        input_img_arr[22:25, :, :] = norm_5.transpose(2, 0, 1)
        if self.masks_dir is not None:
            bool_mask_2d = mask_img == 0
            for c in range(input_img_arr.shape[0]):
                input_img_arr[c][bool_mask_2d] = 0
            for c in range(label_img.shape[0]):
                label_img[c][bool_mask_2d] = 0
        if self.transform:
            det_tf = self.transform.to_deterministic()
            input_img_arr = det_tf.augment_image(input_img_arr.transpose(1, 2, 0))
            input_img_arr = input_img_arr.transpose(2, 0, 1)
            label_img = det_tf.augment_image(label_img.transpose(1, 2, 0), hooks=ia.HooksImages(activator=self._activator_masks))
            label_img = label_img.transpose(2, 0, 1)
            if self.masks_dir is not None:
                mask_img = det_tf.augment_image(mask_img, hooks=ia.HooksImages(activator=self._activator_masks))
            c_img = det_tf.augment_image(c_img, hooks=ia.HooksImages(activator=self._activator_masks))
        input_tensor = transforms.ToTensor()(input_img_arr.copy().transpose(1, 2, 0))
        params_tensor = input_tensor[0:7, :, :]
        synthesis_normals_tensor = input_tensor[7:25, :, :]
        c_tensor = torch.from_numpy(c_img.copy()).float().unsqueeze(0) / 255.0
        label_tensor = torch.from_numpy(label_img).float()
        label_tensor = (label_tensor - 127) / 127.0
        label_tensor = nn.functional.normalize(label_tensor, p=2, dim=0)
        if self.masks_dir is not None:
            mask_tensor = torch.from_numpy(mask_img.copy()).unsqueeze(0)
        if self.masks_dir is not None:
            return (params_tensor, synthesis_normals_tensor, label_tensor, mask_tensor, c_tensor)
        else:
            return (params_tensor, synthesis_normals_tensor, label_tensor, c_tensor)

    def _create_lists_filenames(self):
        assert os.path.isdir(self.dolp_dir), 'Dataloader given input DoLP directory that does not exist: "%s"' % self.dolp_dir
        assert os.path.isdir(self.aolp_dir), 'Dataloader given input AoLP directory that does not exist: "%s"' % self.aolp_dir
        assert os.path.isdir(self.synthesis_normals_dir), 'Dataloader given input synthesis normals directory that does not exist: "%s"' % self.synthesis_normals_dir
        assert os.path.isdir(self.labels_dir), 'Dataloader given labels directory that does not exist: "%s"' % self.labels_dir
        assert os.path.isdir(self.c_dir), 'Dataloader given C directory that does not exist: "%s"' % self.c_dir
        if self.masks_dir is not None:
            assert os.path.isdir(self.masks_dir), 'Dataloader given masks_dir images directory that does not exist: "%s"' % self.masks_dir
        assert os.path.isdir(self.dolp_dir)
        assert os.path.isdir(self.aolp_dir)
        assert os.path.isdir(self.s0_dir)
        assert os.path.isdir(self.i0_dir)
        assert os.path.isdir(self.i45_dir)
        assert os.path.isdir(self.i90_dir)
        assert os.path.isdir(self.i135_dir)
        assert os.path.isdir(self.synthesis_normals_dir)
        assert os.path.isdir(self.labels_dir)
        assert os.path.isdir(self.c_dir)
        if self.masks_dir is not None:
            assert os.path.isdir(self.masks_dir)
        dolp_search_str = os.path.join(self.dolp_dir, '*.png')
        self._datalist_dolp = sorted(glob.glob(dolp_search_str))
        numDoLP = len(self._datalist_dolp)
        if numDoLP == 0:
            raise ValueError('No input DoLP files found in given directory. Searched in dir: {} '.format(dolp_search_str))
        aolp_search_str = os.path.join(self.aolp_dir, '*.png')
        self._datalist_aolp = sorted(glob.glob(aolp_search_str))
        numNAoLP = len(self._datalist_aolp)
        if numNAoLP == 0:
            raise ValueError('No input AoLP files found in given directory. Searched in dir: {} '.format(aolp_search_str))
        s0_search_str = os.path.join(self.s0_dir, '*.png')
        self._datalist_s0 = sorted(glob.glob(s0_search_str))
        num_s0 = len(self._datalist_s0)
        if num_s0 == 0:
            raise ValueError(f'No S0 images found: {s0_search_str}')
        i0_search_str = os.path.join(self.i0_dir, '*.png')
        self._datalist_i0 = sorted(glob.glob(i0_search_str))
        num_i0 = len(self._datalist_i0)
        if num_i0 == 0:
            raise ValueError(f'No I0 images found: {i0_search_str}')
        i45_search_str = os.path.join(self.i45_dir, '*.png')
        self._datalist_i45 = sorted(glob.glob(i45_search_str))
        num_i45 = len(self._datalist_i45)
        if num_i45 == 0:
            raise ValueError(f'No I45 images found: {i45_search_str}')
        i90_search_str = os.path.join(self.i90_dir, '*.png')
        self._datalist_i90 = sorted(glob.glob(i90_search_str))
        num_i90 = len(self._datalist_i90)
        if num_i90 == 0:
            raise ValueError(f'No I90 images found: {i90_search_str}')
        i135_search_str = os.path.join(self.i135_dir, '*.png')
        self._datalist_i135 = sorted(glob.glob(i135_search_str))
        num_i135 = len(self._datalist_i135)
        if num_i135 == 0:
            raise ValueError(f'No I135 images found: {i135_search_str}')
        c_search_str = os.path.join(self.c_dir, '*.png')
        self._datalist_c = sorted(glob.glob(c_search_str))
        num_c = len(self._datalist_c)
        if num_c == 0:
            raise ValueError(f'No precomputed C maps found: {c_search_str}')
        input_normal_0_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-0')
        input_normal_0_search_str = os.path.join(input_normal_0_search_str, '*-normal.png')
        self._datalist_input_normal_0 = sorted(glob.glob(input_normal_0_search_str))
        numNorm_0 = len(self._datalist_input_normal_0)
        if numNorm_0 == 0:
            raise ValueError('No input normal_0 files found in given directory. Searched in dir: {} '.format(input_normal_0_search_str))
        input_normal_1_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-1')
        input_normal_1_search_str = os.path.join(input_normal_1_search_str, '*-normal.png')
        self._datalist_input_normal_1 = sorted(glob.glob(input_normal_1_search_str))
        numNorm_1 = len(self._datalist_input_normal_1)
        if numNorm_1 == 0:
            raise ValueError('No input normal_1 files found in given directory. Searched in dir: {} '.format(input_normal_1_search_str))
        input_normal_2_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-2')
        input_normal_2_search_str = os.path.join(input_normal_2_search_str, '*-normal.png')
        self._datalist_input_normal_2 = sorted(glob.glob(input_normal_2_search_str))
        numNorm_2 = len(self._datalist_input_normal_2)
        if numNorm_2 == 0:
            raise ValueError('No input normal_2 files found in given directory. Searched in dir: {} '.format(input_normal_2_search_str))
        input_normal_3_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-3')
        input_normal_3_search_str = os.path.join(input_normal_3_search_str, '*-normal.png')
        self._datalist_input_normal_3 = sorted(glob.glob(input_normal_3_search_str))
        numNorm_3 = len(self._datalist_input_normal_3)
        if numNorm_3 == 0:
            raise ValueError('No input normal_3 files found in given directory. Searched in dir: {} '.format(input_normal_3_search_str))
        if not numNorm_0 == numNorm_1 == numNorm_2 == numNorm_3:
            raise ValueError('Numbers of input normals are different.')
        input_normal_4_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-4')
        input_normal_4_search_str = os.path.join(input_normal_4_search_str, '*-normal.png')
        self._datalist_input_normal_4 = sorted(glob.glob(input_normal_4_search_str))
        numNorm_4 = len(self._datalist_input_normal_4)
        if numNorm_4 == 0:
            raise ValueError('No input normal_4 files found in given directory. Searched in dir: {} '.format(input_normal_3_search_str))
        input_normal_5_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-5')
        input_normal_5_search_str = os.path.join(input_normal_5_search_str, '*-normal.png')
        self._datalist_input_normal_5 = sorted(glob.glob(input_normal_5_search_str))
        numNorm_5 = len(self._datalist_input_normal_5)
        if numNorm_5 == 0:
            raise ValueError('No input normal_4 files found in given directory. Searched in dir: {} '.format(input_normal_3_search_str))
        if not numNorm_0 == numNorm_1 == numNorm_2 == numNorm_3 == numNorm_4 == numNorm_5:
            raise ValueError('Numbers of input normals are different.')
        labels_search_str = os.path.join(self.labels_dir, '*.png')
        self._datalist_label = sorted(glob.glob(labels_search_str))
        numLabels = len(self._datalist_label)
        if numLabels == 0:
            raise ValueError('No input label files found in given directory. Searched in dir: {} '.format(self.labels_dir))
        if self.masks_dir is not None:
            masks_search_str = os.path.join(self.masks_dir, '*.png')
            self._datalist_mask = sorted(glob.glob(masks_search_str))
            numMasks = len(self._datalist_mask)
            if numMasks == 0:
                raise ValueError('No input mask files found in given directory. Searched in dir: {} '.format(self.masks_dir))
        if not numDoLP == numNAoLP == num_s0 == num_i0 == num_i45 == num_i90 == num_i135 == numNorm_0 == numNorm_1 == numNorm_2 == numNorm_3 == numNorm_4 == numNorm_5 == numLabels == numMasks == num_c:
            print(numDoLP, numNAoLP, num_s0, num_i0, num_i45, num_i90, num_i135, numNorm_0, numNorm_1, numNorm_2, numNorm_3, numNorm_4, numNorm_5, numMasks, numLabels, num_c)
            raise ValueError('Numbers of inputs(rgb,normal,label,mask) are different.')

    def _activator_masks(self, images, augmenter, parents, default):
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import torchvision
    imsize = 512
    augs_train = iaa.Sequential([iaa.Resize((imsize, imsize))])
    augs_test = iaa.Sequential([iaa.Resize((imsize, imsize), 0)])
    augs = augs_train
    input_only = ['gaus-blur', 'grayscale', 'gaus-noise', 'brightness', 'contrast', 'hue-sat', 'color-jitter']
    import loss_functions
    input_tensor, label_tensor, mask_tensor = dt_train.__getitem__(11)
    input_img_arr = input_tensor.numpy()
    label_img = label_tensor.numpy()
    mask_img = mask_tensor.numpy()
    fig = plt.figure()
    ax0 = plt.subplot(241)
    ax0.imshow(label_img.transpose(1, 2, 0))
    ax1 = plt.subplot(245)
    ax1.imshow(input_img_arr[1:4, :, :].transpose(1, 2, 0))
    ax2 = plt.subplot(246)
    ax2.imshow(input_img_arr[4:7, :, :].transpose(1, 2, 0))
    ax3 = plt.subplot(247)
    ax3.imshow(input_img_arr[7:10, :, :].transpose(1, 2, 0))
    ax4 = plt.subplot(248)
    ax4.imshow(input_img_arr[10:13, :, :].transpose(1, 2, 0))
    ax5 = plt.subplot(242)
    ax5.imshow(input_img_arr[0, :, :])
    ax6 = plt.subplot(243)
    ax6.imshow(mask_img.squeeze(0))
    print('mask_valid:_nums:', len(np.where(mask_img > 0)[1]))
    print(mask_img.shape)
    print('input_vec:', input_tensor[4:7, :, :].unsqueeze(0).shape)
    print('target_vec:', label_tensor.unsqueeze(0).shape)
    print('mask_vec:', mask_tensor.unsqueeze(0).shape)
    print('loss', loss_functions.loss_fn_cosine(input_vec=input_tensor[1:4, :, :].unsqueeze(0), target_vec=label_tensor.unsqueeze(0), mask_tensor=mask_tensor.unsqueeze(0).squeeze(1), reduction='elementwise_mean'))
    loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(input_tensor[10:13, :, :].unsqueeze(0), label_tensor.double().unsqueeze(0))
    print('loss_deg_mean:', loss_deg_mean)
    print('loss_deg_median:', loss_deg_median)
    print('percentage_1:', percentage_1)
    print('percentage_2:', percentage_2)
    print('percentage_3:', percentage_3)
    plt.show()
