import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from termcolor import colored
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from get_dataloader import get_dataloader
from modeling.Icesfp.Icesfp import Icesfp
import loss_functions
import API.utils
import numpy as np
import random
from tensorboardX import SummaryWriter
import argparse


def evaluation(model, testLoader, device, criterion, epoch, resultPath=None, name=None, writer=None, save_images=False):
    model.eval()
    running_loss = 0.0
    running_mean = 0
    running_median = 0
    running_percentage_1 = 0
    running_percentage_2 = 0
    running_percentage_3 = 0
    mean_list = []
    for iter_num, sample_batched in enumerate(tqdm(testLoader)):
        params_t, normals_t, label_t, mask_t, confidence_map_init = sample_batched
        params_t = params_t.to(device)
        normals_t = normals_t.to(device)
        label_t = label_t.to(device)
        confidence_map_init = confidence_map_init.to(device)
        aolp = params_t[:, 1, :, :]
        with torch.no_grad():
            normal_vectors, confidence_map = model(params_t, normals_t, confidence_map_init)
        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        normal_vectors_norm = normal_vectors_norm
        loss = criterion(normal_vectors_norm, label_t.double(), confidence_map, aolp, mask_tensor=mask_t, reduction='sum', device=device)
        label_t = label_t.detach().cpu()
        normal_vectors_norm = normal_vectors_norm.detach().cpu()
        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(normal_vectors_norm, label_t.double())
        running_mean += loss_deg_mean.item()
        running_median += loss_deg_median.item()
        mean_list.append(loss_deg_mean.item())
        running_loss += loss.item()
        running_percentage_1 += percentage_1.item()
        running_percentage_2 += percentage_2.item()
        running_percentage_3 += percentage_3.item()
        label_t_rgb = label_t.numpy().squeeze(0).transpose(1, 2, 0)
        label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
        predict_norm = normal_vectors_norm.numpy().squeeze(0).transpose(1, 2, 0)
        mask_t = mask_t.squeeze(1)
        predict_norm[mask_t.squeeze(0) == 0, :] = -1
        predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
        confidence_map_rgb = confidence_map.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
        confidence_map_rgb = confidence_map_rgb * 255
        confidence_map_rgb = confidence_map_rgb.astype(np.uint8)
        if save_images:
            if not os.path.exists(resultPath):
                os.makedirs(resultPath, exist_ok=True)
            API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-label.png'), label_t_rgb)
            API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-predict.png'), predict_norm_rgb)
            API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-atten.png'), confidence_map_rgb)
    assert testLoader.batch_size == 1, 'testLoader batch size is need to be 1 instead of : "%d"' % testLoader.batch_size
    numsamples = len(testLoader)
    running_loss = running_loss / numsamples
    running_mean = running_mean / numsamples
    running_median = running_median / numsamples
    running_percentage_1 = running_percentage_1 / numsamples
    running_percentage_2 = running_percentage_2 / numsamples
    running_percentage_3 = running_percentage_3 / numsamples
    if writer is not None:
        writer.add_scalar(name + '/' + 'running_mean', running_mean, epoch)
        writer.add_scalar(name + '/' + 'running_median', running_median, epoch)
        writer.add_scalar(name + '/' + 'running_loss', running_loss, epoch)
        writer.add_scalar(name + '/' + 'running_percentage_1', running_percentage_1, epoch)
        writer.add_scalar(name + '/' + 'running_percentage_2', running_percentage_2, epoch)
        writer.add_scalar(name + '/' + 'running_percentage_3', running_percentage_3, epoch)
    print('mean list:', mean_list)
    return (running_loss, running_mean, running_median, running_percentage_1, running_percentage_2, running_percentage_3)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)
parser = argparse.ArgumentParser(description='Icesfp')
parser.add_argument('-dataset_dir', default='data', help='path for the Icesfp dataset root')
parser.add_argument('-code_dir', default='.', help='path for checkpoints, logs, and result outputs')
parser.add_argument('-batch_size', default=5, type=int, help='batch size')
parser.add_argument('-checkpoint', default='', help='path for checkpoint')
imgHeight = 512
imgWidth = 512
batch_size = 5
num_workers = 2
validation_split = 0.1
shuffle_dataset = True
pin_memory = False
prefetch_factor = 8
augs_train = iaa.Sequential([iaa.Resize({'height': imgHeight, 'width': imgWidth}, interpolation='nearest')])
parsed = parser.parse_args()
root_dir = parsed.dataset_dir
code_root_dir = parsed.code_dir
batch_size = int(parsed.batch_size)
trainLoader, testLoader_dataset_iceapple, testLoader_dataset_icebird2, testLoader_dataset_iceMouse, testLoader_dataset_iceHemisphere, testLoader_dataset_iceRabbit2 = get_dataloader(root_dir, augs_train, batch_size, num_workers, pin_memory)
backbone_model = 'resnet50'
sync_bn = False
numClasses = 3
use_atten = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Icesfp(backbone=backbone_model, num_classes=3, device=device, sync_bn=False)
model = model.to(device)
learningRate = 1e-06
weightDecay = 0.0005
momentum = 0.9
lrScheduler = 'StepLR'
step_size = 7
gamma = 0.1
factor: 0.8
patience: 25
verbose: True
optimizer = torch.optim.SGD(model.parameters(), lr=float(learningRate), momentum=float(momentum), weight_decay=float(weightDecay))
if not lrScheduler:
    pass
elif lrScheduler == 'StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=float(gamma))
elif lrScheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=float(factor), patience=patience, verbose=verbose)
elif lrScheduler == 'lr_poly':
    pass
else:
    raise ValueError("Invalid Scheduler from config file: '{}'. Valid values are ['', 'StepLR', 'ReduceLROnPlateau']".format(lrScheduler))
criterion = loss_functions.Icesfp_loss
writer = SummaryWriter()
MAX_EPOCH = 35
saveModelInterval = 1
CHECKPOINT_DIR = code_root_dir + '/CheckPoints'
total_iter_num = 0
START_EPOCH = 0
continue_train = True
preCheckPoint = os.path.join(CHECKPOINT_DIR, 'best-model.pth')
if continue_train:
    print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
    if not os.path.isfile(preCheckPoint):
        raise ValueError('Invalid path to the given weights file for transfer learning.                The file {} does not exist'.format(preCheckPoint))
    CHECKPOINT = torch.load(preCheckPoint, map_location='cpu')
    if 'model_state_dict' in CHECKPOINT:
        model.load_state_dict(CHECKPOINT['model_state_dict'])
    elif 'state_dict' in CHECKPOINT:
        CHECKPOINT['state_dict'].pop('decoder.last_conv.8.weight')
        CHECKPOINT['state_dict'].pop('decoder.last_conv.8.bias')
        model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
    else:
        model.load_state_dict(CHECKPOINT)
    if continue_train and preCheckPoint:
        if 'optimizer_state_dict' in CHECKPOINT:
            optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
        else:
            print(colored('WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' + '"optimizer_state_dict". Continuing without loading optimizer state. ', 'red'))
    if continue_train and preCheckPoint:
        if 'model_state_dict' in CHECKPOINT:
            total_iter_num = CHECKPOINT['total_iter_num'] + 1
            START_EPOCH = CHECKPOINT['epoch'] + 1
            END_EPOCH = CHECKPOINT['epoch'] + MAX_EPOCH
        else:
            print(colored('Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.                           Starting from epoch num 0', 'red'))
if __name__ == '__main__':
    print('trainLoader size:', len(trainLoader))
    print('Continuing training from checkpoint...Loaded data from checkpoint:')
    results_root = os.path.join(code_root_dir, 'results')
    os.makedirs(results_root, exist_ok=True)
    import time
    mean_list = []
    median_list = []
    for epoch in range(START_EPOCH, START_EPOCH + 1):
        print('\n\nEpoch {}/{}'.format(epoch, MAX_EPOCH - 1))
        print('-' * 30)
        mean_all = 0.0
        median_all = 0.0
        acc_all_1 = 0.0
        acc_all_2 = 0.0
        acc_all_3 = 0.0
        count = 0
        print('\nValidation:')
        print('=' * 10)
        test_loaders = {'iceapple': testLoader_dataset_iceapple, 'icebird2': testLoader_dataset_icebird2, 'iceMouse': testLoader_dataset_iceMouse, 'iceHemisphere': testLoader_dataset_iceHemisphere, 'iceRabbit2': testLoader_dataset_iceRabbit2}
        for name, testLoader in test_loaders.items():
            running_loss, running_mean, running_median, running_percentage_1, running_percentage_2, running_percentage_3 = evaluation(model=model, testLoader=testLoader, device=device, criterion=criterion, epoch=epoch, name=name, writer=writer, resultPath=os.path.join(results_root, name), save_images=True)
            print(f'{name}:')
            print('loss: ', running_loss)
            print('mean: ', running_mean)
            print('median: ', running_median)
            print('percentage_1: ', running_percentage_1)
            print('percentage_2: ', running_percentage_2)
            print('percentage_3: ', running_percentage_3)
            acc_all_1 += running_percentage_1
            acc_all_2 += running_percentage_2
            acc_all_3 += running_percentage_3
            mean_all += running_mean
            median_all += running_median
            count += 1
            print('=' * 10)
            print('\n')
        if count > 0:
            print('all mean: ', mean_all / count)
            print('all median: ', median_all / count)
            print('percentage 1: ', acc_all_1 / count)
            print('percentage 2: ', acc_all_2 / count)
            print('percentage 3: ', acc_all_3 / count)
            mean_list.append(mean_all / count)
            median_list.append(median_all / count)
        else:
            print('No loaders to evaluate (count==0).')
    print('mean_all_list:', mean_list)
    print('median_all_list:', median_list)
