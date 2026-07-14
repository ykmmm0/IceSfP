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
from datetime import datetime


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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Icesfp')
    parser.add_argument('-dataset_dir', default='data', help='path for the Icesfp dataset root')
    parser.add_argument('-code_dir', default='.', help='path for checkpoints, logs, and result outputs')
    parser.add_argument('-batch_size', default=5, type=int, help='batch size')
    parsed = parser.parse_args()
    root_dir = parsed.dataset_dir
    code_root_dir = parsed.code_dir
    batch_size = int(parsed.batch_size)
    setup_seed(20)
    imgHeight = 512
    imgWidth = 512
    batch_size = int(batch_size)
    num_workers = 2
    validation_split = 0.1
    shuffle_dataset = True
    pin_memory = False
    prefetch_factor = 8
    augs_train = iaa.Sequential([iaa.Resize({'height': imgHeight, 'width': imgWidth}, interpolation='nearest')])
    trainLoader, testLoader_dataset_iceapple, testLoader_dataset_icebird2, testLoader_dataset_iceMouse, testLoader_dataset_iceHemisphere, testLoader_dataset_iceRabbit2 = get_dataloader(root_dir, augs_train, batch_size, num_workers, pin_memory)
    backbone_model = 'resnet50'
    sync_bn = False
    numClasses = 3
    use_atten = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Icesfp(backbone=backbone_model, num_classes=3, device=device, sync_bn=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, betas=(0.9, 0.999), weight_decay=weightDecay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-06)
    criterion = loss_functions.Icesfp_loss
    writer = SummaryWriter()
    MAX_EPOCH =  ##
    saveModelInterval = ##
    CHECKPOINT_DIR = code_root_dir + '/CheckPoints'
    total_iter_num = 0
    START_EPOCH = 0
    continue_train = False
    preCheckPoint = os.path.join(CHECKPOINT_DIR, 'check-point-epoch-0000.pth')
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    if not os.path.exists(code_root_dir + '/results'):
        os.mkdir(code_root_dir + '/results')
    if continue_train:
        print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
        if not os.path.isfile(preCheckPoint):
            raise ValueError('Invalid path to the given weights file for transfer learning.                    The file {} does not exist'.format(preCheckPoint))
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
                print(colored('Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.                               Starting from epoch num 0', 'red'))
    best_mean = float('inf')
    import time
    mean_list = []
    median_list = []
    VALIDATE_START_EPOCH = 30
    VALIDATE_INTERVAL = 5
    SAVE_BEST_IMAGES = True
    for epoch in range(START_EPOCH, MAX_EPOCH):
        print('\n\nEpoch {}/{}'.format(epoch, MAX_EPOCH - 1))
        print('-' * 30)
        print('Train:')
        print('=' * 10)
        model.train()
        running_loss = 0.0
        running_mean = 0
        running_median = 0
        for iter_num, batch in enumerate(tqdm(trainLoader)):
            total_iter_num += 1
            params_t, normals_t, label_t, mask_t, confidence_map_init = batch
            params_t = params_t.to(device)
            aolp = params_t[:, 1, :, :]
            normals_t = normals_t.to(device)
            label_t = label_t.to(device)
            confidence_map_init = confidence_map_init.to(device)
            start = time.time()
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            with torch.autograd.set_detect_anomaly(True):
                normal_vectors, confidence_map = model(params_t, normals_t, confidence_map_init)
                normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
                normal_vectors_norm = normal_vectors_norm
                loss = criterion(normal_vectors_norm, label_t.double(), confidence_map, aolp, mask_tensor=mask_t, reduction='sum', device=device)
            loss /= batch_size
            loss.backward()
            optimizer.step()
            label_t = label_t.detach().cpu()
            normal_vectors_norm = normal_vectors_norm.detach().cpu()
            loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(normal_vectors_norm.detach().cpu(), label_t.double())
            running_mean += loss_deg_mean.item()
            running_median += loss_deg_median.item()
            running_loss += loss.item()
            if epoch % 10 == 0:
                label_t_rgb = label_t.numpy()[0, :, :, :].transpose(1, 2, 0)
                label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
                predict_norm = normal_vectors_norm.numpy()[0, :, :, :].transpose(1, 2, 0)
                mask_t = mask_t.squeeze(1)
                predict_norm[mask_t[0, :, :] == 0, :] = -1
                predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
                confidence_map = confidence_map[0, :, :, :]
                confidence_map_rgb = confidence_map.detach().cpu().numpy().transpose(1, 2, 0)
                confidence_map_rgb = confidence_map_rgb * 255
                confidence_map_rgb = confidence_map_rgb.astype(np.uint8)
                if not os.path.exists(code_root_dir + '/results/train'):
                    os.mkdir(code_root_dir + '/results/train')
                API.utils.png_saver(os.path.join(code_root_dir + '/results/train', str(iter_num).zfill(3) + '-label.png'), label_t_rgb)
                API.utils.png_saver(os.path.join(code_root_dir + '/results/train', str(iter_num).zfill(3) + '-predict.png'), predict_norm_rgb)
                API.utils.png_saver(os.path.join(code_root_dir + '/results/train', str(iter_num).zfill(3) + '-atten.png'), confidence_map_rgb)
        num_samples = len(trainLoader)
        epoch_loss = running_loss / num_samples
        print('train running loss:', epoch_loss)
        print('train running mean:', running_mean / num_samples)
        print('train running median:', running_median / num_samples)
        mean_all = 0
        median_all = 0
        acc_all_1 = 0
        acc_all_2 = 0
        acc_all_3 = 0
        count = 0
        print('\nValidation:')
        print('=' * 10)
        running_loss, running_mean, running_median, running_percentage_1, running_percentage_2, running_percentage_3 = evaluation(model=model, testLoader=testLoader_dataset_iceapple, device=device, criterion=criterion, epoch=epoch, name='iceapple', writer=writer, resultPath=code_root_dir + '/results/iceapple', save_images=False)
        print('iceapple:')
        print('loss: ', running_loss)
        print('mean: ', running_mean)
        print('median: ', running_median)
        print('percentage_1: ', running_percentage_1)
        print('percentage_2: ', running_percentage_2)
        print('percentage_3: ', running_percentage_3)
        acc_all_1 += running_percentage_1
        acc_all_2 += running_percentage_2
        acc_all_3 += running_percentage_3
        print('=' * 10)
        print('\n')
        mean_all += running_mean
        median_all += running_median
        count += 1
        running_loss, running_mean, running_median, running_percentage_1, running_percentage_2, running_percentage_3 = evaluation(model=model, testLoader=testLoader_dataset_icebird2, device=device, criterion=criterion, epoch=epoch, name='icebird2', writer=writer, resultPath=code_root_dir + '/results/icebird2', save_images=False)
        print('icebird2:')
        print('loss: ', running_loss)
        print('mean: ', running_mean)
        print('median: ', running_median)
        print('percentage_1: ', running_percentage_1)
        print('percentage_2: ', running_percentage_2)
        print('percentage_3: ', running_percentage_3)
        acc_all_1 += running_percentage_1
        acc_all_2 += running_percentage_2
        acc_all_3 += running_percentage_3
        print('=' * 10)
        print('\n')
        mean_all += running_mean
        median_all += running_median
        count += 1
        running_loss, running_mean, running_median, running_percentage_1, running_percentage_2, running_percentage_3 = evaluation(model=model, testLoader=testLoader_dataset_iceMouse, device=device, criterion=criterion, epoch=epoch, name='icemouse', writer=writer, resultPath=code_root_dir + '/results/icemouse', save_images=False)
        print('icemouse:')
        print('loss: ', running_loss)
        print('mean: ', running_mean)
        print('median: ', running_median)
        print('percentage_1: ', running_percentage_1)
        print('percentage_2: ', running_percentage_2)
        print('percentage_3: ', running_percentage_3)
        acc_all_1 += running_percentage_1
        acc_all_2 += running_percentage_2
        acc_all_3 += running_percentage_3
        print('=' * 10)
        print('\n')
        mean_all += running_mean
        median_all += running_median
        count += 1
        running_loss, running_mean, running_median, running_percentage_1, running_percentage_2, running_percentage_3 = evaluation(model=model, testLoader=testLoader_dataset_iceHemisphere, device=device, criterion=criterion, epoch=epoch, name='iceHemisphere', writer=writer, resultPath=code_root_dir + '/results/iceHemisphere', save_images=False)
        print('iceHemisphere:')
        print('loss: ', running_loss)
        print('mean: ', running_mean)
        print('median: ', running_median)
        print('percentage_1: ', running_percentage_1)
        print('percentage_2: ', running_percentage_2)
        print('percentage_3: ', running_percentage_3)
        acc_all_1 += running_percentage_1
        acc_all_2 += running_percentage_2
        acc_all_3 += running_percentage_3
        print('=' * 10)
        print('\n')
        mean_all += running_mean
        median_all += running_median
        count += 1
        running_loss, running_mean, running_median, running_percentage_1, running_percentage_2, running_percentage_3 = evaluation(model=model, testLoader=testLoader_dataset_iceRabbit2, device=device, criterion=criterion, epoch=epoch, name='iceRabbit2', writer=writer, resultPath=code_root_dir + '/results/iceRabbit2', save_images=False)
        print('iceRabbit2:')
        print('loss: ', running_loss)
        print('mean: ', running_mean)
        print('median: ', running_median)
        print('percentage_1: ', running_percentage_1)
        print('percentage_2: ', running_percentage_2)
        print('percentage_3: ', running_percentage_3)
        acc_all_1 += running_percentage_1
        acc_all_2 += running_percentage_2
        acc_all_3 += running_percentage_3
        print('=' * 10)
        print('\n')
        mean_all += running_mean
        median_all += running_median
        count += 1
        print('all mean: ', mean_all / count)
        print('all median: ', median_all / count)
        print('percentage 1: ', acc_all_1 / count)
        print('percentage 2: ', acc_all_2 / count)
        print('percentage 3: ', acc_all_3 / count)
        current_mean = mean_all / count
        if current_mean < best_mean:
            print(f'鉁?New best model found! mean: {current_mean:.6f} (prev: {best_mean:.6f})')
            best_mean = current_mean
            filename = os.path.join(CHECKPOINT_DIR, 'best-model.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_mean': best_mean, 'total_iter_num': total_iter_num}, filename)
            evaluation(model=model, testLoader=testLoader_dataset_iceapple, device=device, criterion=criterion, epoch=epoch, name='iceapple', writer=None, resultPath=os.path.join(code_root_dir, 'results', f'best_epoch_{epoch}', 'iceapple'), save_images=True)
            evaluation(model=model, testLoader=testLoader_dataset_icebird2, device=device, criterion=criterion, epoch=epoch, name='icebird2', writer=None, resultPath=os.path.join(code_root_dir, 'results', f'best_epoch_{epoch}', 'icebird2'), save_images=True)
            evaluation(model=model, testLoader=testLoader_dataset_iceMouse, device=device, criterion=criterion, epoch=epoch, name='icemouse', writer=None, resultPath=os.path.join(code_root_dir, 'results', f'best_epoch_{epoch}', 'icemouse'), save_images=True)
            evaluation(model=model, testLoader=testLoader_dataset_iceHemisphere, device=device, criterion=criterion, epoch=epoch, name='iceHemisphere', writer=None, resultPath=os.path.join(code_root_dir, 'results', f'best_epoch_{epoch}', 'iceHemisphere'), save_images=True)
            evaluation(model=model, testLoader=testLoader_dataset_iceRabbit2, device=device, criterion=criterion, epoch=epoch, name='iceRabbit2', writer=None, resultPath=os.path.join(code_root_dir, 'results', f'best_epoch_{epoch}', 'iceRabbit2'), save_images=True)
        mean_list.append(mean_all / count)
        median_list.append(median_all / count)
    print('mean_all_list:', mean_list)
    print('median_all_list:', median_list)
