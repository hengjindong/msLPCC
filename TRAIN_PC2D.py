import argparse
import os
import pdb
import torch
import collections
import torchvision
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np

from torch import nn
from tqdm import tqdm
from test import cal_d1
from pathlib import Path
from tabnanny import check
from dataset import MMDataset_from_bin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('point_based_PCGC')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 32]')
    parser.add_argument('--model', default='PC2D_PCT', help='model name')
    parser.add_argument('--epoch',  default=210, type=int, help='number of epoch in training [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--lamb', default=10000, type=int)
    parser.add_argument('--bottleneck_size', default=256, type=int)
    parser.add_argument('--latent_size', default=256, type=int)
    parser.add_argument('--use_hyper', default=True, type=bool)
    parser.add_argument('--recon_points', default=2048, type=int)
    parser.add_argument('--cpu', default=False, type=bool)
    parser.add_argument('--multigpu', default=False, type=bool)
    parser.add_argument('--pretrained', default='', type=str)
    # parser.add_argument('--pretrained', default='./log/PCT_PCC2/10000_512_10000/checkpoints/320.pth', type=str)
    parser.add_argument('--dataset_path', type=str, default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    return parser.parse_args()

def test(args, model, loader, criterion, global_epoch=None):
    mean_loss = []
    mean_bpp = []
    mean_cd = []
    with torch.no_grad():
        for j, data in tqdm(enumerate(loader, 0), total=(len(loader)),smoothing=0.9):
            points = data[0]
            points = points[:, 0:2048, :]
            points_raw = data[1]
            depth = data[2]
            model.eval()
            points = points.cuda()
            depth = depth.cuda()
            depth = depth.unsqueeze(1)
            points_raw = points_raw.cuda()
            rec_depth = model(points)
            loss = criterion(rec_depth, depth)
            mean_cd.append(loss.mean().item())
            mean_loss.append(loss.mean().item())
            mean_bpp.append(loss.mean().item())
    return np.mean(mean_loss), np.mean(mean_bpp), np.mean(mean_cd)

def main(args):
    def log_string(str):
        logger.info(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_name = str(args.lamb)+'_'+str(args.bottleneck_size)+'_'+str(args.recon_points)
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.model)
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(experiment_name)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    TRAIN_DATASET = MMDataset_from_bin(root=args.dataset_path,
                            dataset_name = '*',
                            num_points=args.num_point,
                            # split='train_my')
                            split='train')
    VAL_DATASET = MMDataset_from_bin(root=args.dataset_path,
                          dataset_name = '*',
                          num_points=args.num_point,
                          split='test')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    device = torch.device("cuda:0")
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/utils.py', str(experiment_dir))

    if not args.cpu:
        model = MODEL.get_model(latent_size=args.latent_size, recon_points=args.recon_points).cuda()
        criterion = MODEL.get_loss(lam=args.lamb).cuda()
        if args.multigpu:
            print('multiple gpu used')
            model = nn.DataParallel(model)
    else:
        model = MODEL.get_model(use_hyperprior=False, bottleneck_size=args.bottleneck_size, recon_points=args.recon_points)
        criterion = MODEL.get_loss(lam=args.lamb)

    '''finetune'''
    try:
        checkpoint = torch.load(args.pretrained)
        start_epoch = 200
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Finetuning')
    except:
        log_string('No pretrained model')


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )

    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        log_string('Use pretrain optimizer')
    try:
        assert len(args.pretrained_model) == 0
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(optimizer)
    except:
        log_string('No existing optimizer')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    if args.log_dir:
        pre_min_cd = checkpoint['loss']
    else:
        pre_min_cd = 9999999


    '''TRANING'''
    logger.info('Start training...')
    # for epoch in range(start_epoch,args.epoch):
    for epoch in tqdm(range(0,args.epoch)):
        mean_loss = []
        mean_bpp_loss = []
        mean_cd_loss = []
        cd_lam = 0.5
        if (epoch < 100) :
            cd_lam = 0.5
        elif (epoch >= 100):
            cd_lam = 0.1
        log_string('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, args.epoch))
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=(len(trainDataLoader)), smoothing=0.9):
            points = data[0]
            points = points[:, 0:2048, :]
            points_raw = data[1]
            depth = data[2]

            points = points.cuda()
            points_raw = points_raw.cuda()
            depth = depth.cuda()
            depth = depth.unsqueeze(1)
            optimizer.zero_grad()

            model.train()
            rec_depth = model(points)
            loss = criterion(rec_depth, depth)
            # loss = loss * 10000
            loss.backward()
            optimizer.step()
            mean_loss.append(loss.item())
            mean_bpp_loss.append(loss.item())
            mean_cd_loss.append(loss.item())
            global_step += 1
        ml = np.mean(mean_loss)
        mbpp = np.mean(mean_bpp_loss)
        mcd = np.mean(mean_cd_loss)
        log_string(checkpoints_dir)
        log_string('mean bpp: %f' % mbpp)
        log_string('mean chamfer distance: %f' % mcd)
        if epoch%10==0:
            log_string('Start val...')
            with torch.no_grad():
                mean_loss_test, mean_bpp_test, mean_cd_test = test(args, model.eval(), valDataLoader, criterion, global_step)
                log_string('val loss: %f'% (mean_loss_test))
                log_string('val bpp: %f' % (mean_bpp_test))
                log_string('val cd: %f' % (mean_cd_test))

            if epoch%100==0:
                savepath = str(checkpoints_dir) + '/'+str(epoch)+'.pth'
                state = {
                    'epoch': epoch,
                    'loss': ml,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            if (mean_cd_test < pre_min_cd and epoch >= 20):
                logger.info('Save model...')
                pre_min_cd = mean_loss_test
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': epoch,
                    'loss': mean_loss_test,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
