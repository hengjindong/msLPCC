import argparse
from tabnanny import check
import numpy as np
import os
from tqdm import tqdm
import pdb
import torch
import collections
import torchvision
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from dataset import *
from torch import nn
import open3d as o3d
from test import cal_d1
from info_nce import *
from models.PC2D_PCT import pc2d_pct
from models.PC2D_EnB import pc2d_enb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('point_based_PCGC')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 32]')
    parser.add_argument('--model', default='PCT_PC2D_PCC_SR', help='model name')
    parser.add_argument('--epoch',  default=42, type=int, help='number of epoch in training [default: 100]')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--lamb', default=8005, type=int)
    parser.add_argument('--bottleneck_size', default=512, type=int)
    parser.add_argument('--latent_size', default=512, type=int)
    parser.add_argument('--use_hyper', default=True, type=bool)
    parser.add_argument('--recon_points', default=2048, type=int)
    parser.add_argument('--cpu', default=False, type=bool)
    parser.add_argument('--multigpu', default=False, type=bool)
    parser.add_argument('--pretrained', default='', type=str)
    parser.add_argument('--dataset_path', type=str, default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    return parser.parse_args()

def test(args, model, depth_enc_model, loader, criterion, global_epoch=None):
    mean_loss = []
    mean_bpp = []
    mean_cd = []
    with torch.no_grad():
        for j, data in tqdm(enumerate(loader, 0), total=(len(loader)),smoothing=0.9):
            points = data[0]
            points_raw = data[1]
            points = points[:, 0:2048, :]

            points = points.cuda()
            points_raw = points_raw.cuda()

            model.eval()

            with torch.no_grad():
                ft_depth = depth_enc_model(points)

            bpp, recon_xyz, cd_ds, cd_raw = model(ft_depth, points, points_raw)
            cd = 0.5*cd_ds + 0.5*cd_raw
            loss = criterion(cd, cd)

            mean_cd.append(cd_raw.mean().item())
            mean_loss.append(loss.mean().item())
            mean_bpp.append(bpp.mean().item())
    return np.mean(mean_loss), np.mean(mean_bpp), np.mean(mean_cd)

def show_pc(points):
    if (points.dtype == torch.float32):
        points = points.cpu().detach().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd], window_name='train_pc_pc2d')
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd], window_name='train_pc_pc2d')
    return pcd

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
                            split='train')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    VAL_DATASET = MMDataset_from_bin(root=args.dataset_path,
                          dataset_name = '*',
                          num_points=args.num_point,
                          split='test')
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

    # depth_enc_model = pc2d_pct(latent_size=args.latent_size, recon_points=30000).cuda()
    # checkpoint = torch.load('/home/install/pcc_point_based/MLPCC/log/PC2D_PCT/10000_256_2048/checkpoints/best_model.pth')
    depth_enc_model = pc2d_enb(latent_size=args.latent_size, recon_points=30000).cuda()
    checkpoint = torch.load('/home/install/pcc_point_based/MLPCC/log/PC2D_EnB/10000_256_2048/checkpoints/best_model.pth')
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    depth_enc_model.load_state_dict(new_state_dict)

    '''pretrain or train from scratch'''
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
        start_epoch = 100
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

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
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        log_string('Use pretrain optimizer')
    try:
        assert len(args.pretrained_model) == 0
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(optimizer)
    except:
        log_string('No existing optimizer')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    global_epoch = 0
    global_step = 0
    if args.log_dir:
        pre_min_cd = checkpoint['loss']
        mean_cd_test = checkpoint['loss']
    else:
        pre_min_cd = 9999999
        mean_cd_test = 9999999


    '''TRANING'''
    logger.info('Start training...')
    # for epoch in range(start_epoch,args.epoch):
    for epoch in tqdm(range(0,args.epoch)):
        mean_loss = []
        mean_bpp_loss = []
        mean_cd_loss = []
        cd_lam = 0.5
        # if epoch < 100:
            # cd_lam = 0.5
        # else:
            # cd_lam = 0.1
        log_string('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, args.epoch))
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=(len(trainDataLoader)), smoothing=0.9):
            points = data[0]
            points_raw = data[1]
            points = points[:, 0:2048, :]
            # points_raw = data[1]

            points = points.cuda()
            points_raw = points_raw.cuda()
            optimizer.zero_grad()

            model.train()

            with torch.no_grad():
                ft_depth = depth_enc_model(points)

            bpp, recon_xyz, cd_ds, cd_raw = model(ft_depth, points, points_raw)
            cd = cd_lam*cd_ds + (1-cd_lam)*cd_raw
            # co = co_lam * dim_depth_loss(latent_d, latent_s)
            loss = criterion(cd, cd) + 2 * bpp

            loss.backward()
            optimizer.step()
            mean_loss.append(cd_raw.item())
            mean_bpp_loss.append(bpp.item())
            mean_cd_loss.append(cd.item())
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
                mean_loss_test, mean_bpp_test, mean_cd_test = test(args, model.eval(), depth_enc_model, valDataLoader, criterion, global_step)
                log_string('val loss: %f'% (mean_loss_test))
                log_string('val bpp: %f' % (mean_bpp_test))
                log_string('val cd: %f' % (mean_cd_test))
                if (mean_cd_test < pre_min_cd and epoch >= 20):
                    logger.info('Save model...')
                    pre_min_cd = mean_cd_test
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
