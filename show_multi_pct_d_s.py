import collections
import open3d
import os
import pdb
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from dataset import MMDataset_from_bin
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.pc_error_wrapper import pc_error
from Chamfer3D.loss_utils import chamfer, chamfer_sqrt
from models.bitEstimator import BitEstimator
import time
import importlib
import sys
import math
import argparse
import open3d as o3d
from models.pct_pcr import *
from models.PC2MD import *
from models.PC2Dsem import *
from models.PC2D import *
from models.PC2MSem import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def show_pc(points):
    if (points.dtype == torch.float32):
        points = points.cpu().detach().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd], window_name='test')
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd], window_name='test')
    return pcd

class cal_bpp(nn.Module):
    def __init__(self):
        super(cal_bpp, self).__init__()
        self.be = BitEstimator(2048)
    def forward(self, latent, num_points):
        z = torch.round(latent)
        prob = self.be(z + 0.5) - self.be(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
        bpp = total_bits / num_points
        return bpp

def cal_d1_ds(pc_ds, pc_raw):
    pc_error_metrics = pc_error(infile1=pc_raw, infile2=pc_ds, res=2) # res为数据峰谷差值
    pc_errors = [pc_error_metrics["mse1,PSNR (p2point)"][0],
                pc_error_metrics["mse2,PSNR (p2point)"][0],
                pc_error_metrics["mseF,PSNR (p2point)"][0],
                pc_error_metrics["mse1      (p2point)"][0],
                pc_error_metrics["mse2      (p2point)"][0],
                pc_error_metrics["mseF      (p2point)"][0]]
    return pc_errors

def save_pc_to_ply(points, this_pc_idx, this_pc_layer, checkpoint_path):
    if (points.dtype == torch.float32):
        points = points.cpu().detach().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.squeeze(points))
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.squeeze(points))
    experiment_dir = Path(checkpoint_path, str(this_pc_idx))
    experiment_dir.mkdir(exist_ok=True)
    save_ply_file = checkpoint_path+'/'+str(this_pc_idx)+'/layer_'+this_pc_layer+'.ply'
    open3d.io.write_point_cloud(save_ply_file, pcd, write_ascii=True)

def cal_d1(pc_gt, decoder_output, step, checkpoint_path):
    ori_pcd = open3d.geometry.PointCloud() # 定义点云
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(pc_gt)) # 定义点云坐标位置[N,3]
    orifile = checkpoint_path+'/pc_file/'+'d1_ori_'+str(step)+'.ply'# 保存路径
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
    # 重建点云写入ply文件
    rec_pcd = open3d.geometry.PointCloud()
    rec_pcd.points = open3d.utility.Vector3dVector(np.squeeze(decoder_output))
    recfile = checkpoint_path+'/pc_file/'+'d1_rec_'+str(step)+'.ply'
    open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

    pc_error_metrics = pc_error(infile1=orifile, infile2=recfile, res=2) # res为数据峰谷差值
    pc_errors = [pc_error_metrics["mse1,PSNR (p2point)"][0],
                pc_error_metrics["mse2,PSNR (p2point)"][0],
                pc_error_metrics["mseF,PSNR (p2point)"][0],
                pc_error_metrics["mse1      (p2point)"][0],
                pc_error_metrics["mse2      (p2point)"][0],
                pc_error_metrics["mseF      (p2point)"][0]]
    return pc_errors

def cal_d2(pc_gt, decoder_output, step, checkpoint_path):
    ori_pcd = open3d.geometry.PointCloud() # 定义点云
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(pc_gt)) # 定义点云坐标位置[N,3]
    ori_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30)) # 计算normal
    orifile = checkpoint_path+'/pc_file/'+'d2_ori_'+str(step)+'.ply'# 保存路径
    print(orifile)
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
    # 将ply文件中normal类型double转为float32
    lines = open(orifile).readlines()
    to_be_modified = [7, 8, 9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double','float32')
    file = open(orifile, 'w')
    for line in lines:
        file.write(line)
    file.close()
    # 可视化点云,only xyz
    # open3d.visualization.draw_geometries([ori_pcd])

    # 重建点云写入ply文件
    rec_pcd = open3d.geometry.PointCloud()
    rec_pcd.points = open3d.utility.Vector3dVector(np.squeeze(decoder_output))
    recfile = checkpoint_path+'/pc_file/'+'d2_rec_'+str(step)+'.ply'
    open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

    pc_error_metrics = pc_error(infile1=orifile, infile2=recfile, normal=True, res=2) # res为数据峰谷差值,normal=True为d2
    pc_errors = [pc_error_metrics["mse1,PSNR (p2plane)"][0],
                pc_error_metrics["mse2,PSNR (p2plane)"][0],
                pc_error_metrics["mseF,PSNR (p2plane)"][0],
                pc_error_metrics["mse1      (p2plane)"][0],
                pc_error_metrics["mse1      (p2plane)"][0],
                pc_error_metrics["mse1      (p2plane)"][0],
                pc_error_metrics["mse2      (p2plane)"][0],
                pc_error_metrics["mseF      (p2plane)"][0]]

    return pc_errors

def test(pc_model, sem_enc_model, depth_enc_model, args, batch_size=1):
    experiment_dir = Path('.', 'show_pc_file')
    experiment_dir.mkdir(exist_ok=True)
    checkpoint_path = './show_pc_file'
    experiment_dir = Path(checkpoint_path, str(args.output_number))
    experiment_dir.mkdir(exist_ok=True)
    checkpoint_path = './show_pc_file/' + str(args.output_number)

    print(args.seq_name)
    print("=============loading dataset!=============")
    VAL_DATASET = MMDataset_from_bin(root=args.dataset_path,
                          dataset_name = args.seq_name,
                          num_points=args.num_point,
                          split='test')
    test_loader = DataLoader(VAL_DATASET, num_workers=2, batch_size=batch_size, shuffle=False)

    print("============start test!==============")
    for step, data in tqdm(enumerate(test_loader), total=(len(test_loader)), smoothing=0.9):
        with torch.no_grad():

            this_pc_idx = step
            logger = logging.getLogger("Model")
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler('%s/%s.txt' % (experiment_dir, str(this_pc_idx)))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(args)

            pc_data = data[0]
            pc_gt_raw = data[1]
            pc_test_gt_raw = data[1]
            pc_gt_raw = pc_gt_raw.cuda()

            total_output_pc = data[0]
            total_bpp = 0.0
            save_pc_to_ply(pc_gt_raw, this_pc_idx, "ori", checkpoint_path)

            depth_enc_model.eval()
            sem_enc_model.eval()
            pc_model.eval()

            for level_ipx in tqdm(range(args.level_max)):

                # create layer point cloud
                start = args.input_number * level_ipx
                end = (level_ipx + 1) * args.input_number
                pc_data_ipx = pc_data[:, start:end, :]
                pc_data_ipx = pc_data_ipx.cuda()
                pc_data_rep = pc_data_ipx.repeat(4, 1, 1)

                # get depth&sem feature
                ft_depth = depth_enc_model(pc_data_rep)
                ft_depth = depth_enc_model(pc_data_rep)
                ft_depth = ft_depth[0].unsqueeze(0)
                ft_sem = sem_enc_model(pc_data_rep)
                ft_sem = ft_sem[0].unsqueeze(0)
                ft_d_s = torch.cat((ft_sem, ft_depth), dim = 1)

                # get this layer rec point cloud
                bpp, recon_xyz, cd_ds, cd_raw = pc_model(ft_d_s, pc_data_ipx, pc_gt_raw)

                # merger rec point cloud
                if level_ipx == 0:
                    total_output_pc = recon_xyz
                else:
                    total_output_pc = torch.cat((total_output_pc, recon_xyz), 1)
                total_bpp = total_bpp + bpp

                # save merger rec point cloud
                save_pc_to_ply(total_output_pc, this_pc_idx, str(level_ipx + 1), checkpoint_path)

                # cal merger rec point cloud quailty
                temp_level_ipx = level_ipx + 1
                logger.info('====this level %d' % temp_level_ipx)
                logger.info(total_output_pc.shape)
                decoder_output = total_output_pc
                cd = chamfer_sqrt(pc_gt_raw, decoder_output)
                decoder_output = decoder_output[0].cpu().detach().numpy()
                d2_results = cal_d1(pc_test_gt_raw, decoder_output, step, '.')
                d2_psnr = d2_results[2].item()
                d2_mse = d2_results[5].item()
                logger.info('bpp: %f' % total_bpp)
                logger.info('cd: %f' % cd)
                logger.info('psnr: %f' % d2_psnr)

def parse_args():
    parser = argparse.ArgumentParser('point_based_PCGC')
    parser.add_argument('--dataset_path', type=str, default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    parser.add_argument('--seq_name', type=str, default='*')
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--output_number', type=int, default=2048)
    parser.add_argument('--num_point', type=int, default=2048)
    parser.add_argument('--input_number', type=int, default=2048)
    parser.add_argument('--level_max', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    exp_name = '10000_512_2048'
    model_name = 'PCT_D_S_PCC'
    experiment_dir = './log/'+model_name+'/'+exp_name
    MODELA = importlib.import_module(model_name)
    set_in = exp_name.split('_')
    args.output_number = int(set_int[2])
    pc_model = MODELA.get_model(latent_size=int(set_in[1]), recon_points=int(set_in[2])).cuda()
    pc_model.eval()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v
    pc_model.load_state_dict(new_state_dict)
    print("==============pc_coder loaded!==============")

    sem_enc_model = pc2msem(latent_size=args.latent_size, recon_points=30000).cuda()
    checkpoint = torch.load('/home/install/pcc_point_based/MLPCC/log/PC2MSem/10002_256_2048/checkpoints/best_model.pth')
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    sem_enc_model.load_state_dict(new_state_dict)
    print("==============sem_enc loaded!==============")

    depth_enc_model = pc2d(latent_size=args.latent_size, recon_points=30000).cuda()
    checkpoint = torch.load('/home/install/pcc_point_based/MLPCC/log/PC2D/10002_256_2048/checkpoints/best_model.pth')
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    depth_enc_model.load_state_dict(new_state_dict)
    print("==============depth_enc loaded!==============")

    test(pc_model, sem_enc_model, depth_enc_model, args)
