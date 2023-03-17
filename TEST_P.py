import collections
import open3d
import os
import pdb
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from dataset import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Chamfer3D.loss_utils import *
from utils.pc_error_wrapper import pc_error
from models.bitEstimator import BitEstimator
from Chamfer3D.loss_utils import chamfer, chamfer_sqrt, chamfer_multi
# from iostream import IOStream
import time
import importlib
import sys
import math
import argparse
import open3d as o3d
from pytorch3d.loss import chamfer_distance
sys.path.append('emd/')
from emd_module import emdModule
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
    # rec_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30)) # 计算normal
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

def test(model_1, args, test_loader):
    checkpoint_path = '.'
    my_cal_bpp = cal_bpp().cuda()

    total_chamfer_dist = 0.0
    total_d1_psnr, total_d1_mse = 0.0, 0.0
    total_d2_psnr, total_d2_mse = 0.0, 0.0
    total_bpp = 0.0
    num_samples = 0
    rmse_loss = 0.0
    cd_loss = 0.0
    # print("============start test!==============")
    for step, data in tqdm(enumerate(test_loader), total=(len(test_loader)), smoothing=0.9):
        with torch.no_grad():
            pc_data = data[0]
            pc_gt_raw = data[1]
            total_output_pc = data[0]
            total_output_bpp = 0.0
            for level_ipx in range(args.level):
                start = args.input_number * level_ipx
                end = (level_ipx + 1) * args.input_number
                pc_data_ipx = pc_data[:1, start:end, :]
                pc_data_ipx = pc_data_ipx.cuda()
                pc_gt_raw = pc_gt_raw.cuda()
                bpp_ipx, output_pc_ipx, cd = model_1(pc_data_ipx, pc_gt_raw)
                if level_ipx == 0:
                    total_output_pc = output_pc_ipx
                else:
                    total_output_pc = torch.cat((total_output_pc, output_pc_ipx), 1)
                total_output_bpp += bpp_ipx*3
            total_bpp += total_output_bpp

            decoder_output = total_output_pc

            decoder_output = decoder_output
            pc_gt_raw = pc_gt_raw
            cd, rmse = chamfer_multi(pc_gt_raw, decoder_output)
            cd_loss = cd_loss + cd * 1000000
            rmse_loss = rmse_loss + rmse * 10000

            decoder_output = decoder_output.cpu().detach().numpy()
            pc_gt_raw = pc_gt_raw.cpu().detach().numpy()
            d1_results = cal_d1(decoder_output, pc_gt_raw, step, checkpoint_path)
            d1_psnr = d1_results[2].item()
            d1_mse = d1_results[5].item()
            total_d1_mse += d1_mse
            total_d1_psnr += d1_psnr
        num_samples += 1
    total_chamfer_dist /= num_samples
    total_d1_psnr /= num_samples
    total_d1_mse /= num_samples
    total_bpp /= num_samples
    total_cd = cd_loss/num_samples
    total_rmse = rmse_loss/num_samples
    outstr = "Average_bpp: %.6f, Average_D1_PSNR: %.6f, Average_D1_mse: %.6f, Average_Chamfer_Dist: %.6f, Average_RMSE: %.6f\n" % (
             total_bpp,
             total_d1_psnr, total_d1_mse, total_cd, total_rmse)
    print(outstr)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('point_based_PCGC')
    parser.add_argument('--dataset_path', type=str, default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    parser.add_argument('--seq_name', type=str, default='*')
    parser.add_argument('--num_point', type=int, default=2500, help='Point Number [default: 2048]')
    parser.add_argument('--input_number', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--level', type=int, default=1, help='Point Number [default: 2048]')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    exp_name = '9001_256_2048'
    model_name = 'B_PCT_PCC'
    experiment_dir = './log/'+model_name+'/'+exp_name
    MODELA = importlib.import_module(model_name)

    set_in = exp_name.split('_')
    model_1 = MODELA.get_model(latent_size=int(set_in[1]), recon_points=int(set_in[2])).cuda()
    model_1.eval()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/30.pth')
    print(experiment_dir)

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v
    model_1.load_state_dict(new_state_dict)

    print("=============loading dataset!=============")
    VAL_DATASET = MMDataset_from_bin(root=args.dataset_path,
                          dataset_name = args.seq_name,
                          num_points=args.num_point,
                          split='test')

    test_loader = DataLoader(VAL_DATASET, num_workers=2, batch_size=1, shuffle=False)

    # list_level = [1, 4, 8, 12, 16, 20, 24, 28, 32]
    list_level = [4, 8, 12, 16, 20, 24, 28]
    # list_level = [1, 32]
    for i_level in list_level:
        args.level = i_level
        print('this test level is: ' + str(args.level))
        test(model_1, args, test_loader)
