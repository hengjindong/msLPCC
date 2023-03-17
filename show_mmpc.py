import collections
import open3d as o3d
import open3d
import os
import pdb
from tqdm import tqdm
import numpy as np
import torch
import pytorch3d
from dataset import *
from Chamfer3D.loss_utils import chamfer, chamfer_sqrt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.pc_error_wrapper import pc_error
import time
import importlib
import sys
import argparse
import shutil
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# from models.pointnet_util import farthest_point_sample, index_points, square_distance

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-   1)))
    return res.reshape(*raw_size, -1)

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def my_group_gnn(pc_data, key_num, k_set):
    pc_data = torch.Tensor(pc_data).cuda()
    pc_sample_data = index_points(pc_data, farthest_point_sample(pc_data, key_num))
    dist, group_idx, grouped_xyz = pytorch3d.ops.knn.knn_points(pc_sample_data, pc_data, K=k_set, return_nn=True)
    return grouped_xyz

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

def save_pc(points, save_path):
    points = points.cpu().detach().numpy()
    ori_pcd = open3d.geometry.PointCloud() # 定义点云
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(points)) # 定义点云坐标位置[N,3]
    open3d.io.write_point_cloud(save_path, ori_pcd, write_ascii=True)

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
    # print(orifile)
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

def save_dataset_knn_fps(dataset_path, shape_num, layer_num):
    VAL_DATASET = Dataset(root = dataset_path, dataset_name = '11', num_points = 2048, split='test')
    test_loader = DataLoader(VAL_DATASET, num_workers=2, batch_size=1, shuffle=False)
    for step, data in tqdm(enumerate(test_loader, 0), total=(len(test_loader)), smoothing=0.9):
        pc_data = data[0]
        pc_data_raw = data[1]
        pc_gt_raw = pc_data_raw.cuda()
        save_pc(pc_data_raw[0], './show_mm/ori.ply')

        pc_knn = my_group_gnn(pc_data_raw, shape_num, layer_num)

        print('===================================')
        for layer_idx in range(layer_num):
            pc_data_ipx = pc_knn[:,:,layer_idx,:]
            save_layer_path = './show_mm/knn_'+str(layer_idx)+'.ply'
            save_pc(pc_data_ipx[0], save_layer_path)

            cd = chamfer_sqrt(pc_gt_raw, pc_data_ipx)
            cd = cd*100
            print('knn layer:', layer_idx)
            print(cd)

        print('===================================')
        for layer_idx in range(layer_num):
            start = shape_num * layer_idx
            end = (layer_idx + 1) * shape_num
            pc_data_ipx = pc_data[:1, start:end, :]
            save_layer_path = './show_mm/fps_'+str(layer_idx)+'.ply'
            save_pc(pc_data_ipx[0], save_layer_path)

            pc_data_ipx = pc_data_ipx.cuda()
            cd = chamfer_sqrt(pc_gt_raw, pc_data_ipx)
            cd = cd*100
            print('fps layer:', layer_idx)
            print(cd)

        print('===================================')
        total_output_pc = pc_knn[:,:,0,:]
        for layer_idx in range(1, layer_num):
            pc_data_ipx = pc_knn[:,:,layer_idx,:]
            save_layer_path = './show_mm/knn_'+str(layer_idx)+'_sum.ply'
            total_output_pc = torch.cat((total_output_pc, pc_data_ipx), 1)
            save_pc(total_output_pc[0], save_layer_path)

            cd = chamfer_sqrt(pc_gt_raw, total_output_pc)
            cd = cd*100
            print('knn sum layer:', layer_idx)
            print(cd)

        print('===================================')
        total_output_pc = pc_data[:1, 0:2048, :]
        for layer_idx in range(1, layer_num):
            start = shape_num * layer_idx
            end = (layer_idx + 1) * shape_num
            pc_data_ipx = pc_data[:1, start:end, :]
            save_layer_path = './show_mm/fps_'+str(layer_idx)+'_sum.ply'
            total_output_pc = torch.cat((total_output_pc, pc_data_ipx), 1)
            save_pc(total_output_pc[0], save_layer_path)

            temp_pc = total_output_pc.cuda()
            cd = chamfer_sqrt(pc_gt_raw, temp_pc)
            cd = cd*100
            print('fps sum layer:', layer_idx)
            print(cd)

        pdb.set_trace()

def log_knn_time_cost(dataset_path, shape_num, layer_num):
    VAL_DATASET = Dataset(root = dataset_path, dataset_name = '*', num_points = 2048, split='test')
    test_loader = DataLoader(VAL_DATASET, num_workers=2, batch_size=1, shuffle=False)
    time_sum = 0.0
    for step, data in tqdm(enumerate(test_loader, 0), total=(len(test_loader)), smoothing=0.9):
        pc_data = data[0]
        pc_data_raw = data[1]
        pc_gt_raw = pc_data_raw.cuda()

        start_time = time.time()
        pc_knn = my_group_gnn(pc_data_raw, shape_num, layer_num)
        end_time = time.time()
        cost_time = end_time - start_time
        print(cost_time)
        time_sum += cost_time

    avg_time = time_sum / (len(test_loader))
    print('total in shape num: ', shape_num, ', layer_num:', layer_num, ',avg knn cost time:', avg_time)

def log_fps_time_cost(dataset_path, fps_num):
    VAL_DATASET = Dataset(root = dataset_path, dataset_name = '*', num_points = 2048, split='test')
    test_loader = DataLoader(VAL_DATASET, num_workers=2, batch_size=1, shuffle=False)
    time_sum = 0.0
    for step, data in tqdm(enumerate(test_loader, 0), total=(len(test_loader)), smoothing=0.9):
        pc_data = data[0]
        pc_data_raw = data[1]
        pc_gt_raw = pc_data_raw.cuda()

        start_time = time.time()
        pc_data = torch.Tensor(pc_data).cuda()
        pc_sample_data = index_points(pc_data, farthest_point_sample(pc_data, fps_num))
        end_time = time.time()
        cost_time = end_time - start_time
        print(cost_time)
        time_sum += cost_time

    avg_time = time_sum / (len(test_loader))
    print('total in downsampling num: ', fps_num, ',avg knn cost time:', avg_time)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('point_based_PCGC')
    parser.add_argument('--dataset_path', type=str, default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    parser.add_argument('--shape_num', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--layer_num', type=int, default=32, help='Point Number [default: 2048]')
    parser.add_argument('--fps_num', type=int, default=2048, help='Point Number [default: 2048]')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    shape_num = args.shape_num
    layer_num = args.layer_num
    fps_num = args.fps_num
    # log_knn_time_cost(args.dataset_path, shape_num, layer_num)
    # log_fps_time_cost(args.dataset_path, fps_num)
    save_dataset_knn_fps(args.dataset_path, shape_num, layer_num)

