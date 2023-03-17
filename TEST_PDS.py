import collections
import open3d
import os
import pdb
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from dataset import MMDataset_from_bin
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.pc_error_wrapper import pc_error
from Chamfer3D.loss_utils import chamfer, chamfer_sqrt, chamfer_multi
from models.bitEstimator import BitEstimator
import time
import importlib
import sys
import math
import argparse
import open3d as o3d
from models.PC2Dsem_PCT import pc2dsem_pct
from models.PC2Dsem_EnB import pc2dsem_enb
from models.PC2D_PCT import pc2d_pct
from models.PC2D_EnB import pc2d_enb
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
    pc_error_metrics = pc_error(infile1=pc_raw, infile2=pc_ds, res=2)
    pc_errors = [pc_error_metrics["mse1,PSNR (p2point)"][0],
                pc_error_metrics["mse2,PSNR (p2point)"][0],
                pc_error_metrics["mseF,PSNR (p2point)"][0],
                pc_error_metrics["mse1      (p2point)"][0],
                pc_error_metrics["mse2      (p2point)"][0],
                pc_error_metrics["mseF      (p2point)"][0]]
    return pc_errors

def cal_d1(pc_gt, decoder_output, step, checkpoint_path):
    ori_pcd = open3d.geometry.PointCloud()
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(pc_gt))
    orifile = checkpoint_path+'/pc_file/'+'d1_ori_'+str(step)+'.ply'
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
    rec_pcd = open3d.geometry.PointCloud()
    rec_pcd.points = open3d.utility.Vector3dVector(np.squeeze(decoder_output))
    recfile = checkpoint_path+'/pc_file/'+'d1_rec_'+str(step)+'.ply'
    open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

    pc_error_metrics = pc_error(infile1=orifile, infile2=recfile, res=2)
    pc_errors = [pc_error_metrics["mse1,PSNR (p2point)"][0],
                pc_error_metrics["mse2,PSNR (p2point)"][0],
                pc_error_metrics["mseF,PSNR (p2point)"][0],
                pc_error_metrics["mse1      (p2point)"][0],
                pc_error_metrics["mse2      (p2point)"][0],
                pc_error_metrics["mseF      (p2point)"][0]]
    return pc_errors

def test(model_1, args, batch_size=1):
    checkpoint_path = '.'
    print(args.seq_name)
    print("=============loading dataset!=============")
    VAL_DATASET = MMDataset_from_bin(root=args.dataset_path,
                          dataset_name = args.seq_name,
                          num_points=args.num_point,
                          split='test')
    test_loader = DataLoader(VAL_DATASET, num_workers=2, batch_size=batch_size, shuffle=False)

    enc_name = 'EnB'
    if enc_name == 'EnB':
        sem_enc_model = pc2dsem_enb(latent_size=args.latent_size, recon_points=30000).cuda()
        checkpoint = torch.load('/home/install/pcc_point_based/MLPCC/log/PC2Dsem_EnB/10002_256_2048/checkpoints/best_model.pth')
    else:
        sem_enc_model = pc2dsem_pct(latent_size=args.latent_size, recon_points=30000).cuda()
        checkpoint = torch.load('/home/install/pcc_point_based/MLPCC/log/PC2Dsem_PCT/10002_256_2048/checkpoints/best_model.pth')
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    sem_enc_model.load_state_dict(new_state_dict)

    if enc_name == 'EnB':
        depth_enc_model = pc2d_enb(latent_size=args.latent_size, recon_points=30000).cuda()
        checkpoint = torch.load('/home/install/pcc_point_based/MLPCC/log/PC2D_EnB/10000_256_2048/checkpoints/best_model.pth')
    else:
        depth_enc_model = pc2d_pct(latent_size=args.latent_size, recon_points=30000).cuda()
        checkpoint = torch.load('/home/install/pcc_point_based/MLPCC/log/PC2D_PCT/10000_256_2048/checkpoints/best_model.pth')
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    depth_enc_model.load_state_dict(new_state_dict)

    total_chamfer_dist = 0.0
    total_d1_psnr, total_d1_mse = 0.0, 0.0
    total_d2_psnr, total_d2_mse = 0.0, 0.0
    total_bpp = 0.0
    num_samples = 0
    cd_loss = 0.0
    rmse_loss = 0.0

    print("============start test!==============")
    for step, data in tqdm(enumerate(test_loader), total=(len(test_loader)), smoothing=0.9):
        with torch.no_grad():

            pc_data = data[0]
            pc_gt_raw = data[1]
            depth = data[2]
            depth = depth.cuda()
            depth = depth.unsqueeze(1)
            total_output_pc = data[0]
            total_output_bpp = 0.0

            depth_enc_model.eval()
            sem_enc_model.eval()
            model_1.eval()

            for level_ipx in range(args.level):
                start = args.input_number * level_ipx
                end = (level_ipx + 1) * args.input_number
                pc_data_ipx = pc_data[:, start:end, :]
                pc_data_ipx = pc_data_ipx.cuda()
                pc_gt_raw = pc_gt_raw.cuda()
                pc_data_rep = pc_data_ipx.repeat(4, 1, 1)

                ft_depth = depth_enc_model(pc_data_rep)
                ft_depth = depth_enc_model(pc_data_rep)
                ft_depth = ft_depth[0].unsqueeze(0)
                ft_sem = sem_enc_model(pc_data_rep)
                ft_sem = ft_sem[0].unsqueeze(0)
                ft_d_s = torch.cat((ft_sem, ft_depth), dim = 1)

                bpp, recon_xyz, cd_ds, cd_raw = model_1(ft_d_s, pc_data_ipx, pc_gt_raw)
                if level_ipx == 0:
                    total_output_pc = recon_xyz
                else:
                    total_output_pc = torch.cat((total_output_pc, recon_xyz), 1)
                total_output_bpp += bpp

            total_bpp += total_output_bpp
            print(total_output_pc.shape)

            decoder_output = total_output_pc
            cd, rmse = chamfer_multi(pc_gt_raw, decoder_output)
            cd_loss = cd_loss + cd * 1000000
            rmse_loss = rmse_loss + rmse * 10000
            # print(cd)
            print(rmse)

            pc_gt = pc_data[0].cpu().detach().numpy()
            pc_gt_raw = pc_gt_raw[0].cpu().detach().numpy()
            decoder_output = decoder_output[0].cpu().detach().numpy()

            d2_results = cal_d1(pc_gt_raw, decoder_output, step, checkpoint_path)
            d2_psnr = d2_results[2].item()
            d2_mse = d2_results[5].item()
            total_d2_mse += d2_mse
            total_d2_psnr += d2_psnr

            print("step:", step, "bpp:", bpp)
            print("step:", step, "d2_psnr:", d2_psnr)
            # print("step:", step, "d2_mse:", d2_mse)

        num_samples += 1

    total_chamfer_dist /= num_samples
    total_d1_psnr /= num_samples
    total_d2_mse /= num_samples
    total_d2_psnr /= num_samples
    total_d2_mse /= num_samples
    total_bpp /= num_samples
    outstr = "Average_bpp: %.6f, Average_D1_PSNR: %.6f, Average_D1_mse: %.6f, Average_Chamfer_Dist: %.6f, Average_D2_PSNR: %.6f, Average_D2_mse: %.6f\n" % (
             total_bpp,
             total_d1_psnr, total_d1_mse, total_chamfer_dist, total_d2_psnr, total_d2_mse)
    print(outstr)
    print(cd_loss/num_samples)
    print(rmse_loss/num_samples)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('point_based_PCGC')
    parser.add_argument('--dataset_path', type=str, default='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences')
    parser.add_argument('--seq_name', type=str, default='*')
    parser.add_argument('--num_point', type=int, default=2500, help='Point Number [default: 2048]')
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--input_number', type=int, default=2048)
    parser.add_argument('--level', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args.level)
    exp_name = '8010_512_2048'
    model_name = 'PCT_D_S_PCC_SR'
    experiment_dir = './log/'+model_name+'/'+exp_name
    MODELA = importlib.import_module(model_name)
    set_in = exp_name.split('_')
    model_1 = MODELA.get_model(latent_size=int(set_in[1]), recon_points=int(set_in[2])).cuda()
    model_1.eval()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    print(experiment_dir)
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v
    model_1.load_state_dict(new_state_dict)

    test(model_1, args)
