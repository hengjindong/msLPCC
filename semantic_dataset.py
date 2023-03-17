import os
import pdb
import open3d
from tqdm import tqdm
import torch
import json
import h5py
from glob import glob
from pytorch3d.ops import sample_farthest_points
import numpy as np
import torch.utils.data as data
import open3d as o3d
from pathlib import Path

class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='03', num_points=2048, split='train', load_name=True, load_file=True,):
        self.root = os.path.join(root, dataset_name)
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.path_bin_all = self.get_path()
        self.data_ds, self.data_raw = self.load_bin_data(self.path_bin_all)

    def get_path(self):
        path_bin_all = []
        path_bin_all = os.path.join(
            self.root,
            'velodyne/*.bin'
        )
        path_bin_all = sorted(glob(path_bin_all))
        path_bin_all = path_bin_all[0::10]
        return path_bin_all

    def load_bin_data(self, path):
        all_data_ds = []
        all_data_raw = []
        for bin_name in tqdm(path):
            scan = np.fromfile(bin_name, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            scan = scan[:, 0:3]
            data = scan
            data_raw = remove_outof_range(data, 25)
            data_ds = farthest_sample_pytorch3d(data_raw, 2048)
            # data_ds = farthest_sample_pytorch3d(data_raw, 65536)

            ply_name = bin_name.replace(".bin",".ply",1)
            pcd_raw_path = ply_name.replace("/"+self.dataset_name+"/","/raw_"+self.split+"/"+self.dataset_name+"/",1)
            pcd_ds_path = ply_name.replace("/"+self.dataset_name+"/","/ds_"+self.split+"/"+self.dataset_name+"/",1)

            pcd_raw_path = pcd_raw_path.replace("/velodyne/", "/velodyne_range25__mutil_dim/")
            pcd_ds_path = pcd_ds_path.replace("/velodyne/", "/velodyne_range25_mutil_dim/")

            pcd_raw_dir = os.path.dirname(pcd_raw_path)
            experiment_dir = Path(pcd_raw_dir)
            experiment_dir.mkdir(exist_ok=True)
            pcd_ds_dir = os.path.dirname(pcd_ds_path)
            experiment_dir = Path(pcd_ds_dir)
            experiment_dir.mkdir(exist_ok=True)

            pcd_raw = open3d.geometry.PointCloud()
            pcd_raw.points = open3d.utility.Vector3dVector(data_raw)
            open3d.io.write_point_cloud(pcd_raw_path, pcd_raw, write_ascii=True)

            pcd_ds = open3d.geometry.PointCloud()
            pcd_ds.points = open3d.utility.Vector3dVector(data_ds)
            open3d.io.write_point_cloud(pcd_ds_path, pcd_ds, write_ascii=True)

        return all_data_ds, all_data_raw

    def __getitem__(self, item):
        point_set = self.data_ds[item]
        point_raw_set = self.data_raw[item]
        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        point_raw_set = torch.from_numpy(point_raw_set)
        # point_set = point_set.unsqueeze(0)
        # data, idx = sample_farthest_points(point_set, K=self.num_points)
        # point_set = data.squeeze(0)
        return point_set, point_raw_set

    def __len__(self):
        return len(self.data)

def remove_center(pc):
    pc_cut = np.zeros((140000, 3))
    pc_cut_idx = 0
    for i in range(len(pc)):
        # if ((pc[i][0]*pc[i][0]+pc[i][1]*pc[i][1]<range_K) and pc_cut_idx < 140000):
        this_range = pc[i][0]**2+pc[i][1]**2+pc[i][2]**2
        if (pc[i][2] > -3 and this_range > 8):
            pc_cut[pc_cut_idx] = pc[i]
            pc_cut_idx += 1
    pc_cut = pc_cut.astype(np.float32)
    return pc_cut

def remove_outof_range(pc, range_k):
    pc_cut = np.zeros((140000, 3))
    pc_cut_idx = 0
    range_k = range_k**2
    for i in range(len(pc)):
        this_range = pc[i][0]**2+pc[i][1]**2
        if (pc[i][2] > -3 and this_range < range_k):
            pc_cut[pc_cut_idx] = pc[i]
            pc_cut_idx += 1
    pc_cut = pc_cut.astype(np.float32)
    return pc_cut

def pc_sample(pc, K_input):
    pc = pc.unsqueeze(0)
    pc_sample, idx = sample_farthest_points(pc, K=K_input)
    pc_sample_np = np.array(pc_sample)
    pc_sample_np = pc_sample_np.squeeze(0)
    np.savetxt("./or_sample_"+str(K_input)+".txt", pc_sample_np)

def farthest_sample_pytorch3d(pc, K_input):
    pc = torch.from_numpy(pc)
    pc = pc.unsqueeze(0)
    pc_sample, idx = sample_farthest_points(pc, K=K_input)
    pc_sample_np = np.array(pc_sample)
    pc_sample_np = pc_sample_np.squeeze(0)
    return pc_sample_np

def pc_total(pc, K_input):
    pc_cut = np.zeros((K_input, 3))
    pc_cut_idx = 0
    for i in range(len(pc)):
        pc_cut[pc_cut_idx] = pc[i]
        pc_cut_idx += 1
    pc_cut = pc_cut.astype(np.float32)
    return pc_cut

def pc_split(pc, K_input):
    pc_cut = np.zeros((K_input, 3))
    pc_cut_idx = 0
    for i in range(len(pc)):
        if ((pc[i][0] > 0) and (pc[i][1] > 0) and pc_cut_idx < K_input):
            pc_cut[pc_cut_idx] = pc[i]
            pc_cut_idx += 1
    pc_cut = pc_cut.astype(np.float32)
    return pc_cut

def pc_total_range(pc, range_K):
    pc_cut = np.zeros((140000, 3))
    pc_cut_idx = 0
    for i in range(len(pc)):
        if ((pc[i][0]*pc[i][0]+pc[i][1]*pc[i][1]<range_K) and pc_cut_idx < 140000):
            pc_cut[pc_cut_idx] = pc[i]
            pc_cut_idx += 1
    pc_cut = pc_cut.astype(np.float32)
    return pc_cut

def pc_split_range_two(pc, range_K):
    pc_cut_a = np.zeros((40000, 3))
    pc_cut_b = np.zeros((40000, 3))
    for i in range(len(pc)):
        range_temp =pc[i][0]*pc[i][0]+pc[i][1]*pc[i][1]
        if (range_temp > range_K):
            pc_cut_b[i] = pc[i]
        else:
            pc_cut_a[i] = pc[i]
    pc_cut_a = pc_cut_a.astype(np.float32)
    pc_cut_b = pc_cut_b.astype(np.float32)
    return pc_cut_a, pc_cut_b

def pc_split_2(pc, K_input):
    pc_cut = np.zeros((K_input, 3))
    pc_cut_idx = 0, pc
    for i in range(len(pc)):
        if ((pc[i][0] > 0) and pc_cut_idx < K_input):
            pc_cut[pc_cut_idx] = pc[i]
            pc_cut_idx += 1
    pc_cut = pc_cut.astype(np.floaat32)
    return pc_cut

def pc_split_8_1(pc, K_input):
    pc_cut = np.zeros((K_input, 3))
    pc_cut_idx = 0
    for i in range(len(pc)):
        if ((pc[i][0] > 0) and (pc[i][1] > 0) and (pc[i][0] / pc[i][1]) < 1 and pc_cut_idx < K_input):
            pc_cut[pc_cut_idx] = pc[i]
            pc_cut_idx += 1
    pc_cut = pc_cut.astype(np.float32)
    return pc_cut

def pc_range_split(pc, K_input):
    pc_cut = np.zeros((90000, 3))
    pc_cut_idx = 0
    for i in range(len(pc)):
        if (pc[i][0]*pc[i][0]+pc[i][1]*pc[i][1] < 0.09 and pc_cut_idx < 90000):
        # if (pc[i][0]*pc[i][0]+pc[i][1]*pc[i][1] < 0.0225 ):
            pc_cut[pc_cut_idx] = pc[i]
            pc_cut_idx += 1
    # pdb.set_trace()
    pc_cut_idx = 0
    pc_cut = pc_cut.astype(np.float32)
    pc_cut2 = np.zeros((K_input, 3))
    for i in range(len(pc_cut)):
        if ((pc_cut[i][0] > 0) and (pc_cut[i][1] > 0) and pc_cut_idx < K_input):
            pc_cut2[pc_cut_idx] = pc_cut[i]
            pc_cut_idx += 1
    pc_cut2 = pc_cut2.astype(np.float32)
    return pc_cut2

def pc_voxel_split(pc, K_input):
    pc_voxel = np.zeros((100000, 3))
    pc_o3d = open3d.geometry.PointCloud()
    pc_o3d.points = open3d.utility.Vector3dVector(pc)
    # step = 0.2
    step = 0.5
    pc_sampling = open3d.geometry.PointCloud.voxel_down_sample(pc_o3d, step)
    pc_sampling = np.asarray(pc_sampling.points)
    ds_len = len(pc_sampling)
    if ds_len > K_input:
        min_len = K_input
    elif ds_len <= K_input:
        min_len = ds_len
    for i in range(min_len):
        pc_voxel[i] = pc_sampling[i]
    pc_voxel = pc_voxel.astype(np.float32)

    pc_cut = np.zeros((K_input, 3))
    pc_cut_idx = 0
    for i in range(len(pc_voxel)):
        if ((pc_voxel[i][0] > 0) and (pc_voxel[i][1] > 0) and pc_cut_idx < K_input):
            pc_cut[pc_cut_idx] = pc_voxel[i]
            pc_cut_idx += 1
    pc_cut = pc_cut.astype(np.float32)
    return pc_cut

def pc_voxel(pc, K_input):
    pc_voxel = np.zeros((K_input, 3))
    pc_o3d = open3d.geometry.PointCloud()
    pc_o3d.points = open3d.utility.Vector3dVector(pc)
    if K_input == 30000:
        step = 0.2
    elif K_input == 10000:
        step = 0.5
    elif K_input == 60000:
        step = 0.1
    elif K_input == 4000:
        step = 1
    pc_sampling = open3d.geometry.PointCloud.voxel_down_sample(pc_o3d, step)
    pc_sampling = np.asarray(pc_sampling.points)
    ds_len = len(pc_sampling)
    if ds_len > K_input:
        min_len = K_input
    elif ds_len <= K_input:
        min_len = ds_len
    for i in range(min_len):
        pc_voxel[i] = pc_sampling[i]
    pc_voxel = pc_voxel.astype(np.float32)
    return pc_voxel


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

if __name__ == '__main__':
    rootdir = '/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences'
    # seq_name = ['11','12','13','14','15','16','17','18','19','20','21']
    seq_name = ['00','01','02','03','04','05','06','07','08','09','10']
    # seq_name = ['10', '09', '08', '07', '06']
    for seq in seq_name:
        # split = 'test'
        split = 'train'
        d = Dataset(root=rootdir, dataset_name=seq, num_points=30000, split=split)
