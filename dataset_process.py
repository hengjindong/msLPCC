import os
import pdb
from tkinter import W
import open3d
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import torch
import json
import h5py
from glob import glob
from pytorch3d.ops import sample_farthest_points
import numpy as np
import torch.utils.data as data
import yaml
from kitti_tool.laserscan import LaserScan, SemLaserScan

class MMDataset(data.Dataset):
    def __init__(self, root, dataset_name='03', num_points=2048, split='train', depth_width = 512, depth_hight = 64):
        self.root_raw = os.path.join(root, "raw_"+split)
        self.root_ds = os.path.join(root, "ds_"+split)
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.depth_width = depth_width
        self.depth_hight = depth_hight

        self.path_raw_all, self.path_ds_all = self.get_path()
        self.data_raw, self.data_depth, self.data_depth_sem = self.load_ply(self.path_raw_all)
        self.data_ds = self.load_ply_ds(self.path_ds_all)
        self.len = int(len(self.data_ds) / 10) - 1

    def get_path(self):
        path_raw_all = []
        seq_name = self.dataset_name
        path_raw_all = os.path.join(
            self.root_raw,
            seq_name + '/velodyne_total_remove/*.ply'
            # seq_name + '/velodyne_voxel/*.ply'
        )
        path_raw_all = sorted(glob(path_raw_all))
        path_ds_all = []
        path_ds_all = os.path.join(
            self.root_ds,
            seq_name + '/velodyne_total_remove/*.ply'
            # seq_name + '/velodyne_hq/*.ply'
        )
        path_ds_all = sorted(glob(path_ds_all))
        path_raw_all = path_raw_all[0::10]
        path_ds_all = path_raw_all[0::10]
        return path_raw_all, path_ds_all

    def load_ply(self, path):
        all_data = []
        all_depth_data = []
        all_depth_sem_data = []
        for ply_name in tqdm(path):
            pcd = open3d.io.read_point_cloud(ply_name)
            data = np.asarray(pcd.points)
            data = data/100
            all_data.append(data)
            bin_path = ply_name.replace('.ply', '.bin')
            dir_part_name = 'velodyne_total_remove'
            dir_depth_name = 'depth_data_4'
            dir_depth_sem_name = 'depth_sem_data_4'
            bin_path = bin_path.replace(dir_part_name, 'velodyne')
            # bin_path = bin_path.replace('velodyne_voxel', 'velodyne')
            if (self.split == 'train'):
                bin_path = bin_path.replace('/raw_train', '')
                range_sem = bin_to_range_sem(bin_path, self.depth_hight, self.depth_width)
            else:
                bin_path = bin_path.replace('/raw_test', '')
                range_sem = bin_to_range(bin_path, self.depth_hight, self.depth_width)

            depth_dir_path = os.path.dirname(ply_name)
            # depth_dir_path = depth_dir_path.replace('velodyne_voxel', 'depth_data')
            depth_dir_path = depth_dir_path.replace(dir_part_name,dir_depth_name)
            experiment_dir = Path(depth_dir_path)
            experiment_dir.mkdir(exist_ok=True)

            depth_dir_path = os.path.dirname(ply_name)
            # depth_dir_path = depth_dir_path.replace('velodyne_voxel', 'depth_sem_data')
            depth_dir_path = depth_dir_path.replace(dir_part_name,dir_depth_sem_name)
            experiment_dir = Path(depth_dir_path)
            experiment_dir.mkdir(exist_ok=True)

            # depth_name = ply_name.replace('velodyne_voxel', 'depth_data')
            depth_name = ply_name.replace(dir_part_name, dir_depth_name)
            depth_name = depth_name.replace('.ply', '.bin')
            # sem_name = ply_name.replace('velodyne_voxel', 'depth_sem_data')
            sem_name = ply_name.replace(dir_part_name, dir_depth_sem_name)
            sem_name = sem_name.replace('.ply', '.bin')

            depth_data = range_sem[0]
            sem_data = range_sem[1]
            depth_data.tofile(depth_name)
            sem_data.tofile(sem_name)

        return all_data, all_depth_data, all_depth_sem_data

    def load_ply_ds(self, path):
        all_data = []
        for ply_name in tqdm(path):
            pcd = open3d.io.read_point_cloud(ply_name)
            data = np.asarray(pcd.points)
            data = data[0:65536]
            data = data/100
            all_data.append(data)
        return all_data

    def __getitem__(self, item):
        item_10 = item
        points_ds = self.data_ds[item_10]
        points_ds = torch.from_numpy(points_ds)
        points_ds = points_ds.float()

        points_raw = self.data_raw[item_10]
        points_raw = torch.from_numpy(points_raw)
        points_raw = points_raw.float()

        depth = self.data_depth[item_10]
        depth = torch.from_numpy(depth)
        depth = depth.float()

        depth_sem = self.data_depth_sem[item_10]
        depth_sem = torch.from_numpy(depth_sem)
        depth_sem = depth_sem.float()

        return points_ds, points_raw, depth, depth_sem

    def __len__(self):
        return len(self.data_ds)

def bin_to_range(bin_path, img_H, img_W):
    scan = LaserScan(project=True,
                     H=img_H,
                     W=img_W,
                     fov_up=2,
                     fov_down=-25)
    scan.open_scan(bin_path)
    proj_range = scan.proj_range
    proj_range = remove_balck_line_and_remote_points(proj_range)
    return proj_range

def bin_to_range_sem(bin_path, img_H, img_W):
    CFG = yaml.safe_load(open("./kitti_tool/config/semantic-kitti.yaml", 'r'))
    color_dict = CFG["color_map"]
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses,
                        color_dict,
                        project=True,
                        H=img_H,
                        W=img_W,
                        fov_up=2,
                        fov_down=-25)
    scan.open_scan(bin_path)
    label_path = bin_path.replace("data_odometry_velodyne", "data_odometry_labels")
    label_path = label_path.replace("/velodyne", "/labels")
    label_path = label_path.replace(".bin", ".label")
    scan.open_label(label_path)
    scan.colorize()
    # proj_sem_color = scan.proj_sem_color[..., ::-1]
    # pdb.set_trace()
    proj_sem_color = (scan.proj_sem_label).astype(np.uint8)
    proj_sem_color = remove_balck_line_and_remote_points(proj_sem_color)
    proj_range = scan.proj_range
    proj_range = remove_balck_line_and_remote_points(proj_range)
    return proj_range, proj_sem_color

def bin_to_range(bin_path, img_H, img_W):
    CFG = yaml.safe_load(open("./kitti_tool/config/semantic-kitti.yaml", 'r'))
    scan = LaserScan(project=True,
                        H=img_H,
                        W=img_W,
                        fov_up=2,
                        fov_down=-25)
    scan.open_scan(bin_path)
    proj_range = scan.proj_range
    proj_range = remove_balck_line_and_remote_points(proj_range)
    return proj_range, proj_range

def remove_balck_line_and_remote_points(range_image):
    range_x = range_image.shape[0]
    range_y = range_image.shape[1]
    # initialize
    range_image_completion = range_image
    for height in range(1, range_x-1):
        for width in range(1, range_y-1):
            left = max(width - 1, 0)
            right = min(width + 1, range_y)
            up = max(height - 1, 0)
            down = min(height + 1, range_x)
            if range_image[height, width] == 0:
                # remove straight line
                if range_image[up][width] > 0 and range_image[down][width] > 0 and (range_image[height][left] == 0 or range_image[height][right] == 0):
                    range_image_completion[height][width] = (range_image[up][width] + range_image[down][width]) / 2

    for height in range(1, range_x-1):
        for width in range(1, range_y-1):
            left = max(width - 1, 0)
            right = min(width + 1, range_y)
            up = max(height - 1, 0)
            down = min(height + 1, range_x)
            if range_image_completion[height][width] == 0:
                point_up = range_image_completion[up][width]
                point_down = range_image_completion[down][width]
                point_left = range_image_completion[height][left]
                point_right = range_image_completion[height][right]
                point_left_up = range_image_completion[up][left]
                point_right_up = range_image_completion[up][right]
                point_left_down = range_image_completion[down][left]
                point_right_down = range_image_completion[down][right]
                surround_points = int(point_up != 0) + int(point_down != 0) + int(point_left != 0) + int(
                    point_right != 0) + int(point_left_up != 0) + int(point_right_up != 0) + int(
                    point_left_down != 0) + int(point_right_down != 0)
                if surround_points >= 7:
                    surround_points_sum = point_up + point_down + point_left + point_right + point_left_up + point_right_up + point_left_down + point_right_down
                    range_image_completion[height][width] = surround_points_sum / surround_points

    return range_image_completion

def save_pc_level(data, level):
    if (not data.dtype == torch.float32):
        data = torch.from_numpy(data)
    data_level = torch.chunk(data, chunks=int(len(data)/2048), dim=0)
    pdb.set_trace()
    data_level_cut = data_level[level]
    save_pcd = open3d.geometry.PointCloud() # 定义点云
    save_pcd.points = open3d.utility.Vector3dVector(np.squeeze(data_level_cut))
    save_path = "./pc_file/level/" + str(level) + ".ply"
    open3d.io.write_point_cloud(save_path, save_pcd, write_ascii=True)

def save_pc_level_sum(data, level):
    if (not data.dtype == torch.float32):
        data = torch.from_numpy(data)
    end = 2048 * level
    data_level_sum = data[0:end, :]
    save_pcd = open3d.geometry.PointCloud() # 定义点云
    save_pcd.points = open3d.utility.Vector3dVector(np.squeeze(data_level_sum))
    save_path = "./pc_file/level_sum/" + str(level) + ".ply"
    open3d.io.write_point_cloud(save_path, save_pcd, write_ascii=True)

TRAIN_DATASET = MMDataset(root='/media/install/serverdata/KITTI/SemanticKITTI/data_odometry_velodyne/dataset/sequences',
                        # dataset_name = '07',
                        dataset_name = '*',
                        num_points=2048,
                        split='test',
                        # split='train',
                        depth_width=4,
                        depth_hight=4)
