import os
import pdb
from tkinter import W
import open3d
import open3d as o3d
from tqdm import tqdm
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
        # seq_name = '11'
        seq_name = self.dataset_name
        path_raw_all = os.path.join(
            self.root_raw,
            seq_name + '/velodyne_total/*.ply'
            # seq_name + '/velodyne_voxel/*.ply'
        )
        path_raw_all = sorted(glob(path_raw_all))
        path_ds_all = []
        path_ds_all = os.path.join(
            self.root_ds,
            seq_name + '/velodyne_total/*.ply'
            # seq_name + '/velodyne_hq/*.ply'
        )
        path_ds_all = sorted(glob(path_ds_all))
        path_raw_all_sampling = []
        path_ds_all_sampling = []
        len_new = 0
        if self.split == 'test' :
            len_sampling = 50
        else :
            len_sampling = 22
        len_new = int(len(path_raw_all) / len_sampling) - 1
        for i in range(len_new) :
            idx = i * len_sampling
            path_raw_all_sampling.append(path_raw_all[idx])
            path_ds_all_sampling.append(path_ds_all[idx])
        return path_raw_all_sampling, path_ds_all_sampling

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
            bin_path = bin_path.replace('velodyne_total', 'velodyne')
            # bin_path = bin_path.replace('velodyne_voxel', 'velodyne')
            if (self.split == 'train'):
                bin_path = bin_path.replace('/raw_train', '')
                range_sem = bin_to_range_sem(bin_path, self.depth_hight, self.depth_width)
            else:
                bin_path = bin_path.replace('/raw_test', '')
                range_sem = bin_to_range(bin_path, self.depth_hight, self.depth_width)
            all_depth_data.append(range_sem[0])
            all_depth_sem_data.append(range_sem[1])

            # from PIL import Image
            # test_int = range_sem[1].astype(np.uint8)
            # test_int = test_int * 2
            # im = Image.fromarray(test_int)
            # im.save('./pc_file/test.png')
            # depth = test_int
            # depth = np.hsplit(depth, 4)
            # im1 = Image.fromarray(depth[0])
            # im1.save('./pc_file/1.png')
            # im2 = Image.fromarray(depth[1])
            # im2.save('./pc_file/2.png')
            # im3 = Image.fromarray(depth[2])
            # im3.save('./pc_file/3.png')
            # im4 = Image.fromarray(depth[3])
            # im4.save('./pc_file/4.png')
            # show_pc(data)
            # pdb.set_trace()
        return all_data, all_depth_data, all_depth_sem_data

    def load_ply_ds(self, path):
        all_data = []
        for ply_name in tqdm(path):
            pcd = open3d.io.read_point_cloud(ply_name)
            data = np.asarray(pcd.points)
            data = data[0:2048]
            data = data/100
            all_data.append(data)
        return all_data

    def __getitem__(self, item):
        # item_10 = item * 10
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

class MMDataset_from_bin(data.Dataset):
    def __init__(self, root, dataset_name='03', num_points=2048, split='train', depth_width = 64, depth_hight = 64):
        self.root_raw = os.path.join(root, "raw_"+split)
        self.root_ds = os.path.join(root, "ds_"+split)
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.depth_width = depth_width
        self.depth_hight = depth_hight
        # self.folder_name = 'velodyne_total_remove'
        self.folder_name = 'velodyne_total_remove_mutil_dim'
        self.folder_name = 'velodyne_total_remove'
        self.folder_name = 'velodyne_total'
        self.path_raw_all, self.path_ds_all = self.get_path()
        self.data_raw, self.data_depth, self.data_depth_sem = self.load_ply(self.path_raw_all)
        self.data_ds = self.load_ply_ds(self.path_ds_all)
        self.len = int(len(self.data_ds) / 10) - 1

    def get_path(self):
        path_raw_all = []
        seq_name = self.dataset_name
        path_raw_all = os.path.join(
            self.root_raw,
            seq_name + '/'+self.folder_name+'/*.ply'
        )
        path_raw_all = sorted(glob(path_raw_all))

        path_ds_all = []
        path_ds_all = os.path.join(
            self.root_ds,
            seq_name + '/'+self.folder_name+'/*.ply'
        )
        path_ds_all = sorted(glob(path_ds_all))

        if self.split == 'test':
            path_raw_all = path_raw_all[0::60]
            path_ds_all = path_ds_all[0::60]
        else :
            path_raw_all = path_raw_all[0::1]
            path_ds_all = path_ds_all[0::1]

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
            depth_path = bin_path.replace(self.folder_name, 'depth_data')
            depth_sem_path = bin_path.replace(self.folder_name, 'depth_sem_data')

            depth_data = np.fromfile(depth_path, dtype=np.float32)
            depth_sem_data = np.fromfile(depth_sem_path, dtype=np.uint8)

            depth_data.shape = self.depth_hight, self.depth_width
            if self.split == 'test' :
                depth_sem_data = depth_data
            else:
                depth_sem_data.shape = self.depth_hight, self.depth_width

            # depth_data = depth_data / 100
            # depth_sem_path = depth_sem_data / 100
            all_depth_data.append(depth_data)
            all_depth_sem_data.append(depth_sem_data)

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

class MMDataset_mutil_depth(data.Dataset):
    def __init__(self, root, dataset_name='03', num_points=2048, split='train', depth_width = 64, depth_hight = 64):
        self.root_raw = os.path.join(root, "raw_"+split)
        self.root_ds = os.path.join(root, "ds_"+split)
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.depth_width = depth_width
        self.depth_hight = depth_hight

        self.path_raw_all, self.path_ds_all = self.get_path()
        self.data_raw, self.data_depth_big, self.data_depth_sem_big, self.data_depth_mid, self.data_depth_sem_mid, self.data_depth_sml, self.data_depth_sem_sml = self.load_ply(self.path_raw_all)
        self.data_ds = self.load_ply_ds(self.path_ds_all)
        self.len = int(len(self.data_ds) / 10) - 1

    def get_path(self):
        path_raw_all = []
        # seq_name = '11'
        seq_name = self.dataset_name
        path_raw_all = os.path.join(
            self.root_raw,
            # seq_name + '/velodyne_total/*.ply'
            seq_name + '/velodyne_total_remove/*.ply'
        )
        path_raw_all = sorted(glob(path_raw_all))
        path_ds_all = []
        path_ds_all = os.path.join(
            self.root_ds,
            # seq_name + '/velodyne_total/*.ply'
            seq_name + '/velodyne_total_remove/*.ply'
        )
        path_ds_all = sorted(glob(path_ds_all))
        path_raw_all_sampling = []
        path_ds_all_sampling = []
        len_new = 0
        if self.split == 'test' :
            len_sampling = 8
        else :
            len_sampling = 2
        len_new = int(len(path_raw_all) / len_sampling) - 1
        for i in range(len_new) :
            idx = i * len_sampling
            path_raw_all_sampling.append(path_raw_all[idx])
            path_ds_all_sampling.append(path_ds_all[idx])
        return path_raw_all_sampling, path_ds_all_sampling


    def load_ply(self, path):
        all_data = []
        all_depth_data_big = []
        all_depth_sem_data_big = []
        all_depth_data_mid = []
        all_depth_sem_data_mid = []
        all_depth_data_sml = []
        all_depth_sem_data_sml = []
        for ply_name in tqdm(path):
            pcd = open3d.io.read_point_cloud(ply_name)
            data = np.asarray(pcd.points)
            data = data/100
            all_data.append(data)
            depth_data_big, depth_sem_data_big = load_depth(ply_name, 'depth_data', 'depth_sem_data', 64, self.split)
            depth_data_mid, depth_sem_data_mid = load_depth(ply_name, 'depth_data_16', 'depth_sem_data_16', 16, self.split)
            depth_data_sml, depth_sem_data_sml = load_depth(ply_name, 'depth_data_4', 'depth_sem_data_4', 4, self.split)
            depth_data_big = depth_data_big/100
            depth_sem_data_big = depth_sem_data_big/255
            depth_data_mid = depth_data_mid/100
            depth_sem_data_mid = depth_sem_data_mid/255
            depth_data_sml = depth_data_sml/100
            depth_sem_data_sml = depth_sem_data_sml/255
            all_depth_data_big.append(depth_data_big)
            all_depth_sem_data_big.append(depth_sem_data_big)
            all_depth_data_mid.append(depth_data_mid)
            all_depth_sem_data_mid.append(depth_sem_data_mid)
            all_depth_data_sml.append(depth_data_sml)
            all_depth_sem_data_sml.append(depth_sem_data_sml)
        return all_data, all_depth_data_big, all_depth_sem_data_big, all_depth_data_mid, all_depth_sem_data_mid, all_depth_data_sml, all_depth_sem_data_sml

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
        # item_10 = item * 10
        item_10 = item

        points_ds = self.data_ds[item_10]
        points_ds = torch.from_numpy(points_ds)
        points_ds = points_ds.float()

        points_raw = self.data_raw[item_10]
        points_raw = torch.from_numpy(points_raw)
        points_raw = points_raw.float()

        depth_big = self.data_depth_big[item_10]
        depth_big = torch.from_numpy(depth_big)
        depth_big = depth_big.float()

        depth_sem_big = self.data_depth_sem_big[item_10]
        depth_sem_big = torch.from_numpy(depth_sem_big)
        depth_sem_big = depth_sem_big.float()

        depth_mid = self.data_depth_mid[item_10]
        depth_mid = torch.from_numpy(depth_mid)
        depth_mid = depth_mid.float()

        depth_sem_mid = self.data_depth_sem_mid[item_10]
        depth_sem_mid = torch.from_numpy(depth_sem_mid)
        depth_sem_mid = depth_sem_mid.float()

        depth_sml = self.data_depth_sml[item_10]
        depth_sml = torch.from_numpy(depth_sml)
        depth_sml = depth_sml.float()

        depth_sem_sml = self.data_depth_sem_sml[item_10]
        depth_sem_sml = torch.from_numpy(depth_sem_sml)
        depth_sem_sml = depth_sem_sml.float()
        return points_ds, points_raw, depth_big, depth_sem_big, depth_mid, depth_sem_mid, depth_sml, depth_sem_sml

    def __len__(self):
        return len(self.data_ds)

def load_depth(ply_name, dir_part_depth, dir_part_depth_sem, size, is_test):
    bin_path = ply_name.replace('.ply', '.bin')
    depth_path = bin_path.replace('velodyne_total_remove',dir_part_depth)
    depth_sem_path = bin_path.replace('velodyne_total_remove',dir_part_depth_sem)

    depth_data = np.fromfile(depth_path, dtype=np.float32)
    depth_sem_data = np.fromfile(depth_sem_path, dtype=np.uint8)

    depth_data.shape = size, size
    if is_test == 'test' :
        depth_sem_data = depth_data
    else:
        depth_sem_data.shape = size, size

    return depth_data, depth_sem_data

class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='03', num_points=2048, split='train', load_name=True, load_file=True,):
        self.root_raw = os.path.join(root, "raw_"+split)
        self.root_ds = os.path.join(root, "ds_"+split)
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file

        self.path_raw_all, self.path_ds_all = self.get_path()
        self.data_raw = self.load_ply(self.path_raw_all)
        self.data_ds = self.load_ply_ds(self.path_ds_all)

    def get_path(self):
        path_raw_all = []
        seq_name = self.dataset_name
        path_raw_all = os.path.join(
            self.root_raw,
            seq_name +  '/velodyne_total/*.ply'
        )
        path_raw_all = sorted(glob(path_raw_all))
        path_ds_all = []
        path_ds_all = os.path.join(
            self.root_ds,
            seq_name +  '/velodyne_total/*.ply'
        )
        path_ds_all = sorted(glob(path_ds_all))
        if self.split == 'test':
            path_raw_all = path_raw_all[0::2]
            path_ds_all = path_ds_all[0::2]
        path_raw_all = path_raw_all[0::1]
        path_ds_all = path_ds_all[0::1]
        return path_raw_all, path_ds_all

    def load_ply(self, path):
        all_data = []
        for ply_name in tqdm(path):
            pcd = open3d.io.read_point_cloud(ply_name)
            data = np.asarray(pcd.points)
            # data = data/15
            data = data/100
            all_data.append(data)
        return all_data

    def load_ply_ds(self, path):
        all_data = []
        for ply_name in tqdm(path):
            pcd = open3d.io.read_point_cloud(ply_name)
            data = np.asarray(pcd.points)
            data = data[0:2048]
            # data = data/15
            data = data/100
            all_data.append(data)
        return all_data

    def __getitem__(self, item):
        points_ds = self.data_ds[item]
        points_ds = torch.from_numpy(points_ds)
        points_ds = points_ds.float()

        points_raw = self.data_raw[item]
        points_raw = torch.from_numpy(points_raw)
        points_raw = points_raw.float()

        return points_ds, points_raw

    def __len__(self):
        return len(self.data_ds)

def farthest_sample_pytorch3d(pc, K_input):
    pc = torch.from_numpy(pc)
    pc = pc.unsqueeze(0)
    pc_sample, idx = sample_farthest_points(pc, K=K_input)
    pc_sample_np = np.array(pc_sample)
    pc_sample_np = pc_sample_np.squeeze(0)
    return pc_sample_np

def bin_to_range(bin_path, img_H, img_W):
    scan = LaserScan(project=True,
                     H=img_H,
                     W=img_W,
                     fov_up=3,
                     fov_down=-25)
    scan.open_scan(bin_path)
    proj_range = scan.proj_range
    # proj_range = remove_balck_line_and_remote_points(proj_range)
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
                        fov_up=3,
                        fov_down=-25)
    scan.open_scan(bin_path)
    ps = bin_path.split('/')
    bin_label = os.path.join('/'.join(ps[0:6]), 'data_odometry_labels', '/'.join(ps[7:10]), "labels", ps[11].split('.')[0]+".label")
    scan.open_label(bin_label)
    scan.colorize()
    # proj_sem_color = scan.proj_sem_color[..., ::-1]
    # pdb.set_trace()
    proj_sem_color = (scan.proj_sem_label).astype(np.uint8)
    proj_sem_color = remove_balck_line_and_remote_points(proj_sem_color)
    proj_range = scan.proj_range
    proj_range = remove_balck_line_and_remote_points(proj_range)
    # proj_range = remove_balck_line_and_remote_points(proj_range)
    return proj_range, proj_sem_color

def bin_to_range(bin_path, img_H, img_W):
    CFG = yaml.safe_load(open("./kitti_tool/config/semantic-kitti.yaml", 'r'))
    scan = LaserScan(project=True,
                        H=img_H,
                        W=img_W,
                        fov_up=3,
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

def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

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

