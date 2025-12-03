import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import random

sys.path.extend(['../'])
from feeder import tools


class Feeder(Dataset):
    """
    EgoGesture Dataset Feeder for AimCLR
    Arguments:
        data_path: path to data file (.npy)
        label_path: path to label file (.pkl)
        split: 'train' or 'test'
        random_choose: randomly choose a portion of the input sequence
        random_shift: randomly pad zeros at the begining or end of sequence
        random_move: randomly move the sequence
        window_size: The length of the output sequence
        normalization: normalize input sequence
        debug: only use first 100 samples for debugging
        use_mmap: use memory map to load data
        bone: use bone stream instead of joint stream
        vel: use velocity stream
        random_rot: randomly rotate the skeleton
        p_interval: sampling interval for test
        shear_amplitude: amplitude of shear augmentation
        temperal_padding_ratio: temporal padding ratio for augmentation
    """

    def __init__(self,
                 data_path,
                 label_path,
                 split='train',
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 normalization=False,
                 debug=False,
                 use_mmap=True,
                 bone=False,
                 vel=False,
                 random_rot=False,
                 p_interval=1,
                 shear_amplitude=0.5,
                 temperal_padding_ratio=6):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.bone = bone
        self.vel = vel
        self.random_rot = random_rot
        self.p_interval = p_interval
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data()

        if normalization:
            self.get_mean_map()

    def load_data(self):
        # Load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # Load label
        try:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # Handle different pickle formats
            with open(self.label_path, 'rb') as f:
                self.label, self.sample_name = pickle.load(f)

        # Debug mode: only use first 100 samples
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        print(f"Data shape: {self.data.shape}")
        print(f"Number of samples: {len(self.label)}")

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape

        if self.bone:
            print("Calculating mean/std for Bone stream...")
            bone_data = np.zeros_like(data)
            for v1, v2 in self.get_bone_connections():
                bone_data[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
            data = bone_data

        elif self.vel:  # <--- 🚨 必须确认这段逻辑存在！
            print("Calculating mean/std for Motion stream...")
            motion_data = np.zeros_like(data)
            motion_data[:, :-1] = data[:, 1:] - data[:, :-1]
            motion_data[:, -1] = 0
            data = motion_data

        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape(
            (C, 1, V, 1)) + 1e-4

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # =========================================================
        # 步骤 1: 时序裁剪/填充 (Crop/Resize) - 恢复到最前面！
        # =========================================================
        # 此时 data_numpy 还是纯净的 Joint 数据，使用 repeat padding 是安全的
        if self.window_size > 0:
            C, T, V, M = data_numpy.shape
            if T == self.window_size:
                pass
            elif T < self.window_size:
                # 自动填充 (Joint数据用重复填充代表静止，物理上是合理的)
                pad_len = self.window_size - T
                pad_data = np.repeat(data_numpy[:, [-1], :, :], pad_len, axis=1)
                data_numpy = np.concatenate([data_numpy, pad_data], axis=1)
            else:
                # 随机裁剪
                if self.random_choose:
                    data_numpy = tools.random_choose(data_numpy, self.window_size)
                else:
                    begin = (T - self.window_size) // 2
                    data_numpy = data_numpy[:, begin:begin + self.window_size, :, :]

        # Random shift (时序平移)
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)

        # =========================================================
        # 步骤 2: 空间增强 (Augment) - 作用于裁剪后的 Joint 数据
        # =========================================================
        # 恢复了对短序列的强增强，解决了 Joint 流过拟合问题
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        # =========================================================
        # 步骤 3: 流转换 (Joint -> Bone/Motion) - 关键！
        # =========================================================
        # 在增强后进行转换。
        # 1. 此时 Joint 已经带有随机平移/旋转。
        # 2. 计算差分时，平移被抵消，旋转被保留。Bone/Motion 物理性质完美！

        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.get_bone_connections():
                bone_data_numpy[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]
            data_numpy = bone_data_numpy

        elif self.vel:
            # 计算速度 (Motion)
            # 此时 Joint 已填充完毕，计算差分会自动得到 0 (静止)，非常完美
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        # =========================================================
        # 步骤 4: 归一化
        # =========================================================
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
            data_numpy = np.nan_to_num(data_numpy, copy=False, nan=0.0, posinf=100.0, neginf=-100.0)

        # =========================================================
        # 步骤 5: AimCLR 特定增强
        # =========================================================
        if self.split == 'train':
            if self.shear_amplitude > 0:
                data_numpy = tools.shear(data_numpy, self.shear_amplitude)
            if self.temperal_padding_ratio > 0:
                data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def get_bone_connections(self):
        """
        Define bone connections for hand skeleton (22 joints for SHREC)
        """
        num_joints = self.data.shape[3]

        if num_joints == 22:
            # SHREC'17 Track Layout (22 Joints)
            connections = [
                (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4),
                (6, 1), (7, 6), (8, 7), (9, 8),
                (10, 1), (11, 10), (12, 11), (13, 12),
                (14, 1), (15, 14), (16, 15), (17, 16),
                (18, 1), (19, 18), (20, 19), (21, 20)
            ]
        else:
            # EgoGesture (21 Joints)
            connections = [
                (0, 0), (1, 0), (2, 1), (3, 2), (4, 3),
                (5, 0), (6, 5), (7, 6), (8, 7),
                (9, 0), (10, 9), (11, 10), (12, 11),
                (13, 0), (14, 13), (15, 14), (16, 15),
                (17, 0), (18, 17), (19, 18), (20, 19)
            ]

        return connections

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
