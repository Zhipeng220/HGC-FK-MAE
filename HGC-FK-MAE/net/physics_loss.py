import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------
# 创新点 2: 解剖学约束的重建 (Anatomically-Constrained Reconstruction)
# ---------------------------------------------------------------------------------

class AnatomicalLoss(nn.Module):
    """
    计算解剖学损失 L_Anat
    目前只实现 骨骼长度 (Bone Length) 约束
    """

    def __init__(self, num_joints=21, dataset='egogesture'):
        super().__init__()

        if dataset != 'egogesture' or num_joints != 21:
            # 仅为 EgoGesture (21 关节) 定义
            print(f"警告: PhysicsLoss 未为 {dataset} (关节数={num_joints}) 定义。跳过骨骼损失。")
            self.bone_pairs = []
            self.target_lengths = torch.tensor([])
        else:
            # 定义 EgoGesture (21 关节) 的骨骼连接
            # (父关节, 子关节)
            self.bone_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
                (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
                (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
                (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
                (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
            ]

            # 定义 EgoGesture 的标准骨骼长度 (这些是示例值，应从数据集中统计)
            # (N, C, T, V, M)
            # 我们假设一个标准的、归一化的手部骨骼长度 (V=21)
            # 这些值应该是通过分析整个 EgoGesture 训练集得到的平均骨骼长度
            target_lengths = [
                0.10, 0.08, 0.07, 0.06,  # 拇指
                0.12, 0.10, 0.08, 0.07,  # 食指
                0.13, 0.11, 0.09, 0.08,  # 中指
                0.12, 0.10, 0.08, 0.07,  # 无名指
                0.11, 0.09, 0.07, 0.06  # 小指
            ]
            self.register_buffer('target_lengths', torch.tensor(target_lengths))

    def compute_bone_length_loss(self, x_hat):
        """
        x_hat: (N, C, T, V, M) - 重建的骨骼坐标
        """
        if self.bone_pairs is None or self.target_lengths.numel() == 0:
            return torch.tensor(0.0, device=x_hat.device)

        N, C, T, V, M = x_hat.shape
        loss = 0.0

        for m in range(M):  # 遍历每个人
            x = x_hat[:, :, :, :, m]  # (N, C, T, V)

            # (N, T, V, C)
            x_permuted = x.permute(0, 2, 3, 1)

            bone_lengths = []
            for (p_idx, c_idx) in self.bone_pairs:
                parent = x_permuted[:, :, p_idx, :]  # (N, T, C)
                child = x_permuted[:, :, c_idx, :]  # (N, T, C)

                # (N, T)
                dist = torch.norm(parent - child, dim=2)
                bone_lengths.append(dist)

            # (N, T, num_bones)
            bone_lengths_tensor = torch.stack(bone_lengths, dim=2)

            # (num_bones,) -> (1, 1, num_bones)
            target_lengths = self.target_lengths.view(1, 1, -1)

            # 计算 L1 损失 (L1 通常比 L2 更鲁棒)
            # (N, T, num_bones)
            length_error = torch.abs(bone_lengths_tensor - target_lengths)
            loss += torch.mean(length_error)

        return loss / M

    def compute_joint_angle_loss(self, x_hat):
        """
        (占位符)
        计算关节角度损失 (例如，防止手指反向弯折)
        这需要定义 (a, b, c) 关节三元组和它们的最小/最大角度
        """
        return torch.tensor(0.0, device=x_hat.device)

    def forward(self, x_hat_reconstructed):
        """
        x_hat_reconstructed: (N, C, T, V, M) - 解码器输出
        """

        # 1. 骨骼长度约束
        loss_bone = self.compute_bone_length_loss(x_hat_reconstructed)

        # 2. 关节角度约束 (占位符)
        loss_angle = self.compute_joint_angle_loss(x_hat_reconstructed)

        # 3. 手掌平面性约束 (占位符)

        total_anat_loss = loss_bone + loss_angle

        return total_anat_loss