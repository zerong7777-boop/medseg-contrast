import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dice_coeff(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    # Flatten the input and target tensors
    input = input.reshape(-1)
    target = target.reshape(-1)

    # Calculate the intersection and sum
    intersection = (input * target).sum()
    sets_sum = input.sum() + target.sum()

    # Calculate the Dice coefficient
    dice = (2 * intersection + epsilon) / (sets_sum + epsilon)
    return dice


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = True):
    if multiclass:
        # Compute Dice Loss for multi-class
        # Assuming input has shape [batch_size, num_classes, height, width]
        input = F.softmax(input, dim=1)  # Convert logits to probabilities





        # Ensure target is in the form of [batch_size, height, width]
        if target.dim() == 4:  # If target has extra channel dimension, squeeze it
            target = target.squeeze(1)


        # print(f"input.shape{input.shape}")
        # print(f"out.shape{target.shape}")
        # Convert target to one-hot format, ensure target is long type
        target = F.one_hot(target.long(), num_classes=input.shape[1]).permute(0, 3, 1, 2).float()

        dice_per_class = []
        for i in range(input.shape[1]):
            dice_per_class.append(dice_coeff(input[:, i], target[:, i]))  # Compute for each class

        # Return Dice loss as 1 - average dice and per-class dice
        return 1 - sum(dice_per_class) / len(dice_per_class), dice_per_class
    else:
        # Compute Dice Loss for single-class (binary classification)
        return 1 - dice_coeff(input, target), [dice_coeff(input, target)]



def dice_ce_loss(input: torch.Tensor, target: torch.Tensor, dice_weight=0.5, ce_weight=0.5, multiclass=True):
    """
    结合 Dice 损失和交叉熵损失
    """
    ce_input = input.squeeze(1)
    ce_tar = input.squeeze(1).float()

    # 计算交叉熵损失
    ce_loss = F.cross_entropy(ce_input, ce_tar)

    # 计算 Dice 损失
    dice_loss_value, _ = dice_loss(input, target, multiclass)  # 使用 dice_loss 来计算 Dice 损失

    # 加权求和
    total_loss = dice_weight * dice_loss_value + ce_weight * ce_loss
    return total_loss, dice_loss_value, ce_loss

import torch
import torch.nn.functional as F

def ssim_loss(pred, target, window_size=11, size_average=True, upsample_size=(8, 8)):
    """
    计算 SSIM 损失（支持上采样）
    :param pred: 预测值 [B, C, H, W]
    :param target: 目标值 [B, C, H, W]
    :param window_size: 滑动窗口大小
    :param size_average: 是否对结果取平均
    :param upsample_size: 上采样后的目标分辨率
    :return: SSIM 损失值
    """
    def gaussian(window_size, sigma):
        gauss = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(gauss ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.T
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    # 上采样到指定分辨率
    pred = F.interpolate(pred, size=upsample_size, mode='bilinear', align_corners=False)
    target = F.interpolate(target, size=upsample_size, mode='bilinear', align_corners=False)

    # 确保 pred 和 target 的通道数一致
    if pred.size(1) != target.size(1):
        raise ValueError(f"Pred and Target channel mismatch: {pred.size(1)} vs {target.size(1)}")

    _, channel, height, width = pred.size()
    window = create_window(window_size, channel=1).to(pred.device)  # 将窗口通道数设置为 1

    # 计算均值
    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=1)  # 使用单通道分组
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=1)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    # 计算方差和协方差
    sigma_pred_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=1) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=1) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2, groups=1) - mu_pred_target

    # SSIM 计算公式
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    # 平均 SSIM 损失
    ssim = ssim_map.mean() if size_average else ssim_map
    return 1 - ssim


def shape_aware_loss(output, target):
    # 获取输出和目标张量的形状
    output_shape = output.shape
    target_shape = target.shape

    # 确保输出和目标的形状相同
    assert output_shape == target_shape, "Output and target shapes must be the same."

    # 计算每个维度的形状差异
    shape_diff = torch.abs(torch.tensor(output_shape) - torch.tensor(target_shape))

    # 计算权重，这里简单地使用形状差异的倒数
    weights = 1.0 / (shape_diff + 1.0)

    weights=weights.to(device)
    # 计算平均平方误差损失，并应用形状权重
    loss = torch.mean(weights * F.mse_loss(output, target))

    return loss


def boundary_aware_loss(prediction, target, boundary_weight=2.0, epsilon=1e-5):
    """
    Boundary-aware loss for binary segmentation tasks.

    Args:
    - prediction: Predicted binary segmentation map
    - target: Ground truth binary segmentation map
    - boundary_weight: Weight for boundary pixels
    - epsilon: Small value to prevent division by zero

    Returns:
    - Boundary-aware loss
    """
    # Compute binary cross-entropy loss
    bce_loss = F.binary_cross_entropy(prediction, target)

    # Compute boundary-aware loss
    boundary_loss = - (target * torch.log(prediction + epsilon) + (1 - target) * torch.log(1 - prediction + epsilon))
    boundary_loss = torch.sum(boundary_loss)

    # Combine binary cross-entropy loss and boundary-aware loss
    total_loss = bce_loss + boundary_weight * boundary_loss

    return total_loss


from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor


# class PerceptualLoss(torch.nn.Module):
#     def __init__(self, layers=['features.16'], pretrained=True):
#         super(PerceptualLoss, self).__init__()
#         model = vgg19(pretrained=pretrained).features
#         # 使用节点名精确匹配
#         self.feature_extractor = create_feature_extractor(
#             model, return_nodes={layers[0]: 'features'}
#         )
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
#
#         # 降维模块
#         self.reduce_channels = torch.nn.Conv2d(768, 3, kernel_size=1)
#
#     def forward(self, pred, target):
#         pred = self.reduce_channels(pred)
#         target = self.reduce_channels(target)
#
#         pred_features = self.feature_extractor(pred)['features']
#         target_features = self.feature_extractor(target)['features']
#
#         return F.l1_loss(pred_features, target_features)

def combined_sr_loss(pred, target, alpha=0.5,beta=0.5):
    """
    联合 L1 损失与余弦相似度损失
    :param pred: 预测特征 [B, C, H, W]
    :param target: 目标特征 [B, C, H, W]
    :param alpha: 权重因子，控制两种损失的比例
    :return: 联合损失
    """
    # L1 损失
    l1_loss = F.l1_loss(pred, target)

    # Cosine 相似度损失
    cosine_loss = 1 - F.cosine_similarity(pred.flatten(2), target.flatten(2), dim=-1).mean()

    # 联合损失
    return alpha * l1_loss + beta * cosine_loss


def enhanced_sr_loss(pred, target, alpha=0.5, beta=0.3, gamma=0.2):
    """
    改进后的 SR 损失：联合 L1 损失、余弦相似度损失和梯度损失
    :param pred: 预测特征 [B, C, H, W]
    :param target: 目标特征 [B, C, H, W]
    :param alpha: 权重因子，控制 L1 损失的比例
    :param beta: 权重因子，控制余弦相似度损失的比例
    :param gamma: 权重因子，控制梯度损失的比例
    :return: 联合损失
    """
    # 1. 计算 L1 损失
    l1_loss = F.l1_loss(pred, target)

    # 2. 计算余弦相似度损失
    cosine_loss = 1 - F.cosine_similarity(pred.flatten(2), target.flatten(2), dim=-1).mean()

    # 3. 计算梯度损失
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).to(pred.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0).to(pred.device)

    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)

    grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x)
    grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y)
    grad_loss = grad_loss_x + grad_loss_y

    # 4. 联合损失
    combined_loss = alpha * l1_loss + beta * cosine_loss + gamma * grad_loss

    return combined_loss


def combined_sr_loss_with_gradient(pred, target, alpha=0.4, beta=0.4, gamma=0.2):
    """
    综合 L1 损失、余弦相似度与梯度损失
    :param pred: 预测特征 [B, C, H, W]，C 为高通道数（如 768）
    :param target: 目标特征 [B, C, H, W]
    :param alpha: L1 损失权重
    :param beta: 余弦相似度损失权重
    :param gamma: 梯度损失权重
    :return: 联合损失
    """
    # L1 损失
    l1_loss = F.l1_loss(pred, target)

    # Cosine 相似度损失
    cosine_loss = 1 - F.cosine_similarity(pred.flatten(2), target.flatten(2), dim=-1).mean()

    # 梯度损失
    B, C, H, W = pred.shape

    # 创建 Sobel 核，扩展到所有通道
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(pred.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(pred.device)

    # 扩展核到多通道 [C, 1, 3, 3]
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)

    # 使用分组卷积计算梯度
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=C)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=C)
    target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=C)
    target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=C)

    pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
    target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2)

    grad_loss = F.l1_loss(pred_grad, target_grad)

    # 联合损失
    return alpha * l1_loss + beta * cosine_loss + gamma * grad_loss

def dynamic_weighted_sr_loss(pred, target, epoch, total_epochs):
    """
    动态调整 SR 损失权重
    :param pred: 预测特征
    :param target: 目标特征
    :param epoch: 当前训练轮次
    :param total_epochs: 总训练轮次
    :return: SR 损失
    """
    # 随训练进度动态调整权重
    alpha = 0.5 + 0.5 * (epoch / total_epochs)  # L1 损失权重逐渐增加
    beta = 0.5 - 0.5 * (epoch / total_epochs)  # 余弦损失权重逐渐减少
    gamma = 0.2  # 梯度损失保持不变

    return combined_sr_loss_with_gradient(pred, target, alpha=alpha, beta=beta, gamma=gamma)


import torch
import torch.nn.functional as F
from torchvision import models


import torch.nn.functional as F

import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 使用预训练的VGG16模型
        self.vgg = models.vgg16(pretrained=True).features[:16].eval()

        # 将输入通道数调整为3
        self.conv = nn.Conv2d(768, 3, kernel_size=1)  # 将通道数从768调整为3

    def forward(self, pred, target):
        """
        计算感知损失，先通过1x1卷积将输入特征图的通道数从768调整为3
        :param pred: 预测图像 [B, 768, H, W]
        :param target: 目标图像 [B, 768, H, W]
        :return: 感知损失
        """
        # 确保卷积层的权重在正确的设备上（与输入一致）
        device = pred.device  # 获取输入张量的设备
        self.vgg = self.vgg.to(device)  # 将VGG模型移动到正确的设备
        self.conv = self.conv.to(device)  # 将1x1卷积层权重移动到正确的设备

        # 通过1x1卷积调整通道数
        pred = self.conv(pred)  # [B, 3, H, W]
        target = self.conv(target)  # [B, 3, H, W]

        # 如果输入尺寸过小，进行上采样
        if pred.size(2) == 1 and pred.size(3) == 1:
            # 将图像大小放大至32x32，避免0x0大小
            pred = F.interpolate(pred, size=(32, 32), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(32, 32), mode='bilinear', align_corners=False)

        # 使用VGG提取特征
        pred_feats = self.vgg(pred)
        target_feats = self.vgg(target)

        # 计算MSE损失作为感知损失
        loss = F.mse_loss(pred_feats, target_feats)
        return loss


def combined_sr_loss_with_gradient_2(pred, target, alpha=0.4, beta=0.4, gamma=0.2, perceptual_weight=0.2):
    """
    综合L1损失、余弦相似度损失与梯度损失，加入感知损失。
    :param pred: 预测特征 [B, C, H, W]
    :param target: 目标特征 [B, C, H, W]
    :param alpha: L1损失权重
    :param beta: 余弦相似度损失权重
    :param gamma: 梯度损失权重
    :param perceptual_weight: 感知损失权重
    :return: 联合损失
    """
    # L1 损失
    l1_loss = F.l1_loss(pred, target)

    # Cosine 相似度损失
    cosine_loss = 1 - F.cosine_similarity(pred.flatten(2), target.flatten(2), dim=-1).mean()

    # 梯度损失
    B, C, H, W = pred.shape
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        pred.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        pred.device)
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)

    pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=C)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=C)
    target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=C)
    target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=C)

    pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)
    target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2)

    grad_loss = F.l1_loss(pred_grad, target_grad)

    # 创建 PerceptualLoss 实例
    perceptual_loss_fn = PerceptualLoss()

    # 计算感知损失
    perceptual_loss_value = perceptual_loss_fn(pred, target)

    # 综合损失
    return alpha * l1_loss + beta * cosine_loss + gamma * grad_loss + perceptual_weight * perceptual_loss_value


def dynamic_weighted_sr_loss_2(pred, target, epoch, total_epochs):
    """
    动态调整 SR 损失权重
    :param pred: 预测图像 [B, C, H, W]
    :param target: 目标图像 [B, C, H, W]
    :param epoch: 当前训练轮次
    :param total_epochs: 总训练轮次
    :return: 超分损失
    """
    # 随训练进度动态调整权重
    alpha = 0.5 + 0.5 * (epoch / total_epochs)  # L1 损失权重逐渐增加
    beta = 0.5 - 0.5 * (epoch / total_epochs)  # 余弦损失权重逐渐减少
    gamma = 0.2  # 梯度损失保持不变
    perceptual_weight = 0.2  # 感知损失的权重保持不变

    return combined_sr_loss_with_gradient_2(pred, target, alpha=alpha, beta=beta, gamma=gamma, perceptual_weight=perceptual_weight)
