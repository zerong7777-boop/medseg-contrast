import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义PSA模块
class PSA(nn.Module):
    def __init__(self, channel=512, reduction=4, S=4):
        super(PSA, self).__init__()
        self.S = S  # 尺度的数量，用于控制多尺度处理的维度

        # 定义不同尺度的卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1)
            for i in range(S)
        ])

        # 定义每个尺度对应的SE模块
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), # 自适应平均池化到1x1
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False), # 减少通道数
                nn.ReLU(inplace=True),# ReLU激活函数
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False), # 恢复通道数
                nn.Sigmoid()# Sigmoid激活函数，输出通道注意力权重
            ) for i in range(S)
        ])

        self.softmax = nn.Softmax(dim=1)  # 对每个位置的尺度权重进行归一化

    def forward(self, x):
        b, c, h, w = x.size()

         # 将输入在通道维度上分割为S份，对应不同的尺度
        SPC_out = x.view(b, self.S, c // self.S, h, w)

       # 对每个尺度的特征应用对应的卷积层
        conv_out = []
        for idx, conv in enumerate(self.convs):
            conv_out.append(conv(SPC_out[:, idx, :, :, :]))
        SPC_out = torch.stack(conv_out, dim=1)

        # 对每个尺度的特征应用对应的SE模块，获得通道注意力权重
        se_out = [se(SPC_out[:, idx, :, :, :]) for idx, se in enumerate(self.se_blocks)]
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand(-1, -1, -1, h, w)  # 扩展以匹配SPC_out的尺寸

        # 应用Softmax归一化注意力权重
        softmax_out = self.softmax(SE_out)

        # 应用注意力权重并合并多尺度特征
        PSA_out = SPC_out * softmax_out
        PSA_out = torch.sum(PSA_out, dim=1)  # 沿尺度维度合并特征

        return PSA_out

if __name__ == '__main__':# 测试PSA模块
    input = torch.randn(3, 512, 64, 64)# 创建一个随机输入
    psa = PSA(channel=512, reduction=4, S=4)# 实例化PSA模块
    output = psa(input)# 前向传播
    print(output.shape)