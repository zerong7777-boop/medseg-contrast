import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# DCT 频域处理模块
class DCTModule(nn.Module):
    def __init__(self, freq_range):
        super(DCTModule, self).__init__()
        self.freq_range = freq_range  # 频率范围掩码

    def forward(self, x):
        # 进行二维离散余弦变换 (DCT)
        x_fft = torch.fft.fft2(x)
        fre_list = []

        for i, mask in enumerate(self.freq_range):
            mask = mask.to(x.device)  # 将掩码转移到相同设备
            fft_sample = x_fft * mask  # 应用掩码
            ifft_sample = torch.fft.ifft2(fft_sample)  # 逆 DCT
            ifft_sample = torch.real(ifft_sample)  # 取实部
            fre_list.append(ifft_sample)

        # 将不同频率的信息拼接
        x_freq = torch.cat(fre_list, dim=1)  # 在通道维度拼接
        return x_freq


# UNet 编码器部分的基本块
def contracting_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )
    return block


# UNet 解码器部分的基本块
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, e, d):
        d = self.up(d)
        diffY = e.size()[2] - d.size()[2]
        diffX = e.size()[3] - d.size()[3]
        e = e[:, :, diffY // 2:e.size()[2] - diffY // 2, diffX // 2:e.size()[3] - diffX // 2]
        cat = torch.cat([e, d], dim=1)  # 在特征通道上拼接
        return self.block(cat)


# VisionMamba 中的 Block 用于 Transformer 层
class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, drop_path=0.):
        super().__init__()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)

    def forward(self, hidden_states, residual=None):
        if residual is None:
            residual = hidden_states
        residual = residual + self.drop_path(hidden_states)
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


# UNet with Transformer
class DCTEnhancedUNet(nn.Module):
    def __init__(self, in_channel, out_channel, freq_range, patch_size=16, embed_dim=192, num_classes=4, depth=24,
                 drop_rate=0.1, **kwargs):
        super(DCTEnhancedUNet, self).__init__()

        self.dct_module = DCTModule(freq_range)  # DCT 模块用于频域特征提取

        # UNet 编码器部分
        self.conv_encode1 = contracting_block(in_channel, 64)
        self.conv_pool1 = nn.MaxPool2d(2)
        self.conv_encode2 = contracting_block(64, 128)
        self.conv_pool2 = nn.MaxPool2d(2)
        self.conv_encode3 = contracting_block(128, 256)
        self.conv_pool3 = nn.MaxPool2d(2)
        self.conv_encode4 = contracting_block(256, 512)
        self.conv_pool4 = nn.MaxPool2d(2)

        # Transformer 层
        self.transformer_blocks = nn.ModuleList(
            [Block(embed_dim, mixer_cls=partial(Mamba, layer_idx=i), drop_path=drop_rate) for i in range(depth)]
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 用于全局特征聚合

        # UNet 解码器部分
        self.conv_decode4 = expansive_block(512, 256, 256)
        self.conv_decode3 = expansive_block(256, 128, 128)
        self.conv_decode2 = expansive_block(128, 64, 64)
        self.conv_decode1 = expansive_block(64, 32, 32)

        # 输出层：分类头
        self.final_layer = nn.Conv2d(32, num_classes, kernel_size=1)

        # 上采样
        self.upsample_final = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def forward(self, x):
        # 1. DCT 频域特征提取
        x_freq = self.dct_module(x)

        # 编码器部分
        encode_block1 = self.conv_encode1(x_freq)
        encode_pool1 = self.conv_pool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_pool4(encode_block4)

        # Transformer 层：处理全局上下文
        x = encode_pool4.flatten(2).transpose(1, 2)  # Flatten for transformer input
        x_cls = self.cls_token.expand(x.shape[0], -1, -1)  # 将 CLS token 扩展为 batch size

        # 拼接 CLS token 和图像特征
        x = torch.cat((x_cls, x), dim=1)

        residual = None
        for transformer in self.transformer_blocks:
            x, residual = transformer(x, residual)

        # 解码器部分
        decode_block4 = self.conv_decode4(encode_block4, x)
        decode_block3 = self.conv_decode3(encode_block3, decode_block4)
        decode_block2 = self.conv_decode2(encode_block2, decode_block3)
        decode_block1 = self.conv_decode1(encode_block1, decode_block2)

        # 最终输出
        output = self.final_layer(decode_block1)
        output = self.upsample_final(output)  # 恢复原始图像尺寸

        return output


# 创建模型实例
if __name__ == '__main__':
    image = torch.rand((1, 1, 256, 256))  # 假设输入为1通道，256x256的图像
    freq_range = mask_generate(256, 4)  # 频域掩码
    model = DCTEnhancedUNet(in_channel=1, out_channel=4, freq_range=freq_range, num_classes=4)
    output = model(image)
    print(f"Output shape: {output.shape}")  # 输出的形状应为 (1, 4, 256, 256)
