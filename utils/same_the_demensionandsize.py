import os
from PIL import Image
import nibabel as nib
import numpy as np


def resize_and_convert_images(input_folder, output_folder, target_size=(256, 256), target_channels=3):
    """
    调整输入文件夹及其所有子文件夹中图像的大小和通道，并保持原始格式（PNG 或 NII.GZ）。

    :param input_folder: 包含 PNG 或 NII.GZ 图像的输入文件夹路径。
    :param output_folder: 保存调整后图像的输出文件夹路径。
    :param target_size: 图像的目标大小（宽度, 高度）。
    :param target_channels: 目标通道数（1 表示灰度图，3 表示 RGB）。
    """
    for root, _, files in os.walk(input_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, input_folder)
            output_dir = os.path.join(output_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            if filename.endswith('.png') or filename.endswith('.jpg'):  # 处理 PNG 文件
                output_path = os.path.join(output_dir, filename)

                with Image.open(file_path) as img:
                    # 调整图像大小
                    img = img.resize(target_size, Image.LANCZOS)

                    # 转换图像通道
                    if target_channels == 1:
                        img = img.convert('L')
                    elif target_channels == 3:
                        img = img.convert('RGB')

                    # 保存为 PNG 文件
                    img.save(output_path)
                    print(f'已保存: {output_path}')

            elif filename.endswith('.nii.gz'):  # 处理 NII.GZ 文件
                output_path = os.path.join(output_dir, filename)

                # 读取 NII.GZ 文件
                nii_data = nib.load(file_path)
                image_data = nii_data.get_fdata()

                # 假设 NII 文件只包含单张切片或多切片
                if len(image_data.shape) == 3:
                    resized_data = []
                    for slice_idx in range(image_data.shape[2]):
                        slice_data = image_data[:, :, slice_idx]

                        # 归一化切片数据到 [0, 255]
                        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255.0
                        slice_data = slice_data.astype(np.uint8)

                        # 转换为 PIL 图像，调整大小
                        img = Image.fromarray(slice_data).resize(target_size, Image.LANCZOS)

                        # 转换图像通道
                        if target_channels == 1:
                            img = img.convert('L')
                        elif target_channels == 3:
                            img = img.convert('RGB')

                        # 转回 numpy 格式
                        resized_data.append(np.array(img))

                    # 堆叠切片回到 3D 格式
                    resized_data = np.stack(resized_data, axis=-1)

                elif len(image_data.shape) == 2:
                    # 如果是 2D 图像
                    slice_data = image_data
                    slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255.0
                    slice_data = slice_data.astype(np.uint8)

                    img = Image.fromarray(slice_data).resize(target_size, Image.LANCZOS)

                    if target_channels == 1:
                        img = img.convert('L')
                    elif target_channels == 3:
                        img = img.convert('RGB')

                    resized_data = np.array(img)

                else:
                    raise ValueError(f"Unsupported NII data shape: {image_data.shape}")

                # 创建新的 NII 对象并保存
                resized_nii = nib.Nifti1Image(resized_data, affine=nii_data.affine)
                nib.save(resized_nii, output_path)
                print(f'已保存: {output_path}')





if __name__ == '__main__':
    # 输入和输出文件夹
    input_folder = "/home/jgzn/PycharmProjects/RZ/danzi/seg/datasets/ISIC/train/image"
    output_folder = "/home/jgzn/PycharmProjects/RZ/danzi/seg/datasets/ISIC/train/image"

    # 调用函数，调整图像大小和通道
    resize_and_convert_images(input_folder, output_folder, target_size=(256, 256), target_channels=1)
