from PIL import Image
import os
import numpy as np
import nibabel as nib  # 导入 nibabel 用于读取 .nii.gz 文件


# 定义一个函数，用于检查文件夹中的图片标签（灰度值）
def check_image_labels(folder_path):
    """
    检查指定文件夹中的图片标签值（灰度值），并输出每张图片中出现的灰度值。

    参数:
    folder_path (str): 包含图片的文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 判断文件格式
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 打开图片并转换为灰度模式（如果图片不是灰度模式）
            img = Image.open(file_path).convert('L')

            # 将图片转换为NumPy数组
            img_array = np.array(img)

            # 获取图片中的唯一灰度值
            unique_values = np.unique(img_array)

            # 输出图片名和灰度值
            print(f"图片: {filename}，标签值: {unique_values}")

        elif filename.endswith('.nii.gz'):
            # 使用 nibabel 读取 .nii.gz 文件
            nii_img = nib.load(file_path)
            img_array = nii_img.get_fdata().astype(np.int32)  # 转换为整数类型

            # 获取图片中的唯一灰度值
            unique_values = np.unique(img_array)

            # 输出图片名和灰度值
            print(f"图片: {filename}，标签值: {unique_values}")


# 示例调用，假设图片文件夹路径为 "image_folder"
folder_path = "/home/jgzn/PycharmProjects/RZ/danzi/seg/datasets/ISIC16/train/label"  # 替换为你的图片文件夹路径
check_image_labels(folder_path)
