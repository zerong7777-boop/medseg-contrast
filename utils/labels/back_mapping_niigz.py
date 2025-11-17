import os
import numpy as np
import nibabel as nib
import cv2


def convert_class_to_grayscale(label_image):
    """
    将标签图像中的类映射回灰度值
    类 0 -> 灰度 0 (背景)
    类 1 -> 灰度 85
    类 2 -> 灰度 170
    类 3 -> 灰度 255
    """
    grayscale_image = np.zeros_like(label_image, dtype=np.uint8)

    grayscale_image[label_image == 1] = 85
    grayscale_image[label_image == 2] = 170
    grayscale_image[label_image == 3] = 255

    return grayscale_image


def process_nii_files(input_folder, output_folder):
    """
    处理文件夹中的 .nii.gz 标签文件，转换为灰度图像并保存为 .png 格式
    """
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有nii.gz文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            # 读取nii.gz文件
            file_path = os.path.join(input_folder, filename)
            nii_image = nib.load(file_path)
            label_data = nii_image.get_fdata()  # 获取标签数据

            # 假设标签数据是二维的，如果是三维的，可以进行适当的处理
            label_data_2d = label_data.squeeze()  # 如果有多维数据，压缩维度

            # 将标签数据转换为灰度图像
            grayscale_image = convert_class_to_grayscale(label_data_2d)

            # 保存为PNG格式
            output_path = os.path.join(output_folder, filename.replace(".nii.gz", ".png"))
            cv2.imwrite(output_path, grayscale_image)
            print(f"Processed and saved: {output_path}")


if __name__ == "__main__":
    # 输入文件夹路径，包含nii.gz文件
    input_folder = "/home/jgzn/PycharmProjects/RZ/SEG/U-Mamba-main/data/nnUNet_raw/Dataset705_mouse/Umamba_predicted"

    # 输出文件夹路径，保存PNG图像
    output_folder = "/home/jgzn/PycharmProjects/RZ/SEG/U-Mamba-main/data/nnUNet_raw/Dataset705_mouse/Umamba_predicted_png"

    # 处理nii.gz文件并保存为PNG
    process_nii_files(input_folder, output_folder)
