import os
import cv2
import numpy as np
import nibabel as nib


def convert_grayscale_to_class(image):
    """
    将灰度值映射为类值。
    """
    # 创建一个新的图像，初始化为0（背景类）
    new_image = np.zeros_like(image)

    # 映射灰度值到类
    # 假设我们将255的灰度值映射为类1，其它灰度值保持为类0
    new_image[image == 255] = 1

    return new_image


def convert_class_to_grayscale(image):
    """
    将类值映射回灰度值。
    """
    # 创建一个新的图像，初始化为0（背景类）
    new_image = np.zeros_like(image)

    # 映射类值回灰度值
    # 假设类1映射回灰度255，类0保持为0

    #
    # new_image[image == 3] = 255
    # new_image[image == 2] = 170
    new_image[image == 1] = 255

    return new_image


def process_png_images(input_folder, output_folder, reverse=False):
    """
    处理PNG图像，将灰度值映射为类别并保存，或将类值映射回灰度值。
    """
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有PNG图像
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg','bmp')):
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 将灰度值转换为类，或将类值转换为灰度
            if reverse:
                converted_image = convert_class_to_grayscale(image)
            else:
                converted_image = convert_grayscale_to_class(image)

            # 保存转换后的图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, converted_image)
            print(f"Processed and saved PNG: {output_path}")


def process_nii_images(input_folder, output_folder, reverse=False):
    """
    处理NIfTI图像，将灰度值映射为类别并保存，或将类值映射回灰度值。
    """
    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有NIfTI图像
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            # 读取NIfTI图像
            image_path = os.path.join(input_folder, filename)
            nii_image = nib.load(image_path)
            image_data = nii_image.get_fdata().astype(np.uint8)  # 读取数据并转换为uint8

            # 将灰度值转换为类，或将类值转换为灰度
            if reverse:
                converted_data = convert_class_to_grayscale(image_data)
            else:
                converted_data = convert_grayscale_to_class(image_data)

            # 保存转换后的NIfTI图像
            output_path = os.path.join(output_folder, filename)
            converted_nii = nib.Nifti1Image(converted_data, nii_image.affine, nii_image.header)
            nib.save(converted_nii, output_path)
            print(f"Processed and saved NIfTI: {output_path}")


def process_images_in_folder(input_folder, output_folder, reverse=False):
    """
    处理文件夹中的所有图像，包括PNG和NIfTI格式。
    """
    process_png_images(input_folder, output_folder, reverse)
    process_nii_images(input_folder, output_folder, reverse)


if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = "/home/jgzn/PycharmProjects/RZ/danzi/seg/datasets/ISIC16/train/label"

    # 输出文件夹路径
    output_folder = "/home/jgzn/PycharmProjects/RZ/danzi/seg/datasets/ISIC16/train/label_ori"

    # 处理图像，将类值映射回灰度值
    process_images_in_folder(input_folder, output_folder, reverse=True)
