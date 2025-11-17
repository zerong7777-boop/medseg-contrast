import os
import numpy as np
from PIL import Image


def preprocess_labels_in_folder(labels_folder):
    """
    遍历文件夹中的标签图像，确保标签值只有 0 和 255。
    如果标签值为其他值，改为 255。

    :param labels_folder: 标签图像所在的文件夹路径
    """
    # 遍历文件夹中的所有 PNG 文件
    label_files = [f for f in os.listdir(labels_folder) if f.endswith(".png")]

    for label_file in label_files:
        # 构造标签文件的完整路径
        label_path = os.path.join(labels_folder, label_file)

        # 使用 PIL 打开标签图像
        label_image = Image.open(label_path)
        label_array = np.array(label_image)

        # 检查并修正标签值：将非 0 和 255 的像素值设置为 255
        # np.where() 是根据条件进行元素替换的函数
        corrected_label_array = np.where(label_array > 0, 255, 0)  # 将大于 0 的值设为 255，其他值设为 0

        # 检查修正后的标签图像是否与原始图像不同
        if not np.array_equal(label_array, corrected_label_array):
            print(f"修正标签图像: {label_file}")
            # 保存修正后的标签图像
            corrected_label_image = Image.fromarray(corrected_label_array.astype(np.uint8))
            corrected_label_image.save(label_path)  # 直接覆盖原文件
        else:
            print(f"标签图像 {label_file} 没有需要修正的值。")

    print("所有标签图像已检查并修正（如有必要）。")


# 示例用法
labels_folder_path = "/home/jgzn/PycharmProjects/RZ/datadata/kvasir-seg/Kvasir-SEG/deal_myself/train/label"  # 替换为您的标签文件夹路径
preprocess_labels_in_folder(labels_folder_path)
