"""
    用作根据已有的标签来在images的文件夹中挑选对应的图片出来
    单独做一个图片数据集
"""

import os
import shutil


def copy_images_with_labels(labels_path, src_images_path, tar_images_path):
    # 获取所有txt文件名
    txt_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

    # 提取文件名（不带扩展名）
    image_names = [os.path.splitext(f)[0] for f in txt_files]

    # 复制文件到指定目录
    for name in image_names:
        src_file = os.path.join(src_images_path, name + '.jpg')  # 假设图片格式为jpg
        if os.path.exists(src_file):
            shutil.copy(src_file, tar_images_path)


if __name__ == "__main__":
    labels_path = r"E:\BaiduNetdiskDownload\image_and_label_are_corresponding\label"
    src_images_path = r"E:\BaiduNetdiskDownload\裁切后的\data_images"
    tar_images_path = r"E:\BaiduNetdiskDownload\image_and_label_are_corresponding\picked_images"
    copy_images_with_labels(labels_path, src_images_path, tar_images_path)
