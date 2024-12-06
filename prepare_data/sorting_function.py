import os
import re


def natural_sort_key(s):
    """用于自然排序的键函数"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def rename_files(directory):
    """重命名指定目录下的所有文件"""
    # 获取目录下的所有文件
    files = os.listdir(directory)
    # 按照自然排序顺序对文件进行排序
    files.sort(key=natural_sort_key)

    # 重命名文件
    for index, filename in enumerate(files):
        # 构建新的文件名，例如：file1.txt, file2.txt, ...
        new_filename = f"img_{index + 1}{os.path.splitext(filename)[1]}"
        # 构建完整的文件路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_filename}'")


# 使用示例
directory_path = r'E:\BaiduNetdiskDownload\image_and_label_are_corresponding\label'  # 替换为你的目录路径
rename_files(directory_path)