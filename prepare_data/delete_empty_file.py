"""
    用作刚打完标签的数据集，可能有些是空的--0kb,那么就将他们删除了
"""
import os


def delete_empty_txt_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)


if __name__ == "__main__":
    path = r"D:\datasets\label"
    delete_empty_txt_files(path)

