"""
    检查图片尺寸是否为640*640
"""

import os
from PIL import Image


def check_image_size(path):
    if not os.path.exists(path):
        return "Path does not exist."

    invalid_files = []

    for filename in os.listdir(path):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(path, filename)
            with Image.open(file_path) as img:
                width, height = img.size
                if width != 640 or height != 640:
                    invalid_files.append(filename)

    return invalid_files


if __name__ == "__main__":
    path = r"E:\BaiduNetdiskDownload\image_and_label_are_corresponding\images"
    check_image_size(path)
    invalid_files_namelist = check_image_size(path)
    if not invalid_files_namelist:
        print("未有640以外的size")
    else:
        for name in invalid_files_namelist:
            print(name)

