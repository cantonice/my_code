import os
import cv2

def convert_images_to_grayscale(scr_dir, tar_dir):
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    for subdir, _, files in os.walk(scr_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                rel_path = os.path.relpath(subdir, scr_dir)
                output_subdir = os.path.join(tar_dir, rel_path)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                gray_img_path = os.path.join(output_subdir, f'gray_{file}')
                cv2.imwrite(gray_img_path, gray_img)


if __name__ == '__main__':
    scr_dir = r'D:\yolo\GBH-YOLOv5\GBH-YOLOv5-glass_substrate\datasets\images'
    tar_dir = r'D:\yolo\GBH-YOLOv5\GBH-YOLOv5-glass_substrate\datasets\images_gray'

    convert_images_to_grayscale(scr_dir, tar_dir)