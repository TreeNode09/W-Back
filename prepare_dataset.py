"""
将 Out/prc-ed/user_002/user_002 中的图像按 README 要求复制到 WaterLo/Training，
并划分为训练集/验证集，重命名为 train_img01.png、valid_img01.png 等格式。
"""
import os
import random
import shutil

# 路径配置：改 BASE_DIR 即可（项目根目录，如 d:\W）
BASE_DIR = "D:/W"
SOURCE_DIR = os.path.join(BASE_DIR, "Out", "prc-ed", "user_002", "user_002")
TARGET_ROOT = os.path.join(BASE_DIR, "WaterLo", "Training")
TRAIN_DIR = os.path.join(TARGET_ROOT, "train")
VALID_DIR = os.path.join(TARGET_ROOT, "valid")

# 划分比例：训练集比例，其余为验证集
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# 支持的图像扩展名（与 loader.py 一致）
IMAGE_EXTENSIONS = {".jpg", ".png", ".jpeg", ".JPEG"}


def collect_images(src):
    """收集源目录下所有图像文件路径。"""
    paths = []
    for name in os.listdir(src):
        f = os.path.join(src, name)
        if os.path.isfile(f):
            ext = os.path.splitext(name)[1]
            if ext in IMAGE_EXTENSIONS:
                paths.append(f)
    return sorted(paths, key=lambda p: (os.path.splitext(p)[1].lower(), os.path.basename(p)))


def main():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"源目录不存在: {SOURCE_DIR}")

    images = collect_images(SOURCE_DIR)
    if not images:
        raise FileNotFoundError(f"在 {SOURCE_DIR} 中未找到图像文件")

    random.seed(RANDOM_SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_valid = n - n_train
    train_list = images[:n_train]
    valid_list = images[n_train:]

    # 序号宽度：至少 2 位，按数量自动增加（如 7500 张用 4 位）
    width = max(2, len(str(n)))

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALID_DIR, exist_ok=True)

    for i, src_path in enumerate(train_list, start=1):
        ext = os.path.splitext(src_path)[1]
        new_name = f"train_img{i:0{width}d}{ext}"
        dst_path = os.path.join(TRAIN_DIR, new_name)
        shutil.copy2(src_path, dst_path)

    for i, src_path in enumerate(valid_list, start=1):
        ext = os.path.splitext(src_path)[1]
        new_name = f"valid_img{i:0{width}d}{ext}"
        dst_path = os.path.join(VALID_DIR, new_name)
        shutil.copy2(src_path, dst_path)

    ext = os.path.splitext(images[0])[1]
    print(f"完成: 共 {n} 张图像")
    print(f"  训练集: {n_train} 张 -> {TRAIN_DIR}")
    print(f"  验证集: {n_valid} 张 -> {VALID_DIR}")
    print(f"  文件名格式: train_img01{ext}, valid_img01{ext} 等")


if __name__ == "__main__":
    main()
