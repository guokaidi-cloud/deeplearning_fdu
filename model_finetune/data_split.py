# YOLO 数据集划分工具
# 功能：将包含图片和对应标注文件的数据集按比例随机划分为训练集、验证集和测试集
# 同时生成YOLO训练所需的dataset.yaml配置文件
 
import os
import shutil
import random
import yaml
 
# ============ 配置参数 ============
SOURCE_DIR = "data"                 # 源数据目录，图片及其标注 .txt 文件应在此目录
OUTPUT_DIR = "data_split_output2"    # 输出根目录，将生成 train/val/test 及其 images/labels
 
TRAIN_RATIO = 0.7                   # 训练集比例
VAL_RATIO = 0.2                     # 验证集比例
TEST_RATIO = 0.1                    # 测试集比例
SEED = 42                           # 随机种子，确保每次划分结果一致
REQUIRE_LABEL = True                # 仅包含存在对应 .txt 标注的图片
 
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}  # 支持的图片格式
# ==========================================
 
 
def ensure_dir(path: str):
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)
 
 
def main():
    # 检查源目录是否存在
    if not os.path.isdir(SOURCE_DIR):
        print(f"源目录不存在: {SOURCE_DIR}")
        return
 
    # 收集图片及对应标注信息
    candidates = []
    for fname in os.listdir(SOURCE_DIR):
        fpath = os.path.join(SOURCE_DIR, fname)
        # 跳过非文件项
        if not os.path.isfile(fpath):
            continue
         
        # 检查文件是否为图片格式
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMG_EXTS:
            base = os.path.splitext(fname)[0]  # 获取文件名（不含扩展名）
            label_name = base + ".txt"         # 对应的标注文件名
            label_path = os.path.join(SOURCE_DIR, label_name)
 
            # 如果需要标注文件且不存在，则跳过此图片
            if REQUIRE_LABEL and not os.path.exists(label_path):
                continue
 
            # 记录图片文件名和对应的标注文件名（如果有）
            has_label = os.path.exists(label_path)
            candidates.append((fname, label_name if has_label else None))
 
    n = len(candidates)
    if n == 0:
        print("未找到符合条件的图片及标注对。")
        return
 
    # 检查比例设置是否有效
    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if ratio_sum <= 0:
        print("train/val/test 比例之和必须大于0")
        return
 
    # 设置随机种子并打乱数据顺序
    random.seed(SEED)
    random.shuffle(candidates)
 
    # 计算各数据集的数量
    n_train = int(n * (TRAIN_RATIO / ratio_sum))
    n_val = int(n * (VAL_RATIO / ratio_sum))
    n_test = n - n_train - n_val
 
    # 定义输出目录路径
    train_img_out = os.path.join(OUTPUT_DIR, "train", "images")
    train_lab_out = os.path.join(OUTPUT_DIR, "train", "labels")
    val_img_out = os.path.join(OUTPUT_DIR, "val", "images")
    val_lab_out = os.path.join(OUTPUT_DIR, "val", "labels")
    test_img_out = os.path.join(OUTPUT_DIR, "test", "images")
    test_lab_out = os.path.join(OUTPUT_DIR, "test", "labels")
 
    # 创建所有输出目录
    for d in [train_img_out, train_lab_out, val_img_out, val_lab_out, test_img_out, test_lab_out]:
        ensure_dir(d)
 
    def copy_pair(item, dst_img_dir, dst_lab_dir):
        """复制图片和对应的标注文件到目标目录"""
        img_name, label_name = item
        src_img = os.path.join(SOURCE_DIR, img_name)
        dst_img = os.path.join(dst_img_dir, img_name)
        shutil.copy2(src_img, dst_img)  # 复制图片文件
 
        # 如果存在标注文件，则一并复制
        if label_name:
            src_lab = os.path.join(SOURCE_DIR, label_name)
            if os.path.exists(src_lab):
                dst_lab = os.path.join(dst_lab_dir, label_name)
                shutil.copy2(src_lab, dst_lab)  # 复制标注文件
 
    # 按划分结果复制文件到对应目录
    idx = 0
    for _ in range(n_train):
        copy_pair(candidates[idx], train_img_out, train_lab_out)
        idx += 1
    for _ in range(n_val):
        copy_pair(candidates[idx], val_img_out, val_lab_out)
        idx += 1
    for _ in range(n_test):
        copy_pair(candidates[idx], test_img_out, test_lab_out)
        idx += 1
 
    # 读取类别信息
    classes_file = os.path.join(SOURCE_DIR, 'classes.txt')
    class_names = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        print(f"警告: 未找到 classes.txt 文件，使用默认类别")
        class_names = ['class0', 'class1', 'class2']  # 默认类别名
     
    # 生成 YAML 配置文件 (手动控制顺序)
    yaml_path = os.path.join(OUTPUT_DIR, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        # 写入数据集路径
        f.write(f"train: ./train/images\n")
        f.write(f"val: ./val/images\n")
        f.write(f"test: ./test/images\n")
        f.write(f"\n")
        # 写入类别数量
        f.write(f"nc: {len(class_names)}\n")
        # 写入类别名称列表
        f.write(f"names:\n")
        for name in class_names:
            f.write(f"- {name}\n")
     
    # 输出划分结果摘要
    print("划分完成：")
    print(f"  训练集: train/images={train_img_out}, train/labels={train_lab_out}, 张数={n_train}")
    print(f"  验证集:  val/images={val_img_out},   val/labels={val_lab_out}, 张数={n_val}")
    print(f"  测试集:  test/images={test_img_out},  test/labels={test_lab_out}, 张数={n_test}")
    print(f"  配置文件: {yaml_path}")
 
 
if __name__ == "__main__":
    main()
 
 
SOURCE_DIR = "data"                 # 源数据目录，图片及其标注 .txt 文件应在此目录
OUTPUT_DIR = "data_split_output2"    # 输出根目录，将生成 train/val/test 及其 images/labels
 
TRAIN_RATIO = 0.7                   # 训练集比例
VAL_RATIO = 0.2                     # 验证集比例
TEST_RATIO = 0.1                    # 测试集比例