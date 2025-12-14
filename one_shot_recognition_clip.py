"""
单样本图像识别 - 使用CLIP预训练模型（推荐方法）
CLIP模型已经在大量数据上训练，可以直接用于图像相似度比较
"""

import torch
import torch.nn.functional as F
from PIL import Image
import clip
import os
import glob
from pathlib import Path

# 尝试导入tqdm，如果没有则使用简单的进度显示
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        """简单的进度条替代"""
        print(f"{desc}...")
        return iterable


class CLIPOneShotRecognizer:
    """基于CLIP的单样本/多样本识别器（推荐使用）"""
    
    def __init__(self, model_name='ViT-B/32'):
        """
        初始化CLIP模型
        
        Args:
            model_name: CLIP模型名称，可选：
                - 'ViT-B/32' (推荐，速度快)
                - 'ViT-B/16' (精度更高)
                - 'ViT-L/14' (精度最高，但更慢)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        print(f"正在加载CLIP模型: {model_name}...")
        
        # 加载CLIP模型和预处理
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # 存储训练后的特征
        self.trained_features = None
        self.train_image_paths = []
        
        # 多类别训练的特征存储
        self.class_features = {}  # {类别名: 特征向量}
        self.class_image_counts = {}  # {类别名: 图片数量}
        
        print("CLIP模型加载完成！")
    
    def extract_features(self, image_path):
        """
        提取图片特征
        
        Args:
            image_path: 图片路径
        
        Returns:
            torch.Tensor: 特征向量
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 提取图像特征
            image_features = self.model.encode_image(image_tensor)
            # 归一化特征向量
            image_features = F.normalize(image_features, dim=1)
        
        return image_features
    
    def recognize(self, train_image_path, test_image_path, threshold=0.7):
        """
        识别测试图片是否与训练图片匹配
        
        Args:
            train_image_path: 训练图片路径
            test_image_path: 测试图片路径
            threshold: 相似度阈值（0-1之间，默认0.7）
        
        Returns:
            bool: 是否匹配
            float: 相似度分数（余弦相似度，范围0-1）
        """
        if not os.path.exists(train_image_path):
            raise FileNotFoundError(f"训练图片不存在: {train_image_path}")
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"测试图片不存在: {test_image_path}")
        
        # 提取特征
        train_features = self.extract_features(train_image_path)
        test_features = self.extract_features(test_image_path)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(train_features, test_features).item()
        
        # 判断是否匹配
        is_match = similarity >= threshold
        
        return is_match, similarity
    
    def recognize_batch(self, train_image_path, test_image_paths, threshold=0.7):
        """
        批量识别多张测试图片
        
        Args:
            train_image_path: 训练图片路径
            test_image_paths: 测试图片路径列表
            threshold: 相似度阈值
        
        Returns:
            list: [(是否匹配, 相似度分数), ...]
        """
        train_features = self.extract_features(train_image_path)
        results = []
        
        for test_path in test_image_paths:
            if not os.path.exists(test_path):
                results.append((False, 0.0))
                continue
            
            test_features = self.extract_features(test_path)
            similarity = F.cosine_similarity(train_features, test_features).item()
            is_match = similarity >= threshold
            results.append((is_match, similarity))
        
        return results
    
    def train_from_images(self, train_image_paths, aggregation='mean'):
        """
        使用多张图片进行训练，提取并聚合特征
        
        Args:
            train_image_paths: 训练图片路径列表
            aggregation: 特征聚合方式，可选：
                - 'mean': 平均特征（默认，推荐）
                - 'max': 最大特征
                - 'all': 保留所有特征（匹配时取最高相似度）
        
        Returns:
            int: 成功处理的图片数量
        """
        if not train_image_paths:
            raise ValueError("训练图片列表不能为空")
        
        print(f"\n正在从 {len(train_image_paths)} 张图片提取特征...")
        
        features_list = []
        valid_paths = []
        
        for img_path in tqdm(train_image_paths, desc="提取特征"):
            if not os.path.exists(img_path):
                print(f"\n警告: 图片不存在，跳过: {img_path}")
                continue
            try:
                features = self.extract_features(img_path)
                features_list.append(features)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"\n警告: 处理失败，跳过 {img_path}: {str(e)}")
                continue
        
        if len(features_list) == 0:
            raise ValueError("没有成功提取任何特征！请检查图片路径。")
        
        self.train_image_paths = valid_paths
        
        # 聚合特征
        if aggregation == 'mean':
            # 计算平均特征
            all_features = torch.cat(features_list, dim=0)
            self.trained_features = F.normalize(all_features.mean(dim=0, keepdim=True), dim=1)
            print(f"\n已使用平均聚合方式训练，共 {len(valid_paths)} 张图片")
        elif aggregation == 'max':
            # 计算最大特征
            all_features = torch.cat(features_list, dim=0)
            self.trained_features = F.normalize(all_features.max(dim=0, keepdim=True)[0], dim=1)
            print(f"\n已使用最大聚合方式训练，共 {len(valid_paths)} 张图片")
        elif aggregation == 'all':
            # 保留所有特征
            self.trained_features = torch.cat(features_list, dim=0)
            print(f"\n已保留所有特征，共 {len(valid_paths)} 张图片")
        else:
            raise ValueError(f"不支持的聚合方式: {aggregation}")
        
        return len(valid_paths)
    
    def train_from_directory(self, train_dir, aggregation='mean', max_images=None):
        """
        从目录加载图片进行训练
        
        Args:
            train_dir: 训练图片目录路径
            aggregation: 特征聚合方式 ('mean', 'max', 'all')
            max_images: 最多使用的图片数量（None表示全部）
        
        Returns:
            int: 成功处理的图片数量
        """
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"训练目录不存在: {train_dir}")
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG']
        
        # 获取目录中所有图片（递归）
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(train_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(train_dir, '**', ext), recursive=True))
        
        # 去重并排序
        image_paths = sorted(list(set(image_paths)))
        
        if len(image_paths) == 0:
            raise ValueError(f"目录中没有找到图片: {train_dir}")
        
        # 限制数量
        if max_images and len(image_paths) > max_images:
            print(f"找到 {len(image_paths)} 张图片，将使用前 {max_images} 张进行训练")
            image_paths = image_paths[:max_images]
        else:
            print(f"找到 {len(image_paths)} 张图片，将全部用于训练")
        
        return self.train_from_images(image_paths, aggregation=aggregation)
    
    def match_single_image(self, test_image_path, threshold=0.7):
        """
        使用单张测试图片与训练特征进行匹配
        
        Args:
            test_image_path: 测试图片路径
            threshold: 相似度阈值
        
        Returns:
            dict: {
                'is_match': 是否匹配,
                'similarity': 相似度分数,
                'max_similarity': 最高相似度（仅当使用'all'聚合时有意义）,
                'min_similarity': 最低相似度,
                'avg_similarity': 平均相似度
            }
        """
        if self.trained_features is None:
            raise ValueError("请先调用 train_from_images 或 train_from_directory 进行训练！")
        
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"测试图片不存在: {test_image_path}")
        
        # 提取测试图片特征
        test_features = self.extract_features(test_image_path)
        
        # 计算与所有训练特征的相似度
        similarities = F.cosine_similarity(test_features, self.trained_features)
        
        # 统计信息
        max_sim = similarities.max().item()
        min_sim = similarities.min().item()
        avg_sim = similarities.mean().item()
        
        # 使用最高相似度判断是否匹配
        is_match = max_sim >= threshold
        
        return {
            'is_match': is_match,
            'similarity': max_sim,  # 主要相似度分数
            'max_similarity': max_sim,
            'min_similarity': min_sim,
            'avg_similarity': avg_sim
        }
    
    def train_multi_class(self, train_dir, aggregation='mean', max_images_per_class=None):
        """
        多类别训练：目录下的子目录是标签，子目录下的图片是数据
        
        目录结构示例:
            train_dir/
            ├── 人物A/           <- 这是标签
            │   ├── photo1.jpg
            │   ├── photo2.jpg
            │   └── ...
            ├── 人物B/           <- 这是标签
            │   ├── photo1.jpg
            │   └── ...
            └── 人物C/
                └── ...
        
        Args:
            train_dir: 训练目录路径
            aggregation: 特征聚合方式 ('mean', 'max')
            max_images_per_class: 每个类别最多使用的图片数量（None表示全部）
        
        Returns:
            dict: {类别名: 训练图片数量}
        """
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"训练目录不存在: {train_dir}")
        
        # 获取所有子目录作为类别
        subdirs = [d for d in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, d))]
        
        if len(subdirs) == 0:
            raise ValueError(f"训练目录下没有找到子目录（类别）: {train_dir}")
        
        print(f"\n找到 {len(subdirs)} 个类别: {subdirs}")
        print("-" * 70)
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', 
                           '*.JPG', '*.JPEG', '*.PNG']
        
        self.class_features = {}
        self.class_image_counts = {}
        
        for class_name in tqdm(subdirs, desc="训练类别"):
            class_dir = os.path.join(train_dir, class_name)
            
            # 获取该类别下的所有图片
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(class_dir, ext)))
            
            # 去重并排序
            image_paths = sorted(list(set(image_paths)))
            
            if len(image_paths) == 0:
                print(f"\n警告: 类别 '{class_name}' 下没有找到图片，跳过")
                continue
            
            # 限制每个类别的图片数量
            if max_images_per_class and len(image_paths) > max_images_per_class:
                image_paths = image_paths[:max_images_per_class]
            
            # 提取该类别所有图片的特征
            features_list = []
            for img_path in image_paths:
                try:
                    features = self.extract_features(img_path)
                    features_list.append(features)
                except Exception as e:
                    print(f"\n警告: 处理失败 {img_path}: {str(e)}")
                    continue
            
            if len(features_list) == 0:
                print(f"\n警告: 类别 '{class_name}' 没有成功提取任何特征，跳过")
                continue
            
            # 聚合特征
            all_features = torch.cat(features_list, dim=0)
            if aggregation == 'mean':
                class_feature = F.normalize(all_features.mean(dim=0, keepdim=True), dim=1)
            elif aggregation == 'max':
                class_feature = F.normalize(all_features.max(dim=0, keepdim=True)[0], dim=1)
            else:
                raise ValueError(f"不支持的聚合方式: {aggregation}")
            
            self.class_features[class_name] = class_feature
            self.class_image_counts[class_name] = len(features_list)
        
        # 打印训练结果
        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)
        print(f"总类别数: {len(self.class_features)}")
        for class_name, count in self.class_image_counts.items():
            print(f"  - {class_name}: {count} 张图片")
        
        return self.class_image_counts
    
    def classify_image(self, test_image_path, top_k=5):
        """
        对测试图片进行分类，找出最像哪个类别
        
        Args:
            test_image_path: 测试图片路径
            top_k: 返回前k个最相似的类别
        
        Returns:
            list: [(类别名, 相似度), ...] 按相似度降序排列
        """
        if len(self.class_features) == 0:
            raise ValueError("请先调用 train_multi_class 进行多类别训练！")
        
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"测试图片不存在: {test_image_path}")
        
        # 提取测试图片特征
        test_features = self.extract_features(test_image_path)
        
        # 计算与所有类别的相似度
        results = []
        for class_name, class_feature in self.class_features.items():
            similarity = F.cosine_similarity(test_features, class_feature).item()
            results.append((class_name, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个
        return results[:top_k]
    
    def classify_batch(self, test_image_paths, top_k=1):
        """
        批量分类多张测试图片
        
        Args:
            test_image_paths: 测试图片路径列表
            top_k: 每张图片返回前k个最相似的类别
        
        Returns:
            list: [
                {
                    'image_path': 图片路径,
                    'predictions': [(类别名, 相似度), ...]
                },
                ...
            ]
        """
        if len(self.class_features) == 0:
            raise ValueError("请先调用 train_multi_class 进行多类别训练！")
        
        results = []
        for img_path in tqdm(test_image_paths, desc="分类进度"):
            try:
                predictions = self.classify_image(img_path, top_k=top_k)
                results.append({
                    'image_path': img_path,
                    'predictions': predictions
                })
            except Exception as e:
                print(f"\n处理失败 {img_path}: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'predictions': []
                })
        
        return results

    def search_in_directory(self, train_image_path, search_dir, max_images=100, threshold=0.7):
        """
        在目录中搜索与训练图片最相似的图片
        
        Args:
            train_image_path: 训练图片路径
            search_dir: 搜索目录路径
            max_images: 最多处理的图片数量（默认100）
            threshold: 相似度阈值（仅用于标记是否匹配）
        
        Returns:
            list: [(图片路径, 相似度分数, 是否匹配), ...] 按相似度降序排列
        """
        if not os.path.exists(train_image_path):
            raise FileNotFoundError(f"训练图片不存在: {train_image_path}")
        if not os.path.exists(search_dir):
            raise FileNotFoundError(f"搜索目录不存在: {search_dir}")
        
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG']
        
        # 获取目录中所有图片
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(search_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(search_dir, '**', ext), recursive=True))
        
        # 去重并排序
        image_paths = sorted(list(set(image_paths)))
        
        # 限制数量
        total_found = len(image_paths)
        if total_found > max_images:
            image_paths = image_paths[:max_images]
            print(f"找到 {total_found} 张图片，将处理前 {max_images} 张")
        else:
            print(f"找到 {total_found} 张图片，将全部处理")
        
        if len(image_paths) == 0:
            print("目录中没有找到图片文件！")
            return []
        
        # 提取训练图片特征
        print(f"\n正在提取训练图片特征: {train_image_path}")
        train_features = self.extract_features(train_image_path)
        
        # 批量处理图片
        print(f"\n正在处理 {len(image_paths)} 张图片...")
        results = []
        
        for img_path in tqdm(image_paths, desc="处理进度"):
            try:
                test_features = self.extract_features(img_path)
                similarity = F.cosine_similarity(train_features, test_features).item()
                is_match = similarity >= threshold
                results.append((img_path, similarity, is_match))
            except Exception as e:
                print(f"\n处理图片失败 {img_path}: {str(e)}")
                continue
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

def demo_multi_class_classification():
    """
    多类别训练 + 分类匹配 的使用示例
    
    目录结构:
        train_dir/
        ├── 人物A/           <- 子目录名就是标签
        │   ├── photo1.jpg
        │   └── photo2.jpg
        ├── 人物B/
        │   └── photo1.jpg
        └── 人物C/
            └── ...
    
    然后用一张测试图片，找出它最像哪个类别（人物）
    """
    print("=" * 70)
    print("多类别训练 + 分类匹配 模式")
    print("=" * 70)
    
    # 创建识别器
    recognizer = CLIPOneShotRecognizer(model_name='ViT-L/14')
    
    # ========== 配置参数 ==========
    # 训练目录（子目录是标签，子目录下的图片是数据）
    train_directory = "classmate_photo_processed/"  # 修改为你的训练目录
    
    # 测试图片路径（要分类的单张图片）
    test_image_path = "frame_261436_id_0028_n_0006.jpg"  # 修改为你的测试图片
    
    # 特征聚合方式：'mean'（平均，推荐）, 'max'（最大）
    aggregation = 'mean'
    
    # 显示前几个最相似的类别
    top_k = 5
    
    print(f"\n训练目录: {train_directory}")
    print(f"测试图片: {test_image_path}")
    print(f"聚合方式: {aggregation}")
    print("-" * 70)
    
    try:
        # ========== 步骤1: 多类别训练 ==========
        print("\n【步骤1】多类别训练...")
        class_counts = recognizer.train_multi_class(
            train_dir=train_directory,
            aggregation=aggregation,
            max_images_per_class=50  # 每个类别最多使用50张图片
        )
        
        # ========== 步骤2: 分类测试图片 ==========
        print(f"\n【步骤2】正在分类测试图片: {test_image_path}")
        predictions = recognizer.classify_image(
            test_image_path=test_image_path,
            top_k=top_k
        )
        
        # 显示结果
        print("\n" + "=" * 70)
        print("🎯 分类结果（按相似度降序）")
        print("=" * 70)
        
        for i, (class_name, similarity) in enumerate(predictions, 1):
            bar_len = int(similarity * 30)  # 相似度可视化条
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"{i}. {class_name}")
            print(f"   相似度: {similarity:.4f} [{bar}]")
            print()
        
        # 最终预测
        best_class, best_score = predictions[0]
        print("=" * 70)
        print(f"🏆 最终预测: {best_class}")
        print(f"   置信度: {best_score:.4f}")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("使用说明（多类别分类模式）:")
    print("=" * 70)
    print("""
    1. 准备训练数据（目录结构）：
       train_dir/
       ├── 类别A/        <- 子目录名 = 标签名
       │   ├── img1.jpg  <- 该类别的训练图片
       │   └── img2.jpg
       ├── 类别B/
       │   └── ...
       └── 类别C/
           └── ...
    
    2. 调用方法：
       - train_multi_class(train_dir): 多类别训练
       - classify_image(test_path): 分类单张图片
       - classify_batch(test_paths): 批量分类多张图片
    
    3. 适用场景：
       - 人脸识别（每个人一个文件夹）
       - 物体分类（每类物体一个文件夹）
       - 图像检索（找出最相似的类别）
    """)


# 主入口
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("基于CLIP的图像识别系统")
    print("=" * 70)
    print("""
    可用模式:
    1. 多类别分类模式 (multi_class) - 目录下的子目录是标签，测试图片匹配最像的类别
    2. 单样本搜索模式 (search) - 用一张图片在目录中搜索相似图片
    
    使用方法:
        python one_shot_recognition_clip.py              # 默认使用多类别分类模式
        python one_shot_recognition_clip.py multi_class  # 多类别分类模式
        python one_shot_recognition_clip.py search       # 单样本搜索模式
    """)
    
    # 默认使用多类别分类模式
    mode = "multi_class"
    
    # 从命令行参数获取模式（可选）
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    if mode == "multi_class":
        # 运行多类别分类示例
        demo_multi_class_classification()
    else:
        print(f"未知模式: {mode}")
        print("请使用: python one_shot_recognition_clip.py [multi_class|search]")

