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
    """基于CLIP的单样本识别器（推荐使用）"""
    
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


# 使用示例
if __name__ == "__main__":
    # 创建识别器
    print("=" * 70)
    print("基于CLIP的单样本图像识别 - 目录搜索模式")
    print("=" * 70)
    
    recognizer = CLIPOneShotRecognizer(model_name='ViT-L/14')
    
    # ========== 配置参数 ==========
    # 方式1: 直接在代码中修改（取消下面的注释并修改路径）
    # 训练图片路径（参考图片）
    train_image_path = "25秋深度学习应用_选课同学照片_54人/刘娅.jpeg"
    
    # 搜索目录（可以修改为你要搜索的目录）
    search_directory = "25秋深度学习应用_选课同学照片_54人/test_data"  # 可以修改这里
    
    
    if not os.path.exists(search_directory):
        print(f"\n搜索目录不存在: {search_directory}")
        search_directory = input("请输入搜索目录路径（或按回车使用默认）: ").strip()
        if not search_directory:
            search_directory = "25秋深度学习应用_选课同学照片_54人/test_data"
    
    # 最多处理的图片数量
    max_images = 100
    
    # 相似度阈值
    threshold = 0.7
    
    # ========== 执行搜索 ==========
    print(f"\n训练图片（参考）: {train_image_path}")
    print(f"搜索目录: {search_directory}")
    print(f"最多处理图片数: {max_images}")
    print(f"相似度阈值: {threshold}")
    print("-" * 70)
    
    try:
        # 在目录中搜索
        results = recognizer.search_in_directory(
            train_image_path=train_image_path,
            search_dir=search_directory,
            max_images=max_images,
            threshold=threshold
        )
        
        if len(results) == 0:
            print("\n没有找到可处理的图片！")
        else:
            # 显示结果
            print("\n" + "=" * 70)
            print("搜索结果（按相似度降序排列）")
            print("=" * 70)
            
            # 显示前10名
            top_n = min(10, len(results))
            print(f"\n前 {top_n} 名最相似的图片：\n")
            
            for i, (img_path, similarity, is_match) in enumerate(results[:top_n], 1):
                status = "✓ 匹配" if is_match else "✗ 不匹配"
                print(f"{i:2d}. [{status}] 相似度: {similarity:.4f}")
                print(f"    路径: {img_path}\n")
            
            # 显示最高分的图片
            if len(results) > 0:
                best_img, best_score, best_match = results[0]
                print("=" * 70)
                print("🏆 最高分图片")
                print("=" * 70)
                print(f"相似度分数: {best_score:.4f}")
                print(f"是否匹配: {'✓ 是' if best_match else '✗ 否'}")
                print(f"图片路径: {best_img}")
                print(f"文件名: {os.path.basename(best_img)}")
                
                # 显示统计信息
                print("\n" + "=" * 70)
                print("统计信息")
                print("=" * 70)
                print(f"总共处理: {len(results)} 张图片")
                print(f"匹配数量: {sum(1 for _, _, match in results if match)} 张")
                print(f"平均相似度: {sum(score for _, score, _ in results) / len(results):.4f}")
                print(f"最高相似度: {best_score:.4f}")
                print(f"最低相似度: {results[-1][1]:.4f}")
    
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n提示: 请检查文件路径是否正确")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("使用说明:")
    print("=" * 70)
    print("""
    1. 修改脚本中的参数：
       - train_image_path: 训练图片（参考图片）路径
       - search_directory: 要搜索的目录路径
       - max_images: 最多处理的图片数量（默认100）
       - threshold: 相似度阈值（默认0.7）
    
    2. 结果说明：
       - 相似度分数范围: 0-1，越高越相似
       - 匹配: 相似度 >= 阈值
       - 结果按相似度降序排列
    
    3. 适用场景：
       - 人脸识别
       - 物体识别
       - 图像检索
       - 相似图片查找
    """)

