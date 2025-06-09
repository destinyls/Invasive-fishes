#!/usr/bin/env python3
"""
图片饱和度增强使用示例
演示如何使用 ImageSaturationAdjuster 类
"""

import cv2
import numpy as np
from image_saturation import ImageSaturationAdjuster
import os


def demo_single_image():
    """演示单张图片饱和度增强"""
    print("=== 单张图片饱和度增强演示 ===")
    
    # 创建饱和度调整器
    adjuster = ImageSaturationAdjuster()
    
    # 使用项目中已有的示例图片
    demo_image_path = "demo.jpg"
    
    if not os.path.exists(demo_image_path):
        print(f"演示图片不存在: {demo_image_path}")
        print("请确保 demo.jpg 文件存在于当前目录")
        return
    
    # 处理图片
    try:
        # 增强饱和度 (因子为1.8)
        enhanced_path = adjuster.process_image_file(
            demo_image_path, 
            "demo_enhanced.jpg", 
            saturation_factor=1.8, 
            quality=95
        )
        
        # 创建对比图
        comparison_path = adjuster.create_comparison_image(
            demo_image_path, 
            enhanced_path
        )
        
        print(f"✓ 原图: {demo_image_path}")
        print(f"✓ 增强图: {enhanced_path}")
        print(f"✓ 对比图: {comparison_path}")
        
    except Exception as e:
        print(f"处理失败: {e}")


def demo_different_saturation_levels():
    """演示不同饱和度级别的效果"""
    print("\n=== 不同饱和度级别演示 ===")
    
    adjuster = ImageSaturationAdjuster()
    demo_image_path = "demo.jpg"
    
    if not os.path.exists(demo_image_path):
        print(f"演示图片不存在: {demo_image_path}")
        return
    
    # 不同的饱和度级别
    saturation_levels = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    print("生成不同饱和度级别的图片:")
    for level in saturation_levels:
        try:
            output_path = f"demo_saturation_{level:.1f}.jpg"
            adjuster.process_image_file(
                demo_image_path,
                output_path,
                saturation_factor=level,
                quality=95
            )
            print(f"✓ 饱和度 {level:.1f}: {output_path}")
        except Exception as e:
            print(f"✗ 饱和度 {level:.1f} 处理失败: {e}")


def demo_with_mask():
    """演示使用掩码的选择性饱和度增强"""
    print("\n=== 掩码选择性增强演示 ===")
    
    adjuster = ImageSaturationAdjuster()
    demo_image_path = "demo.jpg"
    
    if not os.path.exists(demo_image_path):
        print(f"演示图片不存在: {demo_image_path}")
        return
    
    # 读取图片
    image = cv2.imread(demo_image_path)
    if image is None:
        print("无法读取演示图片")
        return
    
    # 创建一个简单的圆形掩码（中心区域）
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w//2, h//2)
    radius = min(w, h) // 4
    cv2.circle(mask, center, radius, 255, -1)
    
    # 使用掩码进行选择性增强
    enhanced_image = adjuster.enhance_saturation_with_mask(
        image, mask, saturation_factor=2.0
    )
    
    # 保存结果
    cv2.imwrite("demo_mask_enhanced.jpg", enhanced_image)
    cv2.imwrite("demo_mask.jpg", mask)
    
    print(f"✓ 掩码文件: demo_mask.jpg")
    print(f"✓ 掩码增强图: demo_mask_enhanced.jpg")


def demo_api_usage():
    """演示 API 的基本使用方法"""
    print("\n=== API 使用方法演示 ===")
    
    # 1. 创建调整器实例
    adjuster = ImageSaturationAdjuster()
    
    # 2. 基本使用 - 直接处理图片数组
    demo_image_path = "demo.jpg"
    if os.path.exists(demo_image_path):
        image = cv2.imread(demo_image_path)
        
        # 增加饱和度
        enhanced = adjuster.adjust_saturation(image, saturation_factor=1.5)
        cv2.imwrite("demo_api_enhanced.jpg", enhanced)
        print("✓ API 基本使用示例完成")
    
    # 3. 命令行用法示例
    print("\n命令行使用示例:")
    print("# 处理单张图片")
    print("python image_saturation.py demo.jpg")
    print()
    print("# 指定输出路径和饱和度")
    print("python image_saturation.py demo.jpg -o output.jpg -s 2.0")
    print()
    print("# 批量处理目录")
    print("python image_saturation.py /path/to/images --batch")
    print()
    print("# 创建对比图")
    print("python image_saturation.py demo.jpg --comparison")


def main():
    """主演示函数"""
    print("图片饱和度增强工具演示")
    print("=" * 50)
    
    # 检查 OpenCV 是否正确安装
    print(f"OpenCV 版本: {cv2.__version__}")
    
    # 运行各种演示
    demo_single_image()
    demo_different_saturation_levels()
    demo_with_mask()
    demo_api_usage()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("生成的文件:")
    files = [
        "demo_enhanced.jpg",
        "demo_comparison.jpg", 
        "demo_saturation_0.5.jpg",
        "demo_saturation_1.0.jpg",
        "demo_saturation_1.5.jpg", 
        "demo_saturation_2.0.jpg",
        "demo_saturation_2.5.jpg",
        "demo_mask_enhanced.jpg",
        "demo_mask.jpg",
        "demo_api_enhanced.jpg"
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")


if __name__ == "__main__":
    main() 