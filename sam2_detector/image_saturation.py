import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Union, List, Tuple
import logging
import time
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageSaturationAdjuster:
    """图片饱和度调整器"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def adjust_saturation(self, image: np.ndarray, saturation_factor: float = 1.5) -> np.ndarray:
        """
        调整图片饱和度
        
        Args:
            image: 输入图片 (BGR格式)
            saturation_factor: 饱和度调整因子，>1增加饱和度，<1降低饱和度，1保持不变
            
        Returns:
            调整后的图片
        """
        if saturation_factor < 0:
            raise ValueError("饱和度因子必须为非负数")
        
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        
        # 调整饱和度通道(S通道)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
        
        # 限制数值范围在0-255之间
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # 转换回BGR色彩空间
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def rotate_right_90(self, image: np.ndarray) -> np.ndarray:
        """
        将图像向右旋转90度
        
        Args:
            image: 输入图片 (BGR格式)
            
        Returns:
            旋转后的图片
        """
        # 使用cv2.rotate函数向右旋转90度
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated
    
    def enhance_saturation_with_mask(self, image: np.ndarray, mask: np.ndarray = None, 
                                   saturation_factor: float = 1.5) -> np.ndarray:
        """
        使用掩码选择性地增强图片饱和度
        
        Args:
            image: 输入图片 (BGR格式)
            mask: 二值掩码，只对掩码区域进行饱和度调整
            saturation_factor: 饱和度调整因子
            
        Returns:
            调整后的图片
        """
        if mask is None:
            return self.adjust_saturation(image, saturation_factor)
        
        # 确保掩码是单通道
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # 创建掩码的逆掩码
        mask_inv = cv2.bitwise_not(mask)
        
        # 对整张图片调整饱和度
        enhanced = self.adjust_saturation(image, saturation_factor)
        
        # 使用掩码合并原图和增强图
        result = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        original_masked = cv2.bitwise_and(image, image, mask=mask_inv)
        
        # 将两部分合并
        final_result = cv2.add(result, original_masked)
        
        return final_result
    
    def process_image_file(self, input_path: str, output_path: str = None, 
                          saturation_factor: float = 1.5, quality: int = 95, rotate: bool = False) -> str:
        """
        处理单个图片文件
        
        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径，如果为None则自动生成
            saturation_factor: 饱和度调整因子
            quality: 保存质量 (1-100)
            rotate: 是否向右旋转90度
            
        Returns:
            输出文件路径
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 读取图片
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图片: {input_path}")
        
        # logger.info(f"处理图片: {input_path}, 原始尺寸: {image.shape}")
        
        # 调整饱和度
        enhanced_image = self.adjust_saturation(image, saturation_factor)
        
        # 如果需要，向右旋转90度
        if rotate:
            enhanced_image = self.rotate_right_90(enhanced_image)
        
        # 生成输出路径
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_enhanced{input_path_obj.suffix}")
        
        # 保存图片
        save_params = []
        if input_path.lower().endswith(('.jpg', '.jpeg')):
            save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif input_path.lower().endswith('.png'):
            save_params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]
        
        success = cv2.imwrite(output_path, enhanced_image, save_params)
        if not success:
            raise RuntimeError(f"保存图片失败: {output_path}")
        
        # logger.info(f"图片保存成功: {output_path}")
        return output_path
    
    def batch_process(self, input_dir: str, output_dir: str = None, 
                     saturation_factor: float = 1.5, quality: int = 95, rotate: bool = False) -> List[str]:
        """
        批量处理目录中的图片（包括子目录）
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录，如果为None则在输入目录创建enhanced子目录
            saturation_factor: 饱和度调整因子
            quality: 保存质量
            rotate: 是否向右旋转90度
            
        Returns:
            处理成功的文件路径列表
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(input_dir, "enhanced")
        
        # 递归查找所有支持的图片文件
        input_files = []
        input_path = Path(input_dir)
        
        for ext in self.supported_formats:
            # 递归搜索所有子目录
            input_files.extend(input_path.rglob(f"*{ext}"))
            input_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"在目录 {input_dir} 及其子目录中找到 {len(input_files)} 个图片文件")
        
        # 批量处理
        processed_files = []
        for input_file in tqdm(input_files, desc="Processing", unit="file"):
            try:
                # 计算相对路径，保持目录结构
                relative_path = input_file.relative_to(input_path)
                output_file_path = Path(output_dir) / relative_path
                
                # 创建输出目录（如果不存在）
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 处理图片
                self.process_image_file(str(input_file), str(output_file_path), saturation_factor, quality, rotate)
                processed_files.append(str(output_file_path))
                
            except Exception as e:
                logger.error(f"处理文件 {input_file} 失败: {e}")
        
        logger.info(f"批量处理完成，成功处理 {len(processed_files)} 个文件")
        return processed_files
    
    def create_comparison_image(self, original_path: str, enhanced_path: str, 
                              comparison_path: str = None) -> str:
        """
        创建对比图片（左边原图，右边增强图）
        
        Args:
            original_path: 原图路径
            enhanced_path: 增强图路径
            comparison_path: 对比图保存路径
            
        Returns:
            对比图路径
        """
        original = cv2.imread(original_path)
        enhanced = cv2.imread(enhanced_path)
        
        if original is None or enhanced is None:
            raise ValueError("无法读取图片文件")
        
        # 确保两张图片尺寸相同
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
        
        # 水平拼接
        comparison = np.hstack([original, enhanced])
        
        # 添加分隔线
        line_thickness = 3
        line_color = (255, 255, 255)  # 白色分隔线
        cv2.line(comparison, (original.shape[1], 0), 
                (original.shape[1], original.shape[0]), line_color, line_thickness)
        
        # 添加文字标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)  # 白色文字
        
        # 原图标签
        cv2.putText(comparison, "Original", (20, 40), font, font_scale, text_color, font_thickness)
        # 增强图标签
        cv2.putText(comparison, "Enhanced", (original.shape[1] + 20, 40), 
                   font, font_scale, text_color, font_thickness)
        
        # 生成对比图路径
        if comparison_path is None:
            original_path_obj = Path(original_path)
            comparison_path = str(f"{original_path_obj.stem}_comparison{original_path_obj.suffix}")
        
        # 保存对比图
        cv2.imwrite(comparison_path, comparison)
        logger.info(f"对比图保存成功: {comparison_path}")
        
        return comparison_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图片饱和度增强工具")
    parser.add_argument("--input", default="", help="输入图片文件或目录路径")
    parser.add_argument("--output", default="", help="输出路径")
    parser.add_argument("--saturation", type=float, default=2.5, 
                       help="饱和度调整因子 (默认: 3.0)")
    parser.add_argument("-q", "--quality", type=int, default=100, 
                       help="输出图片质量 1-100 (默认: 100)")
    parser.add_argument("--batch", action="store_true", 
                       help="批量处理模式（处理目录中所有图片）")
    parser.add_argument("--comparison", action="store_true", 
                       help="创建对比图")
    parser.add_argument("--rotate", action="store_true", 
                       help="向右旋转90度")
    
    args = parser.parse_args()
    
    # 创建饱和度调整器
    adjuster = ImageSaturationAdjuster()
    
    try:
        if args.batch or os.path.isdir(args.input):
            # 批量处理模式
            logger.info(f"批量处理模式，饱和度因子: {args.saturation}, 旋转: {args.rotate}")
            processed_files = adjuster.batch_process(
                args.input, args.output, args.saturation, args.quality, args.rotate
            )
            # print(f"批量处理完成，成功处理 {len(processed_files)} 个文件")
            
        else:
            # 单文件处理模式
            logger.info(f"处理单个文件，饱和度因子: {args.saturation}, 旋转: {args.rotate}")
            output_path = adjuster.process_image_file(
                args.input, args.output, args.saturation, args.quality, args.rotate
            )
            # print(f"图片处理完成: {output_path}")
            
            # 创建对比图
            if args.comparison:
                comparison_path = adjuster.create_comparison_image(args.input, output_path)
                # print(f"对比图创建完成: {comparison_path}")
                
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
