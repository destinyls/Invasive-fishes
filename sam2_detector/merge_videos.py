#!/usr/bin/env python3
"""
视频合并脚本 - 将输入目录下的所有视频文件合并为一个完整视频
支持多种视频格式，按文件名排序
"""

import os
import glob
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


def get_video_files(input_dir: str, extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v')) -> List[str]:
    """
    获取目录下所有视频文件，按文件名排序
    
    Args:
        input_dir: 输入目录路径
        extensions: 支持的视频文件扩展名
    
    Returns:
        排序后的视频文件路径列表
    """
    video_files = []
    
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        video_files.extend(glob.glob(pattern))
        # 同时支持大写扩展名
        pattern = os.path.join(input_dir, f"*{ext.upper()}")
        video_files.extend(glob.glob(pattern))
    
    # 按文件名排序
    video_files.sort(key=lambda x: os.path.basename(x))
    
    return video_files


def get_video_info(video_path: str) -> Tuple[int, int, float, int]:
    """
    获取视频信息
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        (width, height, fps, frame_count)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return width, height, fps, frame_count


def find_common_resolution(video_files: List[str]) -> Tuple[int, int, float]:
    """
    找到所有视频的公共分辨率和帧率
    
    Args:
        video_files: 视频文件路径列表
    
    Returns:
        (width, height, fps) - 使用第一个视频的参数作为基准
    """
    if not video_files:
        raise ValueError("视频文件列表为空")
    
    # 获取第一个视频的信息作为基准
    width, height, fps, _ = get_video_info(video_files[0])
    
    print(f"使用基准分辨率: {width}x{height}, 帧率: {fps:.2f}fps")
    
    # 检查其他视频的参数
    for video_file in video_files[1:]:
        try:
            w, h, f, _ = get_video_info(video_file)
            if w != width or h != height:
                print(f"警告: {os.path.basename(video_file)} 分辨率 {w}x{h} 与基准不同，将被调整")
            if abs(f - fps) > 0.1:
                print(f"警告: {os.path.basename(video_file)} 帧率 {f:.2f}fps 与基准不同")
        except Exception as e:
            print(f"警告: 无法读取 {os.path.basename(video_file)} 的信息: {e}")
    
    return width, height, fps


def merge_videos(input_dir: str, output_path: str, quality: str = 'high') -> None:
    """
    合并视频文件
    
    Args:
        input_dir: 输入目录路径
        output_path: 输出视频文件路径
        quality: 输出质量 ('high', 'medium', 'low')
    """
    # 获取所有视频文件
    video_files = get_video_files(input_dir)
    
    if not video_files:
        print(f"在目录 {input_dir} 中未找到任何视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(video_file)}")
    
    # 确定输出参数
    width, height, fps = find_common_resolution(video_files)
    
    # 设置编码器和质量参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 根据质量设置调整参数
    quality_settings = {
        'high': {'bitrate': -1},  # 使用默认高质量
        'medium': {'bitrate': -1},
        'low': {'bitrate': -1}
    }
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建视频写入器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"无法创建输出视频文件: {output_path}")
    
    try:
        total_frames = 0
        processed_frames = 0
        
        # 计算总帧数
        print("计算总帧数...")
        for video_file in video_files:
            try:
                _, _, _, frame_count = get_video_info(video_file)
                total_frames += frame_count
            except Exception as e:
                print(f"警告: 无法获取 {os.path.basename(video_file)} 的帧数: {e}")
        
        print(f"总共需要处理 {total_frames} 帧")
        
        # 逐个处理视频文件
        for video_idx, video_file in enumerate(video_files, 1):
            print(f"\n处理视频 {video_idx}/{len(video_files)}: {os.path.basename(video_file)}")
            
            cap = cv2.VideoCapture(video_file)
            
            if not cap.isOpened():
                print(f"跳过无法打开的文件: {video_file}")
                continue
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调整帧尺寸（如果需要）
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                # 写入帧
                out.write(frame)
                
                frame_idx += 1
                processed_frames += 1
                
                # 显示进度
                if frame_idx % 30 == 0 or processed_frames % 100 == 0:
                    progress = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
                    print(f"  进度: {progress:.1f}% ({processed_frames}/{total_frames} 帧)")
            
            cap.release()
            print(f"  完成 {os.path.basename(video_file)}: {frame_idx} 帧")
        
        print(f"\n视频合并完成!")
        print(f"输出文件: {output_path}")
        print(f"总帧数: {processed_frames}")
        print(f"分辨率: {width}x{height}")
        print(f"帧率: {fps:.2f}fps")
        
        # 计算视频时长
        duration = processed_frames / fps if fps > 0 else 0
        print(f"视频时长: {duration:.2f}秒 ({duration/60:.2f}分钟)")
        
    finally:
        out.release()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将输入目录下的所有视频文件合并为一个完整视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python merge_videos.py /path/to/input/dir
  python merge_videos.py /path/to/input/dir -o merged_video.mp4
  python merge_videos.py /path/to/input/dir -o output.mp4 --quality medium
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='包含视频文件的输入目录'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='merged_video.mp4',
        help='输出视频文件路径 (默认: merged_video.mp4)'
    )
    
    parser.add_argument(
        '--quality',
        choices=['high', 'medium', 'low'],
        default='high',
        help='输出视频质量 (默认: high)'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'],
        help='支持的视频文件扩展名 (默认: .mp4 .avi .mov .mkv .flv .wmv .m4v)'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return 1
    
    # 确保输出文件有正确的扩展名
    if not args.output.lower().endswith('.mp4'):
        args.output += '.mp4'
    
    try:
        merge_videos(args.input_dir, args.output, args.quality)
        return 0
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
