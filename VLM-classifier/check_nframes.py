#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查视频文件是否损坏或帧数不足的脚本
"""

import os
import cv2
from pathlib import Path


def check_video_file(video_path):
    """
    检查单个视频文件
    
    Args:
        video_path (str): 视频文件路径
        
    Returns:
        tuple: (is_corrupted, frame_count, error_message)
    """
    try:
        # 尝试打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 检查是否成功打开
        if not cap.isOpened():
            return True, 0, "无法打开视频文件"
        
        # 获取帧数
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果无法通过属性获取帧数，尝试手动计数
        if frame_count <= 0:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
        
        cap.release()
        
        # 检查是否损坏（帧数为0通常表示损坏）
        is_corrupted = frame_count == 0
        
        return is_corrupted, frame_count, None
        
    except Exception as e:
        return True, 0, f"读取视频时出错: {str(e)}"


def find_problematic_videos(root_dir, min_frames=5):
    """
    查找有问题的视频文件
    
    Args:
        root_dir (str): 根目录路径
        min_frames (int): 最小帧数阈值
        
    Returns:
        tuple: (problematic_videos, statistics)
            - problematic_videos: 有问题的视频文件列表
            - statistics: 统计信息字典 {'total_videos', 'total_frames', 'valid_videos', 'corrupted_videos', 'low_frame_videos'}
    """
    problematic_videos = []
    statistics = {
        'total_videos': 0,
        'total_frames': 0,
        'valid_videos': 0,
        'corrupted_videos': 0,
        'low_frame_videos': 0,
        'processed_videos': 0
    }
    
    # 检查目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        return problematic_videos, statistics
    
    print(f"开始检查目录: {root_dir}")
    print("="*60)
    
    # 遍历所有mp4文件
    mp4_files = list(Path(root_dir).rglob("*.mp4"))
    total_files = len(mp4_files)
    statistics['total_videos'] = total_files
    
    if total_files == 0:
        print("未找到任何mp4文件")
        return problematic_videos, statistics
    
    print(f"找到 {total_files} 个mp4文件，开始检查...")
    print("-" * 60)
    
    for i, video_file in enumerate(mp4_files, 1):
        video_path = str(video_file)
        
        is_corrupted, frame_count, error_msg = check_video_file(video_path)
        
        # 更新统计信息
        statistics['processed_videos'] += 1
        if not is_corrupted and frame_count > 0:
            statistics['total_frames'] += frame_count
        
        # 判断是否有问题
        has_problem = False
        problem_description = []
        
        if is_corrupted:
            has_problem = True
            statistics['corrupted_videos'] += 1
            if error_msg:
                problem_description.append(f"损坏 ({error_msg})")
            else:
                problem_description.append("损坏")
        
        if frame_count < min_frames and frame_count > 0:
            has_problem = True
            statistics['low_frame_videos'] += 1
            problem_description.append(f"帧数不足 ({frame_count}帧)")
        
        if has_problem:
            problem_info = {
                'path': video_path,
                'frame_count': frame_count,
                'problems': problem_description,
                'is_corrupted': is_corrupted
            }
            problematic_videos.append(problem_info)
            print(f"  ❌ 问题: {', '.join(problem_description)}")
        else:
            statistics['valid_videos'] += 1
    
    return problematic_videos, statistics


def delete_problematic_videos(problematic_videos):
    """
    删除有问题的视频文件
    
    Args:
        problematic_videos (list): 有问题的视频文件列表
        
    Returns:
        tuple: (deleted_count, failed_count, failed_files)
    """
    if not problematic_videos:
        print("没有需要删除的文件")
        return 0, 0, []
    
    deleted_count = 0
    failed_count = 0
    failed_files = []
    
    print(f"\n开始删除 {len(problematic_videos)} 个有问题的视频文件...")
    print("-" * 60)
    
    for i, video_info in enumerate(problematic_videos, 1):
        video_path = video_info['path']
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                deleted_count += 1
                print(f"[{i}/{len(problematic_videos)}] ✅ 已删除: {video_path}")
            else:
                print(f"[{i}/{len(problematic_videos)}] ⚠️  文件不存在: {video_path}")
        except Exception as e:
            failed_count += 1
            failed_files.append((video_path, str(e)))
            print(f"[{i}/{len(problematic_videos)}] ❌ 删除失败: {video_path} ({e})")
    
    return deleted_count, failed_count, failed_files


def get_user_confirmation(problematic_videos):
    """
    获取用户确认是否删除文件
    
    Args:
        problematic_videos (list): 有问题的视频文件列表
        
    Returns:
        bool: 用户是否确认删除
    """
    if not problematic_videos:
        return False
    
    print(f"\n⚠️  即将删除以下 {len(problematic_videos)} 个有问题的视频文件:")
    print("=" * 60)
    
    for i, video_info in enumerate(problematic_videos, 1):
        print(f"{i:2d}. {video_info['path']}")
        print(f"    问题: {', '.join(video_info['problems'])}")
        if not video_info['is_corrupted']:
            print(f"    帧数: {video_info['frame_count']}")
        print()
    
    print("=" * 60)
    while True:
        response = input("确认删除这些文件吗？(y/yes 确认, n/no 取消): ").lower().strip()
        if response in ['y', 'yes', '是', '确认']:
            return True
        elif response in ['n', 'no', '否', '取消']:
            return False
        else:
            print("请输入 y/yes 或 n/no")


def main():
    """主函数"""
    # 目标目录
    target_dir = "src/r1-v/fish-mllms-dataset/video"
    min_frames = 5
    
    print("视频文件检查工具")
    print("=" * 60)
    
    # 查找有问题的视频
    problematic_videos, statistics = find_problematic_videos(target_dir, min_frames)
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("视频统计信息:")
    print("=" * 60)
    print(f"📁 总视频数: {statistics['total_videos']}")
    print(f"🎬 总帧数: {statistics['total_frames']:,}")
    if statistics['total_videos'] > 0:
        avg_frames = statistics['total_frames'] / statistics['valid_videos'] if statistics['valid_videos'] > 0 else 0
        print(f"📊 平均帧数: {avg_frames:.2f}")
    print(f"✅ 有效视频数: {statistics['valid_videos']}")
    print(f"❌ 损坏视频数: {statistics['corrupted_videos']}")
    print(f"⚠️  帧数不足的视频数: {statistics['low_frame_videos']}")
    print(f"🔄 已处理的视频数: {statistics['processed_videos']}")
    
    # 输出问题视频详情
    print("\n" + "=" * 60)
    print("检查结果汇总:")
    print("=" * 60)
    
    if not problematic_videos:
        print("🎉 所有视频文件都正常！")
        return
    
    print(f"⚠️  发现 {len(problematic_videos)} 个有问题的视频文件:")
    print("-" * 60)
    
    for i, video_info in enumerate(problematic_videos, 1):
        print(f"{i:2d}. {video_info['path']}")
        print(f"    问题: {', '.join(video_info['problems'])}")
        if not video_info['is_corrupted']:
            print(f"    帧数: {video_info['frame_count']}")
        print()
    
    # 询问用户是否删除
    if get_user_confirmation(problematic_videos):
        deleted_count, failed_count, failed_files = delete_problematic_videos(problematic_videos)
        
        # 输出删除结果
        print("\n" + "=" * 60)
        print("删除结果汇总:")
        print("=" * 60)
        print(f"✅ 成功删除: {deleted_count} 个文件")
        
        if failed_count > 0:
            print(f"❌ 删除失败: {failed_count} 个文件")
            print("-" * 40)
            for file_path, error in failed_files:
                print(f"  {file_path}")
                print(f"  错误: {error}")
                print()
        
        print(f"📊 总计处理: {len(problematic_videos)} 个有问题的文件")
        
        # 更新后的统计信息
        if deleted_count > 0:
            print("\n" + "=" * 60)
            print("更新后的统计信息:")
            print("=" * 60)
            remaining_videos = statistics['total_videos'] - deleted_count
            remaining_problematic = len(problematic_videos) - deleted_count
            print(f"📁 剩余视频数: {remaining_videos}")
            print(f"✅ 有效视频数: {statistics['valid_videos']}")
            print(f"⚠️  剩余问题视频数: {remaining_problematic}")
    else:
        print("\n❌ 用户取消删除操作")


if __name__ == "__main__":
    main()
