import os
import shutil
import argparse
from pathlib import Path
import re

def natural_sort_key(s):
    """
    Sort strings with embedded numbers in natural order.
    For example: ['img1.jpg', 'img2.jpg', 'img10.jpg'] will be sorted as
    ['img1.jpg', 'img2.jpg', 'img10.jpg'] instead of ['img1.jpg', 'img10.jpg', 'img2.jpg']
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def split_image_sequence(source_dir, output_base_dir, max_frames=500, copy_mode=False):
    """
    Split an image sequence into multiple directories with at most max_frames per directory.
    
    Args:
        source_dir (str): Path to the source directory containing the image sequence
        output_base_dir (str): Path to the base output directory where subdirectories will be created
        max_frames (int): Maximum number of frames per subdirectory
        copy_mode (bool): If True, copy files instead of moving them
    """
    # Get all image files from the source directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(source_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"No image files found in {source_dir}")
        return
    
    # Sort image files in natural order
    image_files.sort(key=natural_sort_key)
    
    # Create the base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Split images into subdirectories
    total_images = len(image_files)
    num_subdirs = (total_images + max_frames - 1) // max_frames  # Ceiling division
    
    print(f"Found {total_images} images, will create {num_subdirs} subdirectories")
    
    # Get source directory basename for naming subdirectories
    source_basename = os.path.basename(os.path.normpath(source_dir))
    
    for subdir_idx in range(num_subdirs):
        # Create subdirectory with naming pattern: source_dir_001, source_dir_002, etc.
        subdir_name = f"{source_basename}_{subdir_idx + 1:03d}"
        subdir_path = os.path.join(output_base_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        
        # Calculate range of images for this subdirectory
        start_idx = subdir_idx * max_frames
        end_idx = min(start_idx + max_frames, total_images)
        
        print(f"Processing subdirectory {subdir_name}: frames {start_idx + 1}-{end_idx}")
        
        # Copy or move images to the subdirectory
        for img_idx in range(start_idx, end_idx):
            src_path = os.path.join(source_dir, image_files[img_idx])
            dst_path = os.path.join(subdir_path, image_files[img_idx])
            
            if copy_mode:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

def process_video_directory(input_dir, output_dir, max_frames=500, copy_mode=False):
    """
    Process all video directories within the input directory
    
    Args:
        input_dir (str): Root input directory containing video subdirectories
        output_dir (str): Root output directory
        max_frames (int): Maximum frames per subdirectory
        copy_mode (bool): If True, copy files instead of moving them
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each video directory
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        
        # Skip if not a directory
        if not os.path.isdir(item_path):
            continue
            
        # Check if directory contains image files
        has_images = any(
            f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) 
            for f in os.listdir(item_path)
        )
        
        if has_images:
            # This is a video directory with image frames, process it
            print(f"Processing video directory: {item}")
            split_image_sequence(item_path, output_dir, max_frames, copy_mode)
        else:
            # This might be a fish type directory, check its subdirectories
            fish_type_dir = item_path
            fish_type_output_dir = os.path.join(output_dir, item)
            os.makedirs(fish_type_output_dir, exist_ok=True)
            
            for video_dir in os.listdir(fish_type_dir):
                video_path = os.path.join(fish_type_dir, video_dir)
                
                # Skip if not a directory
                if not os.path.isdir(video_path):
                    continue
                    
                # Check if directory contains image files
                has_images = any(
                    f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) 
                    for f in os.listdir(video_path)
                )
                
                if has_images:
                    print(f"Processing video directory: {os.path.join(item, video_dir)}")
                    split_image_sequence(video_path, fish_type_output_dir, max_frames, copy_mode)

def main():
    parser = argparse.ArgumentParser(description='Split image sequences into subdirectories')
    parser.add_argument('--source', type=str, default="../data/fish-dataset-yanghu", help='Source directory containing image sequence or root directory with multiple video directories')
    parser.add_argument('--output', type=str, default="../data/fish-dataset-yanghu", help='Base output directory (default: source_dir + "_split")')
    parser.add_argument('--max-frames', type=int, default=600, help='Maximum frames per subdirectory (default: 1000)')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of moving them')
    parser.add_argument('--recursive', action='store_false', help='Process all video directories recursively')
    
    args = parser.parse_args()
    
    source_dir = args.source
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    output_dir = args.output
    if not output_dir:
        # If source is like /path/to/data/fish-dataset-yanghu/video
        # Then output should be /path/to/data/fish-dataset-yanghu-split/video
        source_parts = os.path.normpath(source_dir).split(os.sep)
        if len(source_parts) >= 2:
            # Replace 'fish-dataset-yanghu' with 'fish-dataset-yanghu-split'
            dataset_part = source_parts[-2]
            if 'dataset' in dataset_part and not dataset_part.endswith('-split'):
                source_parts[-2] = f"{dataset_part}-split"
                # Reconstruct the path
                output_dir = os.sep.join(source_parts)
            else:
                output_dir = f"{source_dir}_split"
        else:
            output_dir = f"{source_dir}_split"
    
    if args.recursive:
        process_video_directory(source_dir, output_dir, args.max_frames, args.copy)
    else:
        split_image_sequence(source_dir, output_dir, args.max_frames, args.copy)
    
    print(f"Image sequence splitting completed. Output in: {output_dir}")

if __name__ == "__main__":
    main()
