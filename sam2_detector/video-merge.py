#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path

def find_image_files(directory):
    """Find all image files in the given directory and its subdirectories."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip the base directory itself
        if root == directory:
            continue
            
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def move_images_to_parent(directory):
    """Move all image files from subdirectories to the parent directory and delete subdirectories."""
    directory = os.path.abspath(directory)
    image_files = find_image_files(directory)
    
    moved_count = 0
    skipped_count = 0
    processed_dirs = set()
    
    for image_path in image_files:
        # Get the subdirectory this image is in
        subdir = os.path.dirname(image_path)
        processed_dirs.add(subdir)
        
        # Get the parent directory of the current directory
        target_dir = directory
        
        # Create the target file path
        file_name = os.path.basename(image_path)
        target_path = os.path.join(target_dir, file_name)
        
        # Handle filename conflicts
        if os.path.exists(target_path):
            base, ext = os.path.splitext(file_name)
            subdir_name = os.path.basename(os.path.dirname(image_path))
            new_name = f"{base}_{subdir_name}{ext}"
            target_path = os.path.join(target_dir, new_name)
        
        try:
            print(f"Moving: {image_path} -> {target_path}")
            shutil.move(image_path, target_path)
            moved_count += 1
        except Exception as e:
            print(f"Error moving {image_path}: {str(e)}")
            skipped_count += 1
    
    # Delete subdirectories
    deleted_dirs = 0
    for subdir in sorted(processed_dirs, key=len, reverse=True):  # Sort by length to delete deeper dirs first
        try:
            # Check if directory is empty or only contains directories (which may be empty)
            should_delete = True
            for _, _, files in os.walk(subdir):
                if files:  # If there are any files left, don't delete
                    should_delete = False
                    break
            
            if should_delete:
                print(f"Deleting directory: {subdir}")
                shutil.rmtree(subdir)
                deleted_dirs += 1
            else:
                print(f"Not deleting directory as it still contains files: {subdir}")
        except Exception as e:
            print(f"Error deleting directory {subdir}: {str(e)}")
    
    print(f"Done! Moved {moved_count} images, skipped {skipped_count}, deleted {deleted_dirs} directories.")
    return moved_count, skipped_count, deleted_dirs

def main():
    parser = argparse.ArgumentParser(description='Move all image files from subdirectories to the parent directory and delete subdirectories.')
    parser.add_argument('directory', nargs='?', default='.', 
                        help='The directory to process (default: current directory)')
    args = parser.parse_args()
    
    move_images_to_parent(args.directory)

if __name__ == '__main__':
    main()
