import os
import shutil
import argparse

def is_video_file(filename):
    video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm'}
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def move_videos_to_root(root_dir):
    # Get the absolute path of the root directory
    root_dir = os.path.abspath(root_dir)
    
    # Check if the directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist!")
        return
    
    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the root directory itself
        if dirpath == root_dir:
            continue
            
        # Process each file in the current directory
        for filename in filenames:
            if is_video_file(filename):
                source_path = os.path.join(dirpath, filename)
                destination_path = os.path.join(root_dir, filename)
                
                # Handle filename conflicts
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(destination_path):
                    new_filename = f"{base}_{counter}{ext}"
                    destination_path = os.path.join(root_dir, new_filename)
                    counter += 1
                
                # Move the file
                try:
                    shutil.move(source_path, destination_path)
                    print(f"Moved: {filename} -> {os.path.basename(destination_path)}")
                except Exception as e:
                    print(f"Error moving {filename}: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Move video files from subdirectories to the root directory.')
    parser.add_argument('input_path', help='The directory to process')
    args = parser.parse_args()
    
    print(f"Moving video files from subdirectories to: {args.input_path}")
    move_videos_to_root(args.input_path)
    print("Done!")