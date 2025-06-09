import os
import cv2
import numpy as np
from glob import glob


def images_to_video(image_folder, output_path, fps=30, sort_by_name=True, bg_color=(0, 0, 0)):
    """
    Convert a folder of images to an MP4 video file.
    If images have 4 channels, use the 4th channel as a mask to filter out the background.
    
    Args:
        image_folder (str): Path to the folder containing images
        output_path (str): Path where the video will be saved
        fps (int): Frames per second for the output video
        sort_by_name (bool): Whether to sort images by name
        bg_color (tuple): Background color (B, G, R) to replace masked areas
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if the image folder exists
    if not os.path.exists(image_folder):
        print(f"Error: Image folder {image_folder} does not exist")
        return False
    
    # Get all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']:
        image_files.extend(glob(os.path.join(image_folder, f"*.{ext}")))
        image_files.extend(glob(os.path.join(image_folder, f"*.{ext.upper()}")))
    
    if not image_files:
        print(f"Error: No image files found in {image_folder}")
        return False
    
    # Sort files if needed
    if sort_by_name:
        image_files.sort()
    
    # Read the first image to determine dimensions and handling approach
    # Use IMREAD_UNCHANGED to preserve alpha channel if present
    first_image = cv2.imread(image_files[0], cv2.IMREAD_UNCHANGED)
    if first_image is None:
        print(f"Error: Cannot read image {image_files[0]}")
        return False
    
    # Determine image dimensions and setup
    if len(first_image.shape) == 3 and first_image.shape[2] == 4:
        print("Detected 4-channel images, will use 4th channel as mask")
        has_alpha = True
        height, width, _ = first_image.shape
    else:
        print("Processing regular 3-channel images")
        has_alpha = False
        # Reread as regular RGB if necessary
        if len(first_image.shape) != 3 or first_image.shape[2] != 3:
            first_image = cv2.imread(image_files[0])
            if first_image is None:
                print(f"Error: Cannot read image {image_files[0]}")
                return False
        height, width, _ = first_image.shape
    
    # Create video writer - always uses 3-channel output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each image to the video
    for image_file in image_files:
        # Read image with alpha channel if present
        img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not read {image_file}, skipping")
            continue
        
        # Process 4-channel images
        if has_alpha and len(img.shape) == 3 and img.shape[2] == 4:
            # Extract RGB and alpha channels
            rgb = img[:, :, 0:3]
            mask = img[:, :, 3]
            
            # Create 3-channel background using bg_color
            bg = np.ones_like(rgb) * np.array(bg_color)
            
            # Normalize mask to range 0-1 if needed
            if mask.max() > 1:
                mask = mask / 255.0
            
            # Apply the mask to blend foreground and background
            # Expand mask dimensions for broadcasting
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            
            # Final result: foreground * mask + background * (1-mask)
            result = (rgb * mask_3d + bg * (1 - mask_3d)).astype(np.uint8)
            video_writer.write(result)
        else:
            # For standard 3-channel images, ensure we have 3 channels
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] > 3:
                img = img[:, :, 0:3]
                
            video_writer.write(img)
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_path}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert a folder of images to a video")
    parser.add_argument("--image_folder", type=str, default="../data/fish-dataset/video/oreochromis_mossambicus/P241211_174400_175202_002", help="Path to folder containing images")
    parser.add_argument("--output_path", type=str, default="raw_oreochromis_mossambicus_P241211_174400_175202_002.mp4", help="Path to save the output video")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 30)")
    parser.add_argument("--no_sort", action="store_false", dest="sort_by_name", 
                        help="Do not sort images by filename")
    parser.add_argument("--bg_color", type=str, default="0,0,0", 
                        help="Background color in B,G,R format (default: 0,0,0)")
    
    args = parser.parse_args()
    
    # Parse background color
    bg_color = tuple(map(int, args.bg_color.split(',')))
    
    success = images_to_video(
        args.image_folder, 
        args.output_path, 
        fps=args.fps, 
        sort_by_name=args.sort_by_name,
        bg_color=bg_color
    )
    
    if not success:
        exit(1)
