import os
import cv2
import argparse
from tqdm import tqdm
import glob
import numpy as np
import json


def get_image_paths(directory):
    """Get all image paths in a directory, sorted by filename."""
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
        image_paths.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
        image_paths.extend(glob.glob(os.path.join(directory, f'*.{ext.upper()}')))
    
    # Sort images by filename (assuming numeric filenames)
    image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    return image_paths


def create_video_from_images(image_dir, output_path, fps=10, show_progress=True, bg_color=(0, 0, 0)):
    """
    Create a video from a sequence of images.
    If images have 4 channels, use the 4th channel as a mask to filter out the background.
    
    Args:
        image_dir (str): Directory containing images
        output_path (str): Path where the video will be saved
        fps (int): Frames per second for the output video
        show_progress (bool): Whether to show progress bar
        bg_color (tuple): Background color (B, G, R) to replace masked areas
    
    Returns:
        bool: True if successful, False otherwise
    """
    image_paths = get_image_paths(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir}")
        return False
    
    # Read the first image to determine dimensions and handling approach
    # Use IMREAD_UNCHANGED to preserve alpha channel if present
    first_img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    if first_img is None:
        print(f"Cannot read first image: {image_paths[0]}")
        return False
    
    # Determine image dimensions and setup
    if len(first_img.shape) == 3 and first_img.shape[2] == 4:
        if show_progress:
            print("Detected 4-channel images, will use 4th channel as mask")
        has_alpha = True
        h, w, _ = first_img.shape
    else:
        has_alpha = False
        # Reread as regular RGB if necessary
        if len(first_img.shape) != 3 or first_img.shape[2] != 3:
            first_img = cv2.imread(image_paths[0])
            if first_img is None:
                print(f"Cannot read first image: {image_paths[0]}")
                return False
        h, w = first_img.shape[:2]
    
    # Create video writer - always uses 3-channel output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print(f"Failed to open video writer for {output_path}")
        return False
    
    # Add each image to the video
    iterator = tqdm(image_paths, desc=f"Creating {os.path.basename(output_path)}", leave=False) if show_progress else image_paths
    for img_path in iterator:
        # Read image with alpha channel if present
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Cannot read image {img_path}, skipping")
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
            out.write(result)
        else:
            # For standard 3-channel images, ensure we have 3 channels
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] > 3:
                img = img[:, :, 0:3]
            
            out.write(img)
    
    out.release()
    return True


def count_total_instances(input_root, species_filter=None):
    """Count the total number of instances to process."""
    total_count = 0
    
    for species_dir in os.listdir(input_root):
        if species_filter and species_dir != species_filter:
            continue
            
        species_path = os.path.join(input_root, species_dir)
        if not os.path.isdir(species_path):
            continue
        
        for video_id in os.listdir(species_path):
            video_path = os.path.join(species_path, video_id)
            if not os.path.isdir(video_path):
                continue
            
            for instance_id in os.listdir(video_path):
                instance_path = os.path.join(video_path, instance_id)
                
                # Skip non-directory files (like frame_counters.json)
                if not os.path.isdir(instance_path):
                    continue
                
                # Check if this directory contains images
                image_paths = get_image_paths(instance_path)
                if image_paths:
                    total_count += 1
    
    return total_count


def process_dataset(input_root, output_root, fps=10, bg_color=(0, 0, 0)):
    """
    Process the entire dataset structure.
    
    Input structure:
    fish-instance-dataset/video/
    ├── species_name/
    │   ├── video_id/
    │   │   ├── instance_id/
    │   │   │   ├── 000000.webp
    │   │   │   ├── 000001.webp
    │   │   │   └── ...
    │   │   └── frame_counters.json
    
    Output structure:
    fish-mllms-dataset/video/
    ├── species_name/
    │   ├── video_id_instance_id.mp4
    │   └── ...
    """
    # Ensure output directory exists
    os.makedirs(output_root, exist_ok=True)
    
    # Count total instances for progress bar
    print("Counting total instances to process...")
    total_instances = count_total_instances(input_root)
    print(f"Found {total_instances} instances to process")
    
    # Create overall progress bar
    overall_pbar = tqdm(total=total_instances, desc="Overall Progress", unit="video")
    
    # Iterate through species directories
    for species_dir in os.listdir(input_root):
        species_path = os.path.join(input_root, species_dir)
        if not os.path.isdir(species_path):
            continue
        
        print(f"\nProcessing species: {species_dir}")
        
        # Create output species directory
        output_species_dir = os.path.join(output_root, species_dir)
        os.makedirs(output_species_dir, exist_ok=True)
        
        # Iterate through video IDs
        for video_id in os.listdir(species_path):
            video_path = os.path.join(species_path, video_id)
            if not os.path.isdir(video_path):
                continue
            
            # Iterate through instance IDs
            for instance_id in os.listdir(video_path):
                instance_path = os.path.join(video_path, instance_id)
                
                # Skip non-directory files (like frame_counters.json)
                if not os.path.isdir(instance_path):
                    continue
                
                # Check if this directory contains images
                image_paths = get_image_paths(instance_path)
                if not image_paths:
                    continue
                
                # Create output video filename with combined video_id and instance_id
                output_video_name = f"{video_id}_{instance_id}.mp4"
                output_video_path = os.path.join(output_species_dir, output_video_name)
                
                # Update progress bar description
                overall_pbar.set_description(f"Processing {species_dir}/{video_id}_{instance_id}")
                
                # Skip if video already exists
                if os.path.exists(output_video_path):
                    overall_pbar.write(f"Video {output_video_path} already exists, skipping")
                    overall_pbar.update(1)
                    continue
                
                # Create video from images
                success = create_video_from_images(instance_path, output_video_path, fps, show_progress=False, bg_color=bg_color)
                if success:
                    overall_pbar.write(f"✓ Created: {species_dir}/{video_id}_{instance_id}.mp4")
                else:
                    overall_pbar.write(f"✗ Failed: {species_dir}/{video_id}_{instance_id}")
                
                # Update overall progress
                overall_pbar.update(1)
    
    overall_pbar.close()


def process_specific_species(input_root, output_root, species, video_id=None, fps=10, bg_color=(0, 0, 0)):
    """Process a specific species or species/video combination."""
    if video_id:
        # Process specific species and video
        input_path = os.path.join(input_root, species, video_id)
        output_path = os.path.join(output_root, species)
        os.makedirs(output_path, exist_ok=True)
        
        # Count instances for this specific video
        total_instances = len([d for d in os.listdir(input_path) 
                              if os.path.isdir(os.path.join(input_path, d)) and 
                              get_image_paths(os.path.join(input_path, d))])
        
        print(f"Processing {species}/{video_id} - {total_instances} instances")
        
        # Create progress bar
        pbar = tqdm(total=total_instances, desc=f"Processing {species}/{video_id}", unit="video")
        
        # Process instances in this specific video
        for instance_id in os.listdir(input_path):
            instance_path = os.path.join(input_path, instance_id)
            if not os.path.isdir(instance_path):
                continue
            
            image_paths = get_image_paths(instance_path)
            if not image_paths:
                continue
            
            output_video_name = f"{video_id}_{instance_id}.mp4"
            output_video_path = os.path.join(output_path, output_video_name)
            
            pbar.set_description(f"Processing {species}/{video_id}_{instance_id}")
            
            if os.path.exists(output_video_path):
                pbar.write(f"Video {output_video_path} already exists, skipping")
                pbar.update(1)
                continue
            
            success = create_video_from_images(instance_path, output_video_path, fps, show_progress=False, bg_color=bg_color)
            if success:
                pbar.write(f"✓ Created: {species}/{video_id}_{instance_id}.mp4")
            else:
                pbar.write(f"✗ Failed: {species}/{video_id}_{instance_id}")
            
            pbar.update(1)
        
        pbar.close()
    else:
        # Process specific species only
        input_path = os.path.join(input_root, species)
        output_path = os.path.join(output_root, species)
        os.makedirs(output_path, exist_ok=True)
        
        # Count total instances for this species
        total_instances = count_total_instances(input_root, species_filter=species)
        print(f"Processing {species} - {total_instances} instances")
        
        # Create progress bar
        pbar = tqdm(total=total_instances, desc=f"Processing {species}", unit="video")
        
        for video_id in os.listdir(input_path):
            video_path = os.path.join(input_path, video_id)
            if not os.path.isdir(video_path):
                continue
            
            for instance_id in os.listdir(video_path):
                instance_path = os.path.join(video_path, instance_id)
                if not os.path.isdir(instance_path):
                    continue
                
                image_paths = get_image_paths(instance_path)
                if not image_paths:
                    continue
                
                output_video_name = f"{video_id}_{instance_id}.mp4"
                output_video_path = os.path.join(output_path, output_video_name)
                
                pbar.set_description(f"Processing {species}/{video_id}_{instance_id}")
                
                if os.path.exists(output_video_path):
                    pbar.write(f"Video {output_video_path} already exists, skipping")
                    pbar.update(1)
                    continue
                
                success = create_video_from_images(instance_path, output_video_path, fps, show_progress=False, bg_color=bg_color)
                if success:
                    pbar.write(f"✓ Created: {species}/{video_id}_{instance_id}.mp4")
                else:
                    pbar.write(f"✗ Failed: {species}/{video_id}_{instance_id}")
                
                pbar.update(1)
        
        pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Convert fish instance dataset images to MLLMs dataset videos.")
    parser.add_argument("--input", type=str, default="../data/fish-instance-dataset/video",
                        help="Input directory containing image sequences")
    parser.add_argument("--output", type=str, default="../data/fish-mllms-dataset/video",
                        help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for output videos")
    parser.add_argument("--species", type=str, default=None,
                        help="Process only specific species (optional)")
    parser.add_argument("--video_id", type=str, default=None,
                        help="Process only specific video ID (optional)")
    parser.add_argument("--bg_color", type=str, default="0,0,0", 
                        help="Background color in B,G,R format (default: 0,0,0)")
    
    args = parser.parse_args()
    
    # Parse background color
    bg_color = tuple(map(int, args.bg_color.split(',')))
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        return
    
    # If specific species is specified, use specialized function
    if args.species:
        process_specific_species(args.input, args.output, args.species, args.video_id, args.fps, bg_color)
    else:
        # Process entire dataset
        process_dataset(args.input, args.output, args.fps, bg_color)
    
    print("\n🎉 Conversion completed!")


if __name__ == "__main__":
    main()
