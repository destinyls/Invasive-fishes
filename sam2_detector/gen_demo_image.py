#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import json
from pathlib import Path
import glob

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_frame_number(filename):
    """Extract frame number from filename"""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    # Find the number in the filename
    # Assuming format like "00001.jpg" or "frame_00001.jpg" or similar
    digits = ''.join([c for c in name if c.isdigit()])
    if digits:
        return int(digits)
    return 0  # Default if no number found

def find_matching_instances(global_frame_path, instance_dir, max_instances=5):
    """
    Find instance images that match the global frame
    
    Args:
        global_frame_path: Path to the global frame image
        instance_dir: Directory containing instance crops
        max_instances: Maximum number of instances to show
        
    Returns:
        List of matching instance image paths
    """
    global_frame_name = os.path.basename(global_frame_path)
    base_name = os.path.splitext(global_frame_name)[0]
    
    # Get all instance directories
    instance_subdirs = [d for d in os.listdir(instance_dir) 
                       if os.path.isdir(os.path.join(instance_dir, d))]
    
    matching_instances = []
    
    # Look for matching instances across all instance directories
    for subdir in instance_subdirs:
        instance_path = os.path.join(instance_dir, subdir)
        
        # Look for exact matches only - files with the same base name as the global frame
        # Try both .jpg and .png extensions
        match_jpg = os.path.join(instance_path, f"{base_name}_nobg.jpg")
        match_png = os.path.join(instance_path, f"{base_name}_nobg.png")
        if os.path.exists(match_jpg):
            matching_instances.append(match_jpg)
        elif os.path.exists(match_png):
            matching_instances.append(match_png)
    # Limit the number of instances
    return matching_instances[:max_instances]

def create_demo_image(global_frame_path, instance_paths, output_path, 
                     canvas_height=1080, instance_grid=(3, 3), 
                     bg_color=(240, 240, 240), 
                     add_labels=True):
    """
    Create a demo image with global view on left and instance views on right
    
    Args:
        global_frame_path: Path to the global frame image
        instance_paths: List of paths to instance images
        output_path: Where to save the result
        canvas_height: Height of the output canvas
        instance_grid: Tuple of (rows, cols) for instance grid
        bg_color: Background color for the instance panel (light gray default)
        add_labels: Whether to add instance ID labels
    """
    # Read the global frame
    global_frame = cv2.imread(global_frame_path)
    if global_frame is None:
        print(f"Error: Cannot read global frame {global_frame_path}")
        return
    
    # Calculate scaling for global frame
    global_h, global_w = global_frame.shape[:2]
    global_scale = canvas_height / global_h
    global_w_scaled = int(global_w * global_scale)
    global_h_scaled = canvas_height
    
    # Resize global frame
    global_frame_resized = cv2.resize(global_frame, (global_w_scaled, global_h_scaled))
    
    # Calculate instance panel size
    instance_panel_w = global_w_scaled  # Same width as global frame
    instance_panel_h = global_h_scaled
    
    # Create instance panel with light gray background
    instance_panel = np.ones((instance_panel_h, instance_panel_w, 3), dtype=np.uint8)
    instance_panel[:] = bg_color
    
    # Calculate grid cell size
    grid_rows, grid_cols = instance_grid
    cell_w = instance_panel_w // grid_cols
    cell_h = instance_panel_h // grid_rows
    cell_padding = 10  # Padding inside each cell
    
    # First, determine a consistent height for all instances
    instance_height = int(cell_h * 0.85)  # Increased from 0.8 to 0.85
    
    # Reduce top margin
    top_margin = int(cell_h * 0.01)  # Small top margin (5% of cell height)
    
    # Load and place instance images
    for i, instance_path in enumerate(instance_paths):
        if i >= grid_rows * grid_cols:
            break  # Don't exceed grid size
            
        # Get the grid position
        row = i // grid_cols
        col = i % grid_cols
        
        # Calculate cell position
        cell_x = col * cell_w
        cell_y = row * cell_h
        
        # Read instance image with alpha channel
        instance_img = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
        if instance_img is None:
            continue
            
        # Get instance ID from path
        instance_id = os.path.basename(os.path.dirname(instance_path))
        
        # Scale to consistent height while maintaining aspect ratio
        img_h, img_w = instance_img.shape[:2]
        print(instance_path, instance_img.shape)
        aspect_ratio = img_w / img_h
        scaled_h = instance_height
        scaled_w = int(scaled_h * aspect_ratio)
        
        # Allocate 60% of the cell width to the image, 40% to the text box
        # but ensure the image width doesn't exceed this allocation
        max_img_width = int(cell_w * 0.6) - 2 * cell_padding
        if scaled_w > max_img_width:
            scaled_w = max_img_width
            scaled_h = int(scaled_w / aspect_ratio)
        
        # Define margin between instance image and text box
        margin_between = 20  # 20px margin between instance and text box
            
        # Text box width (fills remaining space, accounting for margin)
        text_box_width = global_w_scaled - scaled_w - 2 * cell_padding - margin_between
        
        # Resize instance image with alpha channel
        instance_resized = cv2.resize(instance_img, (scaled_w, scaled_h))
        
        # Calculate position with reduced top margin
        x_offset = cell_x + cell_padding
        y_offset = cell_y + top_margin  # Use smaller top margin instead of centering vertically
        
        # Place instance image on panel with alpha blending
        if instance_resized.shape[2] == 4:  # Has alpha channel
            # Extract RGB and alpha channels
            rgb = instance_resized[:, :, :3]
            alpha = instance_resized[:, :, 3] / 255.0
            
            # Create 3D alpha for blending
            alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
            
            # Get the area on the instance panel where we'll place the image
            panel_area = instance_panel[y_offset:y_offset+scaled_h, x_offset:x_offset+scaled_w]
            
            # Blend the instance with the panel based on alpha
            blended = (rgb * alpha_3d) + (panel_area * (1 - alpha_3d))
            
            # Place the blended result back on the panel
            instance_panel[y_offset:y_offset+scaled_h, x_offset:x_offset+scaled_w] = blended.astype(np.uint8)
        else:
            # Fallback for RGB images without alpha
            instance_panel[y_offset:y_offset+scaled_h, x_offset:x_offset+scaled_w] = instance_resized
        
        # Draw a light blue border around the instance image
        cv2.rectangle(instance_panel, 
                     (x_offset, y_offset), 
                     (x_offset+scaled_w, y_offset+scaled_h), 
                     (80, 80, 255), 12)  # Dark blue color in BGR, 10px thickness
        
        # Create text box with dark gray border
        text_box_x = x_offset + scaled_w + margin_between
        text_box_y = y_offset
        
        # Draw text box with same background color as panel
        cv2.rectangle(instance_panel,
                     (text_box_x, text_box_y),
                     (text_box_x + text_box_width, text_box_y + scaled_h),
                     bg_color, -1)
        
        # Draw dark gray border around text box (12px width)
        cv2.rectangle(instance_panel,
                     (text_box_x, text_box_y),
                     (text_box_x + text_box_width, text_box_y + scaled_h),
                     (80, 80, 255), 12)
        
        # Add instance ID label
        if add_labels:
            # Initial label with just the ID
            id_label = f"Answer: Oreochromis mossambicus"
            
            # Descriptive text for fish (without the thinking tag)
            description = "<think>Observing the fish: it has a round head, blunt snout, terminal mouth slightly upturned, and medium lateral eyes-unlike the pointed snout of trout or the superior mouth of mosquitofish. Its scales are large and rough with fine serrations, different from the small, smooth scales of guppies. The body is deep and laterally compressed, over 20 cm long, with dark vertical bars on a silvery-yellow side and olive back-distinct from trout spots or carp coloring. The single dorsal fin has 13 to 15 spines and 10 to 12 soft rays. It swims mainly using its pectoral fins at 2 to 3 Hz, with minimal body flexing-unlike carp or trout. In territorial defense or brooding, females show rapid pectoral and pelvic fin vibrations-a behavior typical of mouthbrooders. No barbels, and the color pattern further narrow it down. All signs point to Mozambique tilapia (Oreochromis mossambicus).</think>"
            
            # Font settings - use a single font scale for both id_label and description
            font_scale = 2.0
            line_spacing = 20
            
            # Get text sizes
            title_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            
            # Starting positions for text
            text_x = text_box_x + 15  # Padding from left edge
            title_y = text_box_y + 80  # Padding from top
            
            # Draw the ID text
            cv2.putText(instance_panel, id_label, (text_x, title_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 220), 4)  # Increased thickness from 2 to 4
            
            # Calculate available height for description
            available_height = scaled_h - (title_y - text_box_y) - title_size[1] - 40  # 40px bottom padding
            
            # Word wrap for description
            words = description.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                test_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                
                if test_size[0] <= text_box_width - 30:  # 30px padding
                    current_line.append(word)
                else:
                    if current_line:  # Avoid empty lines
                        lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:  # Add the last line
                lines.append(' '.join(current_line))
            
            # Draw each line of the description
            y_offset = title_y + title_size[1] + 20  # Start below the title with spacing
            
            # Calculate total height needed for all lines
            line_heights = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][1] for line in lines]
            total_height_needed = sum(line_heights) + line_spacing * (len(lines) - 1)
            
            # Adjust font scale if needed to fit vertically
            if total_height_needed > available_height and len(lines) > 0:
                # Reduce font scale to fit
                adjusted_font_scale = font_scale * (available_height / total_height_needed) * 0.9  # 10% margin
                
                # Redraw the ID label with adjusted font scale
                # Clear the previous ID label area
                cv2.rectangle(instance_panel, 
                             (text_x-2, title_y-title_size[1]-2),
                             (text_x+title_size[0]+2, title_y+2),
                             bg_color, -1)
                            
                # Redraw ID label with adjusted scale
                cv2.putText(instance_panel, id_label, (text_x, title_y),
                           cv2.FONT_HERSHEY_SIMPLEX, adjusted_font_scale, (0, 0, 220), 4)  # Increased thickness from 2 to 4
                
                # Recalculate title size with new font scale
                title_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, adjusted_font_scale, 2)[0]
                
                # Recalculate line breaks with new font scale
                lines = []
                current_line = []
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    test_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, adjusted_font_scale, 2)[0]
                    
                    if test_size[0] <= text_box_width - 30:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Use adjusted font scale for all text
                font_scale = adjusted_font_scale
            
            # Draw the lines
            for line in lines:
                text_height = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][1]
                y_offset += text_height + line_spacing
                cv2.putText(instance_panel, line, (text_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 220), 2)
    
    # Create the final canvas
    final_width = max(global_w_scaled, instance_panel_w)
    final_height = global_h_scaled + instance_panel_h
    final_canvas = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255
    
    # Place global frame on the top
    final_canvas[:global_h_scaled, :global_w_scaled] = global_frame_resized
    
    # Place instance panel below
    final_canvas[global_h_scaled:, :instance_panel_w] = instance_panel
    
    # Add a divider line
    cv2.line(final_canvas, 
            (0, global_h_scaled), 
            (final_width, global_h_scaled), 
            (50, 50, 50), 2)
    
    # Save the result
    cv2.imwrite(output_path, final_canvas)
    return final_canvas

def process_video_frames(global_dir, instance_dir, output_dir, 
                        canvas_height=1080, instance_grid=(2, 3), 
                        max_instances=6, fps=5):
    """
    Process all frames in a video sequence and generate demo images
    
    Args:
        global_dir: Directory containing global frame images
        instance_dir: Directory containing instance crops
        output_dir: Directory to save the demo images
        canvas_height: Height of the output canvas
        instance_grid: Tuple of (rows, cols) for instance grid
        max_instances: Maximum number of instances to show
        fps: Frames per second for optional video output
    """
    ensure_dir(output_dir)
    
    # Get all global frames
    global_frames = glob.glob(os.path.join(global_dir, "*.jpg"))
    global_frames.extend(glob.glob(os.path.join(global_dir, "*.png")))
    global_frames.sort(key=get_frame_number)
    
    print(f"Found {len(global_frames)} global frames")
    
    # Process each frame
    output_frames = []
    for frame_path in tqdm(global_frames, desc="Processing frames"):
        frame_name = os.path.basename(frame_path)
        frame_base = os.path.splitext(frame_name)[0]
        output_path = os.path.join(output_dir, f"demo_{frame_base}.jpg")
        
        # Find matching instance crops
        print("frame_path", frame_path)
        print("instance_dir", instance_dir)
        instance_paths = find_matching_instances(frame_path, instance_dir, max_instances)
        
        if not instance_paths:
            print(f"Warning: No matching instances found for {frame_name}")
            continue
        
        # Create demo image
        create_demo_image(
            frame_path, 
            instance_paths, 
            output_path, 
            canvas_height=canvas_height,
            instance_grid=instance_grid
        )
        
        output_frames.append(output_path)
    
    print(f"Processed {len(output_frames)} frames")
    
    # Optionally create a video from the output frames
    if output_frames:
        video_path = os.path.join(output_dir, "demo_video.mp4")
        create_video(output_frames, video_path, fps)

def create_video(image_paths, output_path, fps=5):
    """Create a video from a list of images"""
    if not image_paths:
        return
    
    # Read the first image to get dimensions
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"Error reading {image_paths[0]}")
        return
    
    h, w = first_img.shape[:2]
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Add each image to the video
    for img_path in tqdm(image_paths, desc="Creating video"):
        img = cv2.imread(img_path)
        if img is not None:
            video.write(img)
    
    # Release the video writer
    video.release()
    print(f"Video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate demo images with global view and instance crops")
    parser.add_argument("--global_dir", default="../data/fish-dataset/masked_image/oreochromis_mossambicus/P241210_030400_031206", help="Directory containing global frame images")
    parser.add_argument("--instance_dir", default="../data/fish-instance-dataset/video/oreochromis_mossambicus/P241210_030400_031206/", help="Directory containing instance crops")
    parser.add_argument("--output_dir", default="outputs/demo_images", help="Directory to save the demo images")
    parser.add_argument("--canvas_height", type=int, default=2160, help="Height of the output canvas")
    parser.add_argument("--grid_rows", type=int, default=3, help="Number of rows in the instance grid")
    parser.add_argument("--grid_cols", type=int, default=3, help="Number of columns in the instance grid")
    parser.add_argument("--max_instances", type=int, default=6, help="Maximum number of instances to show")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video")
    parser.add_argument("--single_frame", type=bool, default=False, help="Process only a single frame instead of a sequence")
    
    args = parser.parse_args()
    
    if args.single_frame:
        # Process a single frame
        ensure_dir(args.output_dir)
        frame_path = args.single_frame
        frame_name = os.path.basename(frame_path)
        frame_base = os.path.splitext(frame_name)[0]
        output_path = os.path.join(args.output_dir, f"{frame_base}.jpg")
        
        # Find matching instance crops
        instance_paths = find_matching_instances(frame_path, args.instance_dir, args.max_instances)
        
        if not instance_paths:
            print(f"Warning: No matching instances found for {frame_name}")
            return
        
        # Create demo image
        create_demo_image(
            frame_path, 
            instance_paths, 
            output_path, 
            canvas_height=args.canvas_height,
            instance_grid=(args.grid_rows, args.grid_cols)
        )
        print(f"Demo image created: {output_path}")
    else:
        # Process a sequence of frames
        process_video_frames(
            args.global_dir,
            args.instance_dir,
            args.output_dir,
            canvas_height=args.canvas_height,
            instance_grid=(args.grid_rows, args.grid_cols),
            max_instances=args.max_instances,
            fps=args.fps
        )

if __name__ == "__main__":
    main()
