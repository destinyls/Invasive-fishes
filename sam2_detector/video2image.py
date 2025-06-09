import cv2
import os
import argparse
from tqdm import tqdm

def video_to_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from a video and save them as images.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Extract every nth frame
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Duration: {total_frames/fps:.2f}s")
    print(f"Extracting frames (every {frame_interval} frame(s))...")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    for _ in tqdm(range(total_frames)):
        ret, frame = video.read()
        
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Save frame as image
            frame_filename = os.path.join(output_dir, f"{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1
    
    # Release video
    video.release()
    
    print(f"Saved {saved_count} frames to {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--input_video", type=str, default="../data/test/193_1747723903.mp4", help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, default="../data/test/193_1747723903", help="Directory to save extracted frames")
    parser.add_argument("--interval", type=int, default=1, help="Extract every nth frame (default: 1)")
    
    args = parser.parse_args()
    
    # Extract frames from video
    video_to_frames(args.input_video, args.output_dir, args.interval)

if __name__ == "__main__":
    main()
