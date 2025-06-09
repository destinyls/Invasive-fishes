#!/usr/bin/env python3
import os
import shutil
import argparse
import random
import json
from tqdm import tqdm

def create_directory(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def prepare_cl_dataset(input_dir, output_dir, val_ratio=0.2, seed=42, max_frames_per_species=1000):
    """
    Prepare classification dataset from fish instance dataset.
    
    Args:
        input_dir (str): Path to fish-instance-dataset
        output_dir (str): Path to output fish-cl-dataset
        val_ratio (float): Ratio of validation set
        seed (int): Random seed for reproducibility
        max_frames_per_species (int): Maximum number of frames per species
    """
    random.seed(seed)
    
    # Create output directory structure
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    create_directory(train_dir)
    create_directory(val_dir)
    
    # Create files to save sequence paths
    train_sequences_file = os.path.join(output_dir, 'train_sequences.json')
    val_sequences_file = os.path.join(output_dir, 'val_sequences.json')
    
    # Initialize data structures for JSON
    train_data = {
        "description": "Training set sequences",
        "format": "species_name -> list of sequences with metadata",
        "generated_with": "prepare_cl_dataset.py",
        "total_species": 0,
        "total_sequences": 0,
        "total_frames": 0,
        "species": {}
    }
    
    val_data = {
        "description": "Validation set sequences", 
        "format": "species_name -> list of sequences with metadata",
        "generated_with": "prepare_cl_dataset.py",
        "total_species": 0,
        "total_sequences": 0,
        "total_frames": 0,
        "species": {}
    }

    # Get all species directories
    video_dir = os.path.join(input_dir, 'video')
    species_list = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
    
    print(f"Found {len(species_list)} fish species")
    
    # Process each species
    for species in species_list:
        species_video_dir = os.path.join(video_dir, species)
        
        # Create species directories in train and val
        train_species_dir = os.path.join(train_dir, species)
        val_species_dir = os.path.join(val_dir, species)
        create_directory(train_species_dir)
        create_directory(val_species_dir)
        
        # Get all sequence directories for this species
        all_sequences = []
        for video_id in os.listdir(species_video_dir):
            video_path = os.path.join(species_video_dir, video_id)
            if not os.path.isdir(video_path):
                continue
                
            for subdir in os.listdir(video_path):
                subdir_path = os.path.join(video_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue
                
                # Check if this sequence has images
                images_in_sequence = [f for f in os.listdir(subdir_path) if f.endswith('.webp')]
                if images_in_sequence:
                    all_sequences.append({
                        'path': subdir_path,
                        'images': images_in_sequence,
                        'count': len(images_in_sequence)
                    })
        
        # Sort sequences by image count (descending) for better selection when limiting frames
        all_sequences.sort(key=lambda x: x['count'], reverse=True)
        
        # Limit total number of frames per species by selecting sequences
        total_frames = sum(seq['count'] for seq in all_sequences)
        if total_frames > max_frames_per_species:
            print(f"Limiting {species} from {total_frames} to {max_frames_per_species} frames")
            selected_sequences = []
            current_frame_count = 0
            
            # Shuffle sequences first to ensure random selection within same frame counts
            random.shuffle(all_sequences)
            all_sequences.sort(key=lambda x: x['count'], reverse=True)  # Re-sort after shuffle
            
            for seq in all_sequences:
                if current_frame_count + seq['count'] <= max_frames_per_species:
                    selected_sequences.append(seq)
                    current_frame_count += seq['count']
                elif current_frame_count < max_frames_per_species:
                    # If adding this sequence exceeds limit, but we haven't reached the limit yet,
                    # we can still add it if there's room for at least one more frame
                    selected_sequences.append(seq)
                    current_frame_count += seq['count']
                    break
                else:
                    break
            
            all_sequences = selected_sequences
            total_frames = current_frame_count
            print(f"Selected {len(all_sequences)} sequences with {total_frames} total frames")
        
        # Shuffle sequences and split by sequence (not by individual images)
        random.shuffle(all_sequences)
        split_idx = int(len(all_sequences) * (1 - val_ratio))
        train_sequences = all_sequences[:split_idx]
        val_sequences = all_sequences[split_idx:]
        
        train_frame_count = sum(seq['count'] for seq in train_sequences)
        val_frame_count = sum(seq['count'] for seq in val_sequences)
        
        print(f"Species: {species}")
        print(f"  Train: {len(train_sequences)} sequences, {train_frame_count} frames")
        print(f"  Val: {len(val_sequences)} sequences, {val_frame_count} frames")
        
        # Save sequence data to JSON structures
        train_data["species"][species] = {
            "sequence_count": len(train_sequences),
            "frame_count": train_frame_count,
            "sequences": [
                {
                    "path": seq['path'],
                    "frame_count": seq['count']
                }
                for seq in train_sequences
            ]
        }
        
        val_data["species"][species] = {
            "sequence_count": len(val_sequences), 
            "frame_count": val_frame_count,
            "sequences": [
                {
                    "path": seq['path'],
                    "frame_count": seq['count']
                }
                for seq in val_sequences
            ]
        }
        
        # Update totals
        train_data["total_sequences"] += len(train_sequences)
        train_data["total_frames"] += train_frame_count
        val_data["total_sequences"] += len(val_sequences)
        val_data["total_frames"] += val_frame_count
        
        # Copy images from train sequences
        train_img_idx = 0
        for seq in tqdm(train_sequences, desc=f"Processing {species} train sequences"):
            for img_file in sorted(seq['images']):  # Sort to ensure consistent ordering
                src_path = os.path.join(seq['path'], img_file)
                dest_path = os.path.join(train_species_dir, f"{train_img_idx:06d}.webp")
                shutil.copy(src_path, dest_path)
                train_img_idx += 1
        
        # Copy images from validation sequences
        val_img_idx = 0
        for seq in tqdm(val_sequences, desc=f"Processing {species} val sequences"):
            for img_file in sorted(seq['images']):  # Sort to ensure consistent ordering
                src_path = os.path.join(seq['path'], img_file)
                dest_path = os.path.join(val_species_dir, f"{val_img_idx:06d}.webp")
                shutil.copy(src_path, dest_path)
                val_img_idx += 1
    
    # Update final totals
    train_data["total_species"] = len(species_list)
    val_data["total_species"] = len(species_list)
    
    # Save JSON files
    with open(train_sequences_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_sequences_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset prepared at {output_dir}")
    print(f"Training sequences saved to: {train_sequences_file}")
    print(f"Validation sequences saved to: {val_sequences_file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare fish classification dataset')
    parser.add_argument('--input_dir', type=str, default="../data/fish-instance-dataset", help='Path to fish-instance-dataset')
    parser.add_argument('--output_dir', type=str, default="../data/fish-cl-dataset", help='Path to output fish-cl-dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_frames', type=int, default=4600000, help='Maximum frames per species')
    args = parser.parse_args()
    
    prepare_cl_dataset(args.input_dir, args.output_dir, args.val_ratio, args.seed, args.max_frames)

if __name__ == '__main__':
    main() 
