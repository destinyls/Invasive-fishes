import os
import json
import glob
from tqdm import tqdm
import random

choices = {"brown_trout": "A. brown trout", "crucian_carp": "B. crucian carp", "guppies": "C. guppies", "mosquitofish": "D. mosquitofish", "rainbow_trout": "E. rainbow trout", "carp": "F. carp", "grass_carp": "G. grass carp", "largemouth_bass": "H. largemouth bass", "oreochromis_mossambicus": "I. oreochromis mossambicus"}

def process_video_files(input_dir, num_samples=100000):
    """Process video files and generate dataset entries"""
    video_files = glob.glob(os.path.join(input_dir, "video", "**", "*.mp4"), recursive=True)
    
    dataset = []
    for video_file in tqdm(video_files):
        rel_path = os.path.relpath(video_file, input_dir)
        fish_name = rel_path.split("/")[-2]
        entry = {
            "problem_id": len(dataset),
            "problem": "Describe what kind of fish is in this video.",
            "data_type": "video",
            "problem_type": "multiple choice",
            "options": ["A. brown trout", "B. crucian carp", "C. guppies", "D. mosquitofish", "E. rainbow trout", "F. carp", "G. grass carp", "H. largemouth bass", "I. oreochromis mossambicus"],
            "path": rel_path,
            "process": f"The fish is a {fish_name}.",
            "solution": f"<answer>{choices[fish_name]}</answer>",
            "data_source": f"video/{fish_name}",
        }
        dataset.append(entry)
        
        if len(dataset) >= num_samples:
            break
            
    return dataset

def main():
    input_dir = "src/r1-v/fish-mllms-dataset"
    output_file = os.path.join(input_dir, "fish-cot-100k.json")
    
    # Process videos and generate dataset
    dataset = process_video_files(input_dir)
    
    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Generated dataset with {len(dataset)} samples")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
