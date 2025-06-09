import os
import sys
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
import argparse
from glob import glob
from tqdm import tqdm

# This demo shows the continuous object tracking plus reverse tracking with Grounding DINO and SAM 2
"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def process_video(video_path, annotation_det_path, annotation_mask_path, masked_image_path, masked_video_path, text_prompt="fish."):
    # Extract the fish type from the path (the first component after data/fish-dataset/video/)
    path_parts = video_path.split(os.sep)
    video_name = os.path.basename(video_path)
    
    # Create the correct paths directly based on the expected structure
    fish_type = None
    for i, part in enumerate(path_parts):
        if part == "video" and i+1 < len(path_parts):
            fish_type = path_parts[i+1]
            break
    
    if not fish_type:
        print(f"Warning: Could not determine fish type from path {video_path}")
        fish_type = "unknown"
    
    # Create output directories with correct structure
    det_output_path = os.path.join(annotation_det_path, fish_type, video_name)
    mask_output_path = os.path.join(annotation_mask_path, fish_type, video_name)
    masked_image_output_path = os.path.join(masked_image_path, fish_type, video_name)
    masked_video_output_path = os.path.join(masked_video_path, fish_type)
    
    os.makedirs(det_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)
    os.makedirs(masked_image_output_path, exist_ok=True)
    os.makedirs(masked_video_output_path, exist_ok=True)
    
    # scan all the JPEG frame names in this directory
    frame_names = []
    for f in os.listdir(video_path):
        if os.path.splitext(f)[-1].lower() in [".jpg", ".jpeg", ".png"]:
            if os.path.isfile(os.path.join(video_path, f)):
                frame_names.append(f)
    
    if not frame_names:
        print(f"No frame images found in {video_path}")
        return
    
    # Sort frames by frame number
    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        # Handle case where filenames don't follow a simple numeric pattern
        print(f"Warning: Could not sort frames by number, using lexicographic sorting")
        frame_names.sort()
    
    # Create a mapping between original filenames and 6-digit padded filenames for output
    original_to_padded = {}
    padded_to_original = {}
    
    for frame in frame_names:
        name, ext = os.path.splitext(frame)
        try:
            number = int(name)
            padded_name = f"{number:06d}{ext}"
        except ValueError:
            # If the filename doesn't convert to an integer, use it as is
            print(f"Warning: Frame name {name} is not a numeric value, using as is")
            padded_name = frame
        
        original_to_padded[frame] = padded_name
        padded_to_original[padded_name] = frame
    
    # Keep the original frame_names for file access, but use padded names for output files

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=video_path)
    step = 2  # the step to sample frames for Grounding DINO predictor

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask"  # box, mask or point
    objects_count = 0
    frame_object_count = {}
    
    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print(f"Processing {video_name} - Total frames: {len(frame_names)}")
    for start_frame_idx in range(0, len(frame_names), step):
        print(f"  Frame {start_frame_idx}/{len(frame_names)}")
        img_path = os.path.join(video_path, frame_names[start_frame_idx])
        image = Image.open(img_path).convert("RGB")
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        # Get padded filename without extension for output
        padded_base_name = original_to_padded[frame_names[start_frame_idx]].split(".")[0]
        # Use padded filename format for output files
        mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"{padded_base_name}.npy")

        # run Grounding DINO on the image
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.45,
            text_threshold=0.45,
            target_sizes=[image.size[::-1]]
        )

        # process the detection results
        input_boxes = results[0]["boxes"]
        # Filter boxes based on area ratio
        filtered_indices = []
        image_area = image.size[0] * image.size[1]  # width * height
        area_threshold = 0.15  # Adjust this threshold as needed (e.g., 1% of image area)
        
        for i, box in enumerate(input_boxes):
            # Calculate box area (box format is [x0, y0, x1, y1])
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            box_area = box_width * box_height
            area_ratio = box_area / image_area

            print("area_ratio: ", area_ratio)
            
            if area_ratio > area_threshold:
                filtered_indices.append(i)
        print("filtered_indices: ", filtered_indices)
        # Filter boxes, labels and other related data
        if filtered_indices:
            input_boxes = input_boxes[filtered_indices]
            OBJECTS = [results[0]["labels"][i] for i in filtered_indices]
        else:
            OBJECTS = []
            
        if input_boxes.shape[0] != 0:
            # prompt SAM image predictor to get the mask for the object
            image_predictor.set_image(np.array(image.convert("RGB")))

            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # convert the mask shape to (n, H, W)
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)
                
            """
            Step 3: Register each object's positive points to video predictor
            """
            # If you are using point prompts, we uniformly sample positive points based on the mask
            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")
        else:
            print(f"No object detected in frame {frame_names[start_frame_idx]}")
            mask_dict = sam2_masks

        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
        frame_object_count[start_frame_idx] = objects_count
        
        if len(mask_dict.labels) == 0:
            mask_dict.save_empty_mask_and_json(mask_output_path, det_output_path, image_name_list=frame_names[start_frame_idx:start_frame_idx+step])
            print(f"No object detected in frame {start_frame_idx}")
            continue
        else:
            video_predictor.reset_state(inference_state)

            for object_id, object_info in mask_dict.labels.items():
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state,
                        start_frame_idx,
                        object_id,
                        object_info.mask,
                    )
            
            video_segments = {}  # output the following {step} frames tracking masks
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                frame_masks = MaskDictionaryModel()
                
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0)
                    object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=mask_dict.get_target_class_name(out_obj_id), logit=mask_dict.get_target_logit(out_obj_id))
                    object_info.update_box()
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    # Use padded filename format for output files
                    padded_base_name = original_to_padded[frame_names[out_frame_idx]].split(".")[0]
                    frame_masks.mask_name = f"{padded_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)

        """
        Step 5: save the tracking masks and json files
        """
        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_output_path, frame_masks_info.mask_name), mask_img)

            json_data_path = os.path.join(det_output_path, frame_masks_info.mask_name.replace(".npy", ".json"))
            frame_masks_info.to_json(json_data_path)
    
    """
    Step 6: Perform reverse tracking
    """
    try:
        print(f"Performing reverse tracking for {video_name}")
        start_object_id = 0
        object_info_dict = {}
        for frame_idx, current_object_count in frame_object_count.items():
            try:
                # 初始化一个标志，指示是否已添加掩码
                masks_added = False
                
                if frame_idx != 0:
                    video_predictor.reset_state(inference_state)
                    image_base_name = frame_names[frame_idx].split(".")[0]
                    # Use padded filename format for output files
                    padded_base_name = original_to_padded[frame_names[frame_idx]].split(".")[0]
                    json_data_path = os.path.join(det_output_path, f"{padded_base_name}.json")
                    mask_data_path = os.path.join(mask_output_path, f"{padded_base_name}.npy")
                    
                    # 检查文件是否存在
                    if not os.path.exists(json_data_path) or not os.path.exists(mask_data_path):
                        print(f"Warning: Required files not found for frame {frame_idx}, skipping reverse tracking")
                        continue
                        
                    json_data = MaskDictionaryModel().from_json(json_data_path)
                    try:
                        mask_array = np.load(mask_data_path)
                    except Exception as e:
                        print(f"Error loading mask file for frame {frame_idx}: {str(e)}")
                        continue
                    
                    # 检查是否有新的目标需要添加
                    new_objects_count = 0
                    for object_id in range(start_object_id+1, current_object_count+1):
                        if object_id in json_data.labels:
                            object_info_dict[object_id] = json_data.labels[object_id]
                            video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
                            new_objects_count += 1
                    
                    # 如果成功添加了新的掩码，设置标志为True
                    masks_added = new_objects_count > 0
                    
                start_object_id = current_object_count
                
                # 只有在成功添加了掩码的情况下才执行反向追踪
                if masks_added:    
                    try:
                        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step*2, start_frame_idx=frame_idx, reverse=True):
                            try:
                                image_base_name = frame_names[out_frame_idx].split(".")[0]
                                # Use padded filename format for output files
                                padded_base_name = original_to_padded[frame_names[out_frame_idx]].split(".")[0]
                                json_data_path = os.path.join(det_output_path, f"{padded_base_name}.json")
                                mask_data_path = os.path.join(mask_output_path, f"{padded_base_name}.npy")
                                
                                # 检查输出文件是否存在
                                if not os.path.exists(json_data_path) or not os.path.exists(mask_data_path):
                                    print(f"Warning: Required output files not found for frame {out_frame_idx}, skipping")
                                    continue
                                    
                                try:
                                    json_data = MaskDictionaryModel().from_json(json_data_path)
                                    mask_array = np.load(mask_data_path)
                                except Exception as e:
                                    print(f"Error loading output files for frame {out_frame_idx}: {str(e)}")
                                    continue
                                
                                # merge the reverse tracking masks with the original masks
                                for i, out_obj_id in enumerate(out_obj_ids):
                                    try:
                                        out_mask = (out_mask_logits[i] > 0.0).cpu()
                                        if out_mask.sum() == 0:
                                            continue
                                        # 确保object_id在字典中存在
                                        if out_obj_id not in object_info_dict:
                                            print(f"Warning: Object ID {out_obj_id} not found in tracking dictionary, skipping")
                                            continue
                                        object_info = object_info_dict[out_obj_id]
                                        object_info.mask = out_mask[0]
                                        object_info.update_box()
                                        json_data.labels[out_obj_id] = object_info
                                        mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                                        mask_array[object_info.mask] = out_obj_id
                                    except Exception as e:
                                        print(f"Error processing object ID {out_obj_id}: {str(e)}")
                                        continue
                                
                                np.save(mask_data_path, mask_array)
                                json_data.to_json(json_data_path)
                            except Exception as e:
                                print(f"Error processing output frame {out_frame_idx}: {str(e)}")
                                continue
                    except Exception as e:
                        print(f"Error during reverse propagation for frame {frame_idx}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error processing frame {frame_idx} for reverse tracking: {str(e)}")
                continue
    except Exception as e:
        print(f"Error performing reverse tracking: {str(e)}")
        print("Continuing with visualization despite reverse tracking errors")
    
    """
    Step 7: Draw the results and save the masked images and videos
    """
    print("Drawing masks and creating visualization outputs...")
    
    # Define video output path
    output_video_name = f"{os.path.basename(video_path)}.mp4"
    output_video_path = os.path.join(masked_video_output_path, output_video_name)
    
    print("video_path", video_path)
    print("mask_output_path", mask_output_path)
    print("det_output_path", det_output_path)
    print("masked_image_output_path", masked_image_output_path)
    # Draw masks and boxes on images and save to masked_image_output_path
    CommonUtils.draw_masks_and_box_with_supervision(
        video_path,          # Source images 
        mask_output_path,    # Mask data
        det_output_path,     # JSON data
        masked_image_output_path  # Output directory for visualized frames
    )

    # Create a video from the masked images
    print(f"Creating video from masked images...")
    create_video_from_images(masked_image_output_path, output_video_path, frame_rate=10)
    
    print(f"Video saved to {output_video_path}")
    print(f"Masked images saved to {masked_image_output_path}")
    print(f"Finished processing {video_name}")

def main(args):
    # Create the base output directories
    os.makedirs(args.output_det_dir, exist_ok=True)
    os.makedirs(args.output_mask_dir, exist_ok=True)
    os.makedirs(args.output_masked_image_dir, exist_ok=True)
    os.makedirs(args.output_masked_video_dir, exist_ok=True)
    
    # Find all fish type directories
    fish_types = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    print("fish_types: ", fish_types)
    for fish_type in fish_types:
        fish_dir = os.path.join(args.input_dir, fish_type)
        print(f"Processing fish type: {fish_type}")
        
        # Look for videos or frame directories
        for item in os.listdir(fish_dir):
            item_path = os.path.join(fish_dir, item)
            if os.path.isdir(item_path):
                # Check if directory contains images
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(item_path)):
                    video_dir = item_path
                else:
                    print(f"Directory {item_path} does not contain image frames, skipping")
                    continue
            elif os.path.isfile(item_path) and item.lower().endswith(('.mp4', '.avi', '.mov')):
                video_dir = item_path
            else:
                print(f"Skipping {item_path} as it is neither a video nor a directory containing images")
                continue
            
            # Process the video directory
            try:
                print(f"Processing {video_dir}...")
                process_video(
                    video_dir,
                    args.output_det_dir,
                    args.output_mask_dir,
                    args.output_masked_image_dir,
                    args.output_masked_video_dir,
                    args.text_prompt
                )
            except Exception as e:
                print(f"Error processing {video_dir}: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grounded SAM2 Tracking Demo")
    parser.add_argument("--input_dir", type=str, default="../data/test/video", 
                        help="Input directory containing video folders")
    parser.add_argument("--output_det_dir", type=str, default="../data/test/annotation_det", 
                        help="Output directory for detection JSON files")
    parser.add_argument("--output_mask_dir", type=str, default="../data/test/annotation_mask", 
                        help="Output directory for mask files")
    parser.add_argument("--output_masked_image_dir", type=str, default="../data/test/masked_image", 
                        help="Output directory for visualized mask images")
    parser.add_argument("--output_masked_video_dir", type=str, default="../data/test/masked_video", 
                        help="Output directory for visualized mask videos")
    parser.add_argument("--text_prompt", type=str, default="fish.", 
                        help="Text prompt for object detection (must end with a period)")
    args = parser.parse_args()

    # init sam image predictor and video predictor model
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # init grounding dino model from huggingface
    model_id = "./checkpoints/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    main(args)