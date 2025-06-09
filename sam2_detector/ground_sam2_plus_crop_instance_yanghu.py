#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import json
import torch
import numpy as np
import supervision as sv
from PIL import Image
import argparse
from tqdm import tqdm
from collections import defaultdict
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import copy

# Helper functions for instance cropping
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def is_valid_box(box):
    """Check if a bounding box is valid (width and height > 0)"""
    x1, y1, x2, y2 = box
    return (x1 < x2 and y1 < y2) and not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0)

def calculate_instance_sizes(json_dir, padding=10, scale_factor=1.0, percentile=80):
    """Pre-scan JSON files to determine optimal window size for each instance ID"""
    instance_boxes = defaultdict(list)
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    json_files.sort()
    
    print("Pre-scanning bounding boxes to determine optimal window sizes...")
    for json_file in tqdm(json_files, desc="Scanning JSON files"):
        json_path = os.path.join(json_dir, json_file)
        
        try:
            with open(json_path, 'r') as f:
                detection_data = json.load(f)
                
            if "labels" not in detection_data:
                continue
                
            for obj_id, obj_data in detection_data["labels"].items():
                instance_id = obj_data.get("instance_id", int(obj_id))
                
                x1 = int(obj_data.get("x1", 0))
                y1 = int(obj_data.get("y1", 0))
                x2 = int(obj_data.get("x2", 0))
                y2 = int(obj_data.get("y2", 0))
                
                if is_valid_box((x1, y1, x2, y2)):
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    instance_boxes[instance_id].append((x1, y1, x2, y2, center_x, center_y, width, height))
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    instance_sizes = {}
    
    small_obj_threshold = 150
    large_obj_threshold = 500
    max_padding = padding
    
    for instance_id, boxes in instance_boxes.items():
        if not boxes:
            continue
            
        widths = [box[6] for box in boxes]
        heights = [box[7] for box in boxes]
        
        if len(widths) >= 5:
            width_percentile = np.percentile(widths, percentile)
            height_percentile = np.percentile(heights, percentile)
        else:
            width_percentile = max(widths)
            height_percentile = max(heights)
        
        max_width = max(widths)
        max_height = max(heights)
        
        obj_size = max(width_percentile, height_percentile)
        
        if obj_size <= small_obj_threshold:
            dynamic_padding = max_padding
        elif obj_size >= large_obj_threshold:
            dynamic_padding = 0
        else:
            ratio = (obj_size - small_obj_threshold) / (large_obj_threshold - small_obj_threshold)
            dynamic_padding = int(max_padding * (1 - ratio))
        
        window_width = int(width_percentile * scale_factor) + dynamic_padding * 2
        window_height = int(height_percentile * scale_factor) + dynamic_padding * 2
        
        min_width_threshold = max_width * 0.8
        min_height_threshold = max_height * 0.8
        
        window_width = max(window_width, int(min_width_threshold) + dynamic_padding * 2)
        window_height = max(window_height, int(min_height_threshold) + dynamic_padding * 2)
        
        crop_size = max(window_width, window_height)
        window_width = crop_size
        window_height = crop_size
        
        window_width = (window_width + 1) // 2 * 2
        window_height = (window_height + 1) // 2 * 2
        
        instance_sizes[instance_id] = (window_width, window_height, max_width, max_height)
    
    return instance_sizes

def get_centered_crop_coordinates(image_shape, center_x, center_y, crop_width, crop_height, obj_width, obj_height):
    """Calculate crop coordinates ensuring the crop is a square with the object centered"""
    img_height, img_width = image_shape[:2]
    
    crop_size = max(crop_width, crop_height)
    
    x1 = int(center_x - crop_size / 2)
    y1 = int(center_y - crop_size / 2)
    
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x1 + crop_size > img_width:
        x1 = max(0, img_width - crop_size)
    if y1 + crop_size > img_height:
        y1 = max(0, img_height - crop_size)
    
    x2 = min(img_width, x1 + crop_size)
    y2 = min(img_height, y1 + crop_size)
    
    if x2 - x1 < crop_size and x1 > 0:
        x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size and y1 > 0:
        y1 = max(0, y2 - crop_size)
    
    return int(x1), int(y1), int(x2), int(y2)

def process_frame(frame, annotation_path, mask_path, output_path, frame_idx, instance_sizes, min_size, padding, fixed_window, class_filter, valid_instance_ids=None):
    """Process a single frame and extract instance crops"""
    instances_processed = 0
    
    counter_file = os.path.join(output_path, "frame_counters.json")
    instance_frame_counters = {}
    
    if os.path.exists(counter_file):
        try:
            with open(counter_file, 'r') as f:
                instance_frame_counters = json.load(f)
        except:
            instance_frame_counters = {}
    
    mask = None
    if mask_path and os.path.exists(mask_path):
        try:
            mask = np.load(mask_path)
        except Exception as e:
            print(f"  Error loading mask file {mask_path}: {str(e)}")
    
    try:
        with open(annotation_path, 'r') as f:
            detection_data = json.load(f)
        
        if "labels" in detection_data:
            for obj_id, obj_data in detection_data["labels"].items():
                class_name = obj_data.get("class_name", "unknown")
                instance_id = obj_data.get("instance_id", int(obj_id))
                instance_id_str = str(instance_id)
                
                # 跳过不在有效实例ID列表中的物体
                if valid_instance_ids is not None and instance_id not in valid_instance_ids:
                    continue
                
                if class_filter is not None and class_name not in class_filter:
                    continue
                
                instance_dir = os.path.join(output_path, instance_id_str)
                ensure_dir(instance_dir)
                
                x1 = int(obj_data.get("x1", 0))
                y1 = int(obj_data.get("y1", 0))
                x2 = int(obj_data.get("x2", 0))
                y2 = int(obj_data.get("y2", 0))
                
                if not is_valid_box((x1, y1, x2, y2)):
                    continue
                
                obj_width = x2 - x1
                obj_height = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                if obj_width < min_size or obj_height < min_size:
                    continue
                
                img_height, img_width = frame.shape[:2]
                
                if fixed_window and instance_id in instance_sizes:
                    crop_width, crop_height, max_obj_width, max_obj_height = instance_sizes[instance_id]
                    
                    crop_size = max(crop_width, crop_height)
                    crop_width = crop_size
                    crop_height = crop_size
                    
                    x1, y1, x2, y2 = get_centered_crop_coordinates(
                        frame.shape, center_x, center_y, 
                        crop_width, crop_height,
                        obj_width, obj_height
                    )
                else:
                    scale = 1.0
                    crop_size = max(int(obj_width * scale) + (padding * 2), 
                                   int(obj_height * scale) + (padding * 2))
                    
                    crop_size = max(crop_size, padding * 4)
                    crop_size = (crop_size + 1) // 2 * 2
                    
                    x1, y1, x2, y2 = get_centered_crop_coordinates(
                        frame.shape, center_x, center_y, 
                        crop_size, crop_size,
                        obj_width, obj_height
                    )
                
                if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0 or x1 >= img_width or y1 >= img_height:
                    continue
                
                cropped_img = frame[y1:y2, x1:x2]
                if cropped_img.size == 0:
                    continue
                
                if mask is not None:
                    try:
                        if mask.shape[:2] == frame.shape[:2]:
                            cropped_mask = mask[y1:y2, x1:x2]
                            
                            instance_mask = None
                            
                            if cropped_mask.size > 0:
                                if np.issubdtype(mask.dtype, np.integer):
                                    instance_mask = (cropped_mask == instance_id).astype(np.uint8)
                                elif mask.dtype == np.bool_ or (mask.dtype == np.uint8 and np.max(mask) <= 1):
                                    instance_mask = cropped_mask.astype(np.uint8)
                                elif len(mask.shape) == 3 and mask.shape[2] > 1:
                                    for channel in range(mask.shape[2]):
                                        if np.any(cropped_mask[:,:,channel]):
                                            instance_mask = cropped_mask[:,:,channel].astype(np.uint8)
                                            break
                            
                            if instance_mask is not None and instance_mask.size > 0:
                                original_img = cropped_img.copy()
                                
                                dilated_mask = instance_mask.copy()
                                fish_type = mask_path.split("/")[4] if "/" in mask_path else mask_path.split(os.sep)[4]
                                if fish_type in ["guppy", "mosquitofish"]:
                                    kernel_size = 20
                                else:
                                    kernel_size = 1
                                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=1)
                                
                                if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                                    rgba_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)
                                    rgba_img[:, :, 3] = dilated_mask * 255
                                    
                                    instance_frame_idx = instance_frame_counters.get(instance_id_str, 0)
                                    bg_removed_filename = f"{instance_frame_idx:06d}.webp"
                                    bg_removed_path = os.path.join(instance_dir, bg_removed_filename)
                                    cv2.imwrite(bg_removed_path, rgba_img)
                                    
                                elif len(original_img.shape) == 2:
                                    rgba_img = cv2.merge([original_img, original_img, original_img, dilated_mask * 255])
                                    
                                    instance_frame_idx = instance_frame_counters.get(instance_id_str, 0)
                                    bg_removed_filename = f"{instance_frame_idx:06d}.webp"
                                    bg_removed_path = os.path.join(instance_dir, bg_removed_filename)
                                    cv2.imwrite(bg_removed_path, rgba_img)
                    except Exception as e:
                        print(f"  Error applying mask for instance {instance_id}: {str(e)}")
                
                instance_frame_idx = instance_frame_counters.get(instance_id_str, 0)
                instance_frame_counters[instance_id_str] = instance_frame_idx + 1
                instances_processed += 1
        
        with open(counter_file, 'w') as f:
            json.dump(instance_frame_counters, f)
            
    except Exception as e:
        print(f"  Error processing frame {frame_idx}: {str(e)}")
    
    return instances_processed

def count_instance_occurrences(det_output_path, min_frames=100):
    """
    统计每个实例ID出现的帧数，并返回出现次数超过阈值的实例ID列表
    
    Args:
        det_output_path: 检测结果JSON文件目录
        min_frames: 最小帧数阈值，默认100
    
    Returns:
        valid_instance_ids: 出现次数超过阈值的实例ID列表
        instance_counts: 每个实例ID出现的帧数
    """
    instance_counts = defaultdict(int)
    json_files = [f for f in os.listdir(det_output_path) if f.endswith('.json')]
    
    print(f"Counting instance occurrences in {len(json_files)} JSON files...")
    for json_file in tqdm(json_files, desc="Analyzing tracking length"):
        json_path = os.path.join(det_output_path, json_file)
        
        try:
            with open(json_path, 'r') as f:
                detection_data = json.load(f)
            
            if "labels" in detection_data:
                for obj_id, obj_data in detection_data["labels"].items():
                    instance_id = obj_data.get("instance_id", int(obj_id))
                    instance_counts[instance_id] += 1
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # 过滤出现次数超过阈值的实例ID
    valid_instance_ids = [instance_id for instance_id, count in instance_counts.items() if count >= min_frames]
    filtered_count = len(instance_counts) - len(valid_instance_ids)
    
    print(f"Found {len(instance_counts)} unique instances.")
    print(f"Keeping {len(valid_instance_ids)} instances with >= {min_frames} frames.")
    print(f"Filtered out {filtered_count} instances with < {min_frames} frames.")
    
    # 输出每个有效实例的帧数
    if valid_instance_ids:
        print("\nValid instances and their frame counts:")
        for instance_id in valid_instance_ids:
            print(f"  Instance {instance_id}: {instance_counts[instance_id]} frames")
    
    return valid_instance_ids, instance_counts

def process_video(video_path, annotation_det_path, annotation_mask_path, instance_output_path, text_prompt="fish.", 
                do_tracking=True, do_cropping=True, cleanup_temp=True, enable_visualization=True,
                output_masked_image_dir=None, output_masked_video_dir=None, min_tracking_frames=100, **crop_params):
    """Process a video with tracking and instance cropping"""
    # Extract fish type from path
    path_parts = video_path.split(os.sep)
    video_name = os.path.basename(video_path)
    
    fish_type = None
    for i, part in enumerate(path_parts):
        if part == "video" and i+1 < len(path_parts):
            fish_type = path_parts[i+1]
            break
    
    if not fish_type:
        print(f"Warning: Could not determine fish type from path {video_path}")
        fish_type = "unknown"
    
    # Create output directories
    det_output_path = os.path.join(annotation_det_path, fish_type, video_name)
    mask_output_path = os.path.join(annotation_mask_path, fish_type, video_name)
    instance_video_output_path = os.path.join(instance_output_path, fish_type, video_name)
    
    # Only create visualization paths if visualization is enabled
    masked_image_output_path = None
    masked_video_output_path = None
    if enable_visualization and output_masked_image_dir and output_masked_video_dir:
        masked_image_output_path = os.path.join(output_masked_image_dir, fish_type, video_name)
        masked_video_output_path = os.path.join(output_masked_video_dir, fish_type)
    
    # Create temporary directories if they'll be cleaned up
    temp_dirs = []
    if cleanup_temp:
        # Only create these as temp dirs if we're going to clean them up
        if do_tracking:
            temp_dirs.append(mask_output_path)
            if enable_visualization and masked_image_output_path:
                temp_dirs.append(masked_image_output_path)
    
    ensure_dir(det_output_path)
    ensure_dir(mask_output_path)
    if do_cropping:
        ensure_dir(instance_video_output_path)
    
    # Create visualization directories if enabled
    if enable_visualization and masked_image_output_path and masked_video_output_path:
        ensure_dir(masked_image_output_path)
        ensure_dir(masked_video_output_path)
    
    # Find all frame images
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
        print(f"Warning: Could not sort frames by number, using lexicographic sorting")
        frame_names.sort()
    
    # Create filename mapping
    original_to_padded = {}
    padded_to_original = {}
    
    for frame in frame_names:
        name, ext = os.path.splitext(frame)
        try:
            number = int(name)
            padded_name = f"{number:06d}{ext}"
        except ValueError:
            print(f"Warning: Frame name {name} is not a numeric value, using as is")
            padded_name = frame
        
        original_to_padded[frame] = padded_name
        padded_to_original[padded_name] = frame
    
    # PART 1: Object Detection and Tracking
    if do_tracking:
        # Initialize video predictor state
        inference_state = video_predictor.init_state(video_path=video_path)
        step = 10  # the step to sample frames for Grounding DINO predictor
        
        sam2_masks = MaskDictionaryModel()
        PROMPT_TYPE_FOR_VIDEO = "mask"
        objects_count = 0
        frame_object_count = {}
        
        print(f"Processing {video_name} - Total frames: {len(frame_names)}")
        for start_frame_idx in range(0, len(frame_names), step):
            print(f"  Frame {start_frame_idx}/{len(frame_names)}")
            img_path = os.path.join(video_path, frame_names[start_frame_idx])
            image = Image.open(img_path).convert("RGB")
            padded_base_name = original_to_padded[frame_names[start_frame_idx]].split(".")[0]
            mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"{padded_base_name}.npy")
            
            # Run Grounding DINO on the image
            inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)
            
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.24,
                text_threshold=0.24,
                target_sizes=[image.size[::-1]]
            )
            
            # Prompt SAM image predictor
            image_predictor.set_image(np.array(image.convert("RGB")))
            
            input_boxes = results[0]["boxes"]
            OBJECTS = results[0]["labels"]
            if input_boxes.shape[0] != 0:
                masks, scores, logits = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                
                if masks.ndim == 2:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)
                
                if mask_dict.promote_type == "mask":
                    mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
                else:
                    raise NotImplementedError("SAM 2 video predictor only support mask prompts")
            else:
                print(f"No object detected in frame {frame_names[start_frame_idx]}")
                mask_dict = sam2_masks
            
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
                
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                    frame_masks = MaskDictionaryModel()
                    
                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0)
                        object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=mask_dict.get_target_class_name(out_obj_id), logit=mask_dict.get_target_logit(out_obj_id))
                        object_info.update_box()
                        frame_masks.labels[out_obj_id] = object_info
                        padded_base_name = original_to_padded[frame_names[out_frame_idx]].split(".")[0]
                        frame_masks.mask_name = f"{padded_base_name}.npy"
                        frame_masks.mask_height = out_mask.shape[-2]
                        frame_masks.mask_width = out_mask.shape[-1]
                    
                    video_segments[out_frame_idx] = frame_masks
                    sam2_masks = copy.deepcopy(frame_masks)
            
            # Save tracking masks and json files
            for frame_idx, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                for obj_id, obj_info in mask.items():
                    mask_img[obj_info.mask == True] = obj_id
                
                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(mask_output_path, frame_masks_info.mask_name), mask_img)
                
                json_data_path = os.path.join(det_output_path, frame_masks_info.mask_name.replace(".npy", ".json"))
                frame_masks_info.to_json(json_data_path)
        
        # Perform reverse tracking
        try:
            print(f"Performing reverse tracking for {video_name}")
            start_object_id = 0
            object_info_dict = {}
            for frame_idx, current_object_count in frame_object_count.items():
                try:
                    masks_added = False
                    
                    if frame_idx != 0:
                        video_predictor.reset_state(inference_state)
                        padded_base_name = original_to_padded[frame_names[frame_idx]].split(".")[0]
                        json_data_path = os.path.join(det_output_path, f"{padded_base_name}.json")
                        mask_data_path = os.path.join(mask_output_path, f"{padded_base_name}.npy")
                        
                        if not os.path.exists(json_data_path) or not os.path.exists(mask_data_path):
                            print(f"Warning: Required files not found for frame {frame_idx}, skipping reverse tracking")
                            continue
                            
                        json_data = MaskDictionaryModel().from_json(json_data_path)
                        try:
                            mask_array = np.load(mask_data_path)
                        except Exception as e:
                            print(f"Error loading mask file for frame {frame_idx}: {str(e)}")
                            continue
                        
                        new_objects_count = 0
                        for object_id in range(start_object_id+1, current_object_count+1):
                            if object_id in json_data.labels:
                                object_info_dict[object_id] = json_data.labels[object_id]
                                video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
                                new_objects_count += 1
                        
                        masks_added = new_objects_count > 0
                    
                    start_object_id = current_object_count
                    
                    if masks_added:
                        try:
                            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step*2, start_frame_idx=frame_idx, reverse=True):
                                try:
                                    padded_base_name = original_to_padded[frame_names[out_frame_idx]].split(".")[0]
                                    json_data_path = os.path.join(det_output_path, f"{padded_base_name}.json")
                                    mask_data_path = os.path.join(mask_output_path, f"{padded_base_name}.npy")
                                    
                                    if not os.path.exists(json_data_path) or not os.path.exists(mask_data_path):
                                        print(f"Warning: Required output files not found for frame {out_frame_idx}, skipping")
                                        continue
                                        
                                    try:
                                        json_data = MaskDictionaryModel().from_json(json_data_path)
                                        mask_array = np.load(mask_data_path)
                                    except Exception as e:
                                        print(f"Error loading output files for frame {out_frame_idx}: {str(e)}")
                                        continue
                                    
                                    for i, out_obj_id in enumerate(out_obj_ids):
                                        try:
                                            out_mask = (out_mask_logits[i] > 0.0).cpu()
                                            if out_mask.sum() == 0:
                                                continue
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
        
        # Draw results and create videos if visualization is enabled
        if enable_visualization and masked_image_output_path and masked_video_output_path:
            print("Drawing masks and creating visualization outputs...")
            
            output_video_name = f"{os.path.basename(video_path)}.mp4"
            output_video_path = os.path.join(masked_video_output_path, output_video_name)
            
            print("video_path", video_path)
            print("mask_output_path", mask_output_path)
            print("det_output_path", det_output_path)
            print("masked_image_output_path", masked_image_output_path)
            
            CommonUtils.draw_masks_and_box_with_supervision(
                video_path,
                mask_output_path,
                det_output_path,
                masked_image_output_path,
                draw_labels=False,
                draw_boxes=False
            )
            
            print(f"Creating video from masked images...")
            create_video_from_images(masked_image_output_path, output_video_path, frame_rate=15)
            
            print(f"Video saved to {output_video_path}")
            print(f"Masked images saved to {masked_image_output_path}")
    
    # PART 2: Instance Cropping
    if do_cropping:
        if not os.path.exists(det_output_path) or not os.path.exists(mask_output_path):
            print(f"Detection or mask files do not exist for {video_name}, skipping instance cropping")
            return
        
        print(f"Cropping instances from {video_name}")
        
        # 统计每个实例ID出现的帧数，并只处理出现次数超过阈值的实例
        valid_instance_ids, instance_counts = count_instance_occurrences(det_output_path, min_frames=min_tracking_frames)
        
        if not valid_instance_ids:
            print(f"No instances with tracking length >= {min_tracking_frames} found. Skipping cropping.")
            return
        
        # Calculate optimal window sizes for each instance
        instance_sizes = {}
        if crop_params.get('fixed_window', True):
            instance_sizes = calculate_instance_sizes(
                det_output_path, 
                padding=crop_params.get('padding', 10),
                scale_factor=crop_params.get('scale_factor', 1.0),
                percentile=crop_params.get('percentile', 80)
            )
        
        # Process frames
        total_instances = 0
        for frame_idx, frame_name in enumerate(tqdm(frame_names, desc="Cropping instances")):
            # Load the image
            frame_path = os.path.join(video_path, frame_name)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Error: Cannot read frame {frame_path}")
                continue
            
            # Find annotation file
            padded_base_name = original_to_padded[frame_name].split(".")[0]
            json_path = os.path.join(det_output_path, f"{padded_base_name}.json")
            mask_path = os.path.join(mask_output_path, f"{padded_base_name}.npy")
            
            if not os.path.exists(json_path):
                continue
            
            # Process the frame
            instances_processed = process_frame(
                frame,
                json_path,
                mask_path if os.path.exists(mask_path) else None,
                instance_video_output_path,
                frame_idx,
                instance_sizes,
                crop_params.get('min_size', 32),
                crop_params.get('padding', 10),
                crop_params.get('fixed_window', True),
                crop_params.get('class_filter', ["fish", "carp"]),
                valid_instance_ids=valid_instance_ids  # 传递有效实例ID列表
            )
            
            total_instances += instances_processed
        
        print(f"Total instances cropped: {total_instances}")
    
    # Clean up temporary directories
    if cleanup_temp:
        print(f"Cleaning up temporary directories...")
        
        # Delete temporary directories
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                print(f"Removing: {temp_dir}")
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Error removing directory {temp_dir}: {str(e)}")
        
        # Delete parent directories if they're empty
        if do_tracking and mask_output_path:
            parent_dir = os.path.dirname(mask_output_path)
            try:
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    print(f"Removing empty parent directory: {parent_dir}")
                    os.rmdir(parent_dir)
                
                grandparent_dir = os.path.dirname(parent_dir)
                if os.path.exists(grandparent_dir) and not os.listdir(grandparent_dir):
                    print(f"Removing empty grandparent directory: {grandparent_dir}")
                    os.rmdir(grandparent_dir)
            except Exception as e:
                print(f"Error removing parent directories: {str(e)}")
        
        # For masked_image directories
        if enable_visualization and masked_image_output_path:
            parent_dir = os.path.dirname(masked_image_output_path)
            try:
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    print(f"Removing empty parent directory: {parent_dir}")
                    os.rmdir(parent_dir)
                
                grandparent_dir = os.path.dirname(parent_dir)
                if os.path.exists(grandparent_dir) and not os.listdir(grandparent_dir):
                    print(f"Removing empty grandparent directory: {grandparent_dir}")
                    os.rmdir(grandparent_dir)
            except Exception as e:
                print(f"Error removing parent directories: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Grounded SAM2 Tracking with Instance Cropping")
    parser.add_argument("--input_dir", type=str, default="../data/fish-dataset-yanghu/video", 
                        help="Input directory containing video folders")
    parser.add_argument("--output_det_dir", type=str, default="../data/fish-dataset-yanghu/annotation_det", 
                        help="Output directory for detection JSON files")
    parser.add_argument("--output_mask_dir", type=str, default="../data/fish-dataset-yanghu/annotation_mask", 
                        help="Output directory for mask files")
    parser.add_argument("--output_masked_image_dir", type=str, default="../data/fish-dataset-yanghu/masked_image", 
                        help="Output directory for visualized mask images")
    parser.add_argument("--output_masked_video_dir", type=str, default="../data/fish-dataset-yanghu/masked_video", 
                        help="Output directory for visualized mask videos")
    parser.add_argument("--output_instance_dir", type=str, default="../data/fish-instance-dataset-yanghu/video", 
                        help="Output directory for instance crops")
    parser.add_argument("--text_prompt", type=str, default="fish.", 
                        help="Text prompt for object detection (must end with a period)")
    parser.add_argument("--do_tracking", action="store_true", default=True,
                        help="Perform tracking")
    parser.add_argument("--do_cropping", action="store_true", default=True,
                        help="Perform instance cropping")
    parser.add_argument("--cleanup_temp", action="store_true", default=True,
                        help="Clean up intermediate directories after processing")
    parser.add_argument("--enable_visualization", action="store_true", default=True,
                        help="Enable visualization output (masked images and videos)")
    parser.add_argument("--min_tracking_frames", type=int, default=10,
                        help="Minimum number of frames an object needs to be tracked to be cropped (default: 100)")
    parser.add_argument("--padding", type=int, default=50,
                        help="Padding around instance crops")
    parser.add_argument("--min_size", type=int, default=32,
                        help="Minimum size of object to crop")
    parser.add_argument("--fixed_window", action="store_true", default=True,
                        help="Use fixed window size for each instance ID")
    parser.add_argument("--scale_factor", type=float, default=1.0,
                        help="Factor to increase window size to prevent truncation")
    parser.add_argument("--percentile", type=int, default=80,
                        help="Percentile to use for window size (0-100)")
    parser.add_argument("--class_filter", nargs='+', default=["fish", "carp"],
                        help="List of class names to crop")
    parser.add_argument("--fish_types", nargs='+', default=None,
                        help="Specific fish types (subdirectories) to process. If not provided, all fish types will be processed.")
    args = parser.parse_args()

    # Create base output directories
    required_dirs = [args.output_det_dir]
    if not args.cleanup_temp:
        required_dirs.append(args.output_mask_dir)
    if args.do_cropping:
        required_dirs.append(args.output_instance_dir)
    
    # Add visualization directories if visualization is enabled
    if args.enable_visualization:
        required_dirs.extend([args.output_masked_image_dir, args.output_masked_video_dir])
    
    for directory in required_dirs:
        ensure_dir(directory)
    
    # Find all fish type directories
    all_fish_types = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    # Filter fish types if specified
    if args.fish_types:
        fish_types = [ft for ft in all_fish_types if ft in args.fish_types]
        if not fish_types:
            print(f"Warning: None of the specified fish types {args.fish_types} were found in {args.input_dir}")
            print(f"Available fish types: {all_fish_types}")
            return
        print(f"Processing only specified fish types: {fish_types}")
    else:
        fish_types = all_fish_types
        print(f"Processing all fish types: {fish_types}")
    
    # Crop parameters
    crop_params = {
        'padding': args.padding,
        'min_size': args.min_size,
        'fixed_window': args.fixed_window,
        'scale_factor': args.scale_factor,
        'percentile': args.percentile,
        'class_filter': args.class_filter
    }
    
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
                
                # Only pass visualization directories if visualization is enabled
                output_masked_image_dir = args.output_masked_image_dir if args.enable_visualization else None
                output_masked_video_dir = args.output_masked_video_dir if args.enable_visualization else None
                process_video(
                    video_dir,
                    args.output_det_dir,
                    args.output_mask_dir,
                    args.output_instance_dir,
                    args.text_prompt,
                    args.do_tracking,
                    args.do_cropping,
                    args.cleanup_temp,
                    args.enable_visualization,
                    output_masked_image_dir,
                    output_masked_video_dir,
                    args.min_tracking_frames,
                    **crop_params
                )
            except Exception as e:
                print(f"Error processing {video_dir}: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    # Init models
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Use bfloat16 for better performance if supported
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # Init grounding dino model from huggingface
    model_id = "./checkpoints/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    main()
