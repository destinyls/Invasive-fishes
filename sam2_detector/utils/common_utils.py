import os
import json
import cv2
import numpy as np
import supervision as sv
import random

class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Path '{path}' did not exist and has been created.")
            else:
                print(f"Path '{path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")

    @staticmethod
    def draw_masks_and_box_with_supervision(raw_image_path, mask_path, json_path, output_path, draw_labels=True, draw_boxes=True):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            
            # Try to load mask - first without prefix, then with "mask_" prefix
            base_name = raw_image_name.split(".")[0]
            mask_npy_path = os.path.join(mask_path, f"{base_name}.npy")
            
            # If the file doesn't exist, try with the "mask_" prefix
            if not os.path.exists(mask_npy_path):
                mask_npy_path = os.path.join(mask_path, f"mask_{base_name}.npy")
                
            # If still doesn't exist, skip this image but don't fail
            if not os.path.exists(mask_npy_path):
                print(f"Warning: Mask file not found for {raw_image_name}, saving image without annotations")
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
                
            # Load the mask file
            try:
                mask = np.load(mask_npy_path)
            except Exception as e:
                print(f"Error loading mask file {mask_npy_path}: {str(e)}, saving image without annotations")
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
            
            # color map
            unique_ids = np.unique(mask)
            
            # get each mask from unique mask file
            all_object_masks = []
            for uid in unique_ids:
                if uid == 0: # skip background id
                    continue
                else:
                    object_mask = (mask == uid)
                    all_object_masks.append(object_mask[None])
            
            if len(all_object_masks) == 0:
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
            # get n masks: (n, h, w)
            all_object_masks = np.concatenate(all_object_masks, axis=0)
            
            # Check for JSON file with and without prefix
            file_path = os.path.join(json_path, f"{base_name}.json")
            if not os.path.exists(file_path):
                file_path = os.path.join(json_path, f"mask_{base_name}.json")
                
            if not os.path.exists(file_path):
                print(f"Warning: JSON file not found for {raw_image_name}, saving image without annotations")
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
            
            all_object_boxes = []
            all_object_ids = []
            all_class_names = []
            object_id_to_name = {}
            try:
                with open(file_path, "r") as file:
                    json_data = json.load(file)
                    for obj_id, obj_item in json_data["labels"].items():
                        # box id
                        instance_id = obj_item["instance_id"]
                        if instance_id not in unique_ids: # not a valid box
                            continue
                        # box coordinates
                        x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                        all_object_boxes.append([x1, y1, x2, y2])
                        # box name
                        class_name = obj_item["class_name"]
                        
                        # build id list and id2name mapping
                        all_object_ids.append(instance_id)
                        all_class_names.append(class_name)
                        object_id_to_name[instance_id] = class_name
            except Exception as e:
                print(f"Error loading JSON file {file_path}: {str(e)}, saving image without annotations")
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
            
            if not all_object_boxes:
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
                
            # Adjust object id and boxes to ascending order
            paired_id_and_box = zip(all_object_ids, all_object_boxes)
            sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])
            
            # Because we get the mask data as ascending order, so we also need to ascend box and ids
            all_object_ids = [pair[0] for pair in sorted_pair]
            all_object_boxes = [pair[1] for pair in sorted_pair]
            
            detections = sv.Detections(
                xyxy=np.array(all_object_boxes),
                mask=all_object_masks,
                class_id=np.array(all_object_ids, dtype=np.int32),
            )
            
            # custom label to show both id and class name
            labels = [
                f"{instance_id}: {class_name}" for instance_id, class_name in zip(all_object_ids, all_class_names)
            ]
            
            # Create box annotator with thicker lines
            box_annotator = sv.BoxAnnotator(
                thickness=8,  # Thicker box borders
            )
            
            # Create label annotator with much larger text
            label_annotator = sv.LabelAnnotator(
                text_thickness=6,    # Much thicker text
                text_scale=1.5,      # Much larger text
                text_padding=5,     # More padding
            )
            
            # Create mask annotator
            mask_annotator = sv.MaskAnnotator(
                opacity=0.5
            )
            
            # Apply annotations in sequence
            annotated_frame = image.copy()
            if draw_boxes:
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            if draw_labels:
                annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, annotated_frame)
            print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def draw_masks_and_box(raw_image_path, mask_path, json_path, output_path, draw_labels=True):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
                
            # Try to load mask - first without prefix, then with "mask_" prefix
            base_name = raw_image_name.split(".")[0]
            mask_npy_path = os.path.join(mask_path, f"{base_name}.npy")
            
            # If the file doesn't exist, try with the "mask_" prefix
            if not os.path.exists(mask_npy_path):
                mask_npy_path = os.path.join(mask_path, f"mask_{base_name}.npy")
                
            # If still doesn't exist, skip this image but don't fail
            if not os.path.exists(mask_npy_path):
                print(f"Warning: Mask file not found for {raw_image_name}, saving image without annotations")
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
                
            # Load the mask file
            try:
                mask = np.load(mask_npy_path)
            except Exception as e:
                print(f"Error loading mask file {mask_npy_path}: {str(e)}, saving image without annotations")
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
                
            # color map
            unique_ids = np.unique(mask)
            colors = {uid: CommonUtils.random_color() for uid in unique_ids}
            colors[0] = (0, 0, 0)  # background color

            # apply mask to image in RBG channels
            colored_mask = np.zeros_like(image)
            for uid in unique_ids:
                colored_mask[mask == uid] = colors[uid]
            alpha = 0.5  # 调整 alpha 值以改变透明度
            output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

            # Check for JSON file with and without prefix
            file_path = os.path.join(json_path, f"{base_name}.json")
            if not os.path.exists(file_path):
                file_path = os.path.join(json_path, f"mask_{base_name}.json")
                
            if not os.path.exists(file_path):
                print(f"Warning: JSON file not found for {raw_image_name}, saving image without annotations")
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue
                
            try:
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                    # Draw bounding boxes and labels
                    for obj_id, obj_item in json_data["labels"].items():
                        # Extract data from JSON
                        x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                        instance_id = obj_item["instance_id"]
                        class_name = obj_item["class_name"]

                        # Draw rectangle
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Put text
                        if draw_labels:
                            label = f"{instance_id}: {class_name}"
                            cv2.putText(output_image, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 255), 3)

                    # Save the modified image
                    output_image_path = os.path.join(output_path, raw_image_name)
                    cv2.imwrite(output_image_path, output_image)
                    print(f"Annotated image saved as {output_image_path}")
            except Exception as e:
                print(f"Error processing JSON file {file_path}: {str(e)}, saving image without annotations")
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, image)
                continue

    @staticmethod
    def random_color():
        """random color generator"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
