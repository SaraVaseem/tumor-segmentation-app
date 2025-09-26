import os
import time
import numpy as np
import json
import cv2
import torch
from pixellib.instance import custom_segmentation
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm
from torchvision.ops import nms


class SegmentationEvaluator:
    def __init__(self, model_path, class_names, detection_confidence, backbone):
        print("Initializing Segmenter...")
        try:
            self.segmenter = custom_segmentation()
            print("Segmenter created successfully")

            print("Loading model configuration...")
            self.segmenter.inferConfig(
                network_backbone = backbone,
                num_classes=len(class_names) - 1,
                class_names=class_names,
                detection_threshold=detection_confidence
            )
            print("Configuration loaded successfully")

            print(f"Loading model from: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            self.segmenter.load_model(model_path)

            print("Model loaded successfully")

            if not hasattr(self.segmenter, 'model'):
                raise AttributeError("Segmenter doesn't have 'model' attribute")

            self.model = self.segmenter.model
            print("Internal model accessed successfully")

            if not hasattr(self.model, 'config'):
                raise AttributeError("Model doesn't have 'config' attribute")

            print(
                f"Confidence: {self.model.config.DETECTION_MIN_CONFIDENCE}")

           # self.model.config.DETECTION_MIN_CONFIDENCE = detection_confidence
           # self.model.config.DETECTION_NMS_THRESHOLD = nms_threshold
           # print(f"New thresholds set - Confidence: {detection_confidence}, NMS: {nms_threshold}")

            self.class_names = class_names
            self.current_confidence = detection_confidence

            print("Metrics storage initialized")
            self.metrics = {
                'per_image': [],
                'aggregate': {
                    'object_level': {
                        'precision': 0,
                        'recall': 0,
                        'f1': 0,
                        'true_positives': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'objects_detected': 0,
                        'total_objects': 0
                    },
                    'pixel_level': {
                        'precision': 0,
                        'recall': 0,
                        'f1': 0,
                        'iou_mean': 0,
                        'iou_median': 0,
                        'true_pixels': 0,
                        'false_pixels': 0,
                        'missed_pixels': 0
                    },
                    'processing': {
                        'total_images': 0,
                        'processed_images': 0,
                        'skipped_images': 0,
                        'no_detection_images': 0,
                        'avg_inference_time': 0
                    }
                },
                'config': {
                    'detection_confidence': detection_confidence,
                }
            }
            print("Initialization completed successfully")

        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            raise

    def _create_output_dirs(self, parent_dir):
        """Create output directory with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(parent_dir, f"eval_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _load_labelme_mask(self, json_path):
        """Convert LabelMe JSON annotation to binary mask"""
        with open(json_path) as f:
            annotation = json.load(f)

        height = annotation['imageHeight']
        width = annotation['imageWidth']
        mask = np.zeros((height, width), dtype=np.uint8)

        for shape in annotation['shapes']:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], color=1)

        return mask

    def _calculate_combined_metrics(self, segmask, gt_mask):
        """Calculate both object-level and pixel-level metrics"""
        if segmask['masks'].size == 0:
            pred_mask = np.zeros_like(gt_mask)
            detected_objects = 0
        else:
            pred_mask = np.max(segmask['masks'], axis=2) if len(segmask['masks'].shape) == 3 else segmask['masks']
            detected_objects = segmask['masks'].shape[2] if len(segmask['masks'].shape) == 3 else 1

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.float32),
                                   (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        gt_binary = gt_mask.astype(np.uint8)

        # Object-level metrics
        obj_tp = 1 if (detected_objects >= 1 and np.sum(pred_binary & gt_binary) > 0) else 0
        obj_fp = max(0, detected_objects - obj_tp)
        obj_fn = 1 - obj_tp

        # Pixel-level metrics
        px_tp = np.sum(pred_binary & gt_binary)
        px_fp = np.sum(pred_binary & ~gt_binary)
        px_fn = np.sum(~pred_binary & gt_binary)

        try:
            px_precision = precision_score(gt_binary.flatten(), pred_binary.flatten(), zero_division=0)
            px_recall = recall_score(gt_binary.flatten(), pred_binary.flatten(), zero_division=0)
            px_f1 = f1_score(gt_binary.flatten(), pred_binary.flatten(), zero_division=0)
            iou = jaccard_score(gt_binary.flatten(), pred_binary.flatten(), zero_division=0)
        except:
            px_precision = px_recall = px_f1 = iou = 0.0

        obj_metrics = {
            'true_positives': obj_tp,
            'false_positives': obj_fp,
            'false_negatives': obj_fn,
            'detected_objects': detected_objects,
            'precision': obj_tp / (obj_tp + obj_fp + 1e-6),
            'recall': obj_tp / (obj_tp + obj_fn + 1e-6),
            'f1': 2 * obj_tp / (2 * obj_tp + obj_fp + obj_fn + 1e-6)
        }

        px_metrics = {
            'true_pixels': int(px_tp),
            'false_pixels': int(px_fp),
            'missed_pixels': int(px_fn),
            'precision': float(px_precision),
            'recall': float(px_recall),
            'f1': float(px_f1)
        }

        return obj_metrics, px_metrics, float(iou)

    def _save_results(self, output_dir):
        """Save metrics to JSON files"""
        with open(os.path.join(output_dir, 'per_image_metrics.json'), 'w') as f:
            json.dump(self.metrics['per_image'], f, indent=4)

        with open(os.path.join(output_dir, 'aggregate_metrics.json'), 'w') as f:
            json.dump(self.metrics['aggregate'], f, indent=4)

    def process_dataset(self, data_root, output_parent="evaluation_results", show_bboxes=True):
        print(f"\nStarting dataset processing in: {data_root}")
        try:
            # Setup paths and directories
            novel_data_path = os.path.join(data_root, "Novel Data")
            images_folder = os.path.join(novel_data_path, "images")
            annotations_folder = os.path.join(novel_data_path, "annotations")

            print(f"Images folder: {images_folder}")
            print(f"Annotations folder: {annotations_folder}")

            if not os.path.exists(images_folder):
                raise FileNotFoundError(f"Images folder not found: {images_folder}")
            if not os.path.exists(annotations_folder):
                raise FileNotFoundError(f"Annotations folder not found: {annotations_folder}")

            # Create output directories
            output_dir = self._create_output_dirs(output_parent)
            segmentation_dir = os.path.join(output_dir, "segmentations")
            metrics_dir = os.path.join(output_dir, "metrics")
            os.makedirs(segmentation_dir, exist_ok=True)
            os.makedirs(metrics_dir, exist_ok=True)
            print(f"Created output directories in: {output_dir}")

            # Get all image files
            image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images = len(image_files)
            print(f"Found {total_images} images to process")

            # Initialize counters
            obj_tp, obj_fp, obj_fn = 0, 0, 0
            pixel_tp, pixel_fp, pixel_fn = 0, 0, 0
            iou_scores = []
            inference_times = []
            processed_count = 0
            skipped_count = 0
            no_detection_count = 0

            # Process each image
            for img_file in tqdm(image_files, desc="Evaluating Images"):
                img_path = os.path.join(images_folder, img_file)
                json_path = os.path.join(annotations_folder, os.path.splitext(img_file)[0] + ".json")

                # Skip if annotation missing
                if not os.path.exists(json_path):
                    skipped_count += 1
                    continue

                try:
                    # Run segmentation
                    start_time = time.time()
                    output_path = os.path.join(segmentation_dir, f"segmented_{img_file}")
                    segmask, _ = self.segmenter.segmentImage(
                        img_path,
                        show_bboxes=show_bboxes,
                        output_image_name=output_path
                    )
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)

                    # Load ground truth
                    gt_mask = self._load_labelme_mask(json_path)

                    # Calculate metrics
                    obj_metrics, px_metrics, iou = self._calculate_combined_metrics(segmask, gt_mask)

                    # Update counters
                    obj_tp += obj_metrics['true_positives']
                    obj_fp += obj_metrics['false_positives']
                    obj_fn += obj_metrics['false_negatives']
                    pixel_tp += px_metrics['true_pixels']
                    pixel_fp += px_metrics['false_pixels']
                    pixel_fn += px_metrics['missed_pixels']
                    iou_scores.append(iou)

                    if obj_metrics['detected_objects'] == 0:
                        no_detection_count += 1

                    # Store per-image results
                    self.metrics['per_image'].append({
                        'image': img_file,
                        'object_level': obj_metrics,
                        'pixel_level': px_metrics,
                        'iou': iou,
                        'inference_time': inference_time,
                        'had_detection': obj_metrics['detected_objects'] > 0
                    })

                    processed_count += 1
                except Exception as e:
                    print(f"\nError processing {img_file}: {str(e)}")
                    skipped_count += 1

            # Calculate aggregate metrics
            valid_ious = [iou for iou in iou_scores if not np.isnan(iou)]
            self._compute_aggregate_metrics(
                obj_tp, obj_fp, obj_fn,
                pixel_tp, pixel_fp, pixel_fn,
                valid_ious, processed_count,
                skipped_count, no_detection_count,
                inference_times
            )

            # Save results
            self._save_results(metrics_dir)
            print(f"\nSaved metrics to: {metrics_dir}")
            print(f"Saved segmentations to: {segmentation_dir}")

            # Print report
            self._print_detailed_report()
            print("Dataset processing completed successfully")

        except Exception as e:
            print(f"Error in process_dataset: {str(e)}")
            raise

    def _compute_aggregate_metrics(self, obj_tp, obj_fp, obj_fn,
                                   pixel_tp, pixel_fp, pixel_fn,
                                   iou_scores, processed_count,
                                   skipped_count, no_detection_count,
                                   inference_times):
        """Compute final aggregate metrics"""
        self.metrics['aggregate']['object_level'].update({
            'true_positives': obj_tp,
            'false_positives': obj_fp,
            'false_negatives': obj_fn,
            'objects_detected': obj_tp,
            'total_objects': processed_count,
            'precision': obj_tp / (obj_tp + obj_fp + 1e-6),
            'recall': obj_tp / (obj_tp + obj_fn + 1e-6),
            'f1': 2 * obj_tp / (2 * obj_tp + obj_fp + obj_fn + 1e-6)
        })

        self.metrics['aggregate']['pixel_level'].update({
            'true_pixels': int(pixel_tp),
            'false_pixels': int(pixel_fp),
            'missed_pixels': int(pixel_fn),
            'precision': pixel_tp / (pixel_tp + pixel_fp + 1e-6),
            'recall': pixel_tp / (pixel_tp + pixel_fn + 1e-6),
            'f1': 2 * pixel_tp / (2 * pixel_tp + pixel_fp + pixel_fn + 1e-6),
            'iou_mean': np.mean(iou_scores) if iou_scores else 0,
            'iou_median': np.median(iou_scores) if iou_scores else 0
        })

        self.metrics['aggregate']['processing'].update({
            'total_images': processed_count + skipped_count,
            'processed_images': processed_count,
            'skipped_images': skipped_count,
            'no_detection_images': no_detection_count,
            'avg_inference_time': np.mean(inference_times) if inference_times else 0
        })

    def _print_detailed_report(self):
        """Print a comprehensive report to console"""
        agg = self.metrics['aggregate']
        obj = agg['object_level']
        px = agg['pixel_level']
        proc = agg['processing']

        report = [
            "\nCOMPREHENSIVE EVALUATION REPORT",
            "=" * 50,
            f"Images Processed: {proc['processed_images']}/{proc['total_images']}",
            f"- Skipped (missing annotations): {proc['skipped_images']}",
            f"- With no detections: {proc['no_detection_images']}",
            f"Avg Inference Time: {proc['avg_inference_time']:.2f}s/image",
            "",
            "OBJECT-LEVEL METRICS (Detection Performance)",
            "-" * 50,
            f"Precision: {obj['precision']:.4f}  (TP/TP+FP)",
            f"Recall:    {obj['recall']:.4f}  (TP/TP+FN)",
            f"F1 Score:  {obj['f1']:.4f}",
            f"Objects Detected: {obj['objects_detected']}/{obj['total_objects']}",
            "",
            "PIXEL-LEVEL METRICS (Segmentation Quality)",
            "-" * 50,
            f"Precision: {px['precision']:.4f}",
            f"Recall:    {px['recall']:.4f}",
            f"F1 Score:  {px['f1']:.4f}",
            f"IoU (Jaccard): Mean={px['iou_mean']:.4f}, Median={px['iou_median']:.4f}",
            f"Pixel Statistics:",
            f"- True Positives:  {px['true_pixels']}",
            f"- False Positives: {px['false_pixels']}",
            f"- False Negatives: {px['missed_pixels']}"
        ]

        print("\n".join(report))


if __name__ == "__main__":
    print("Starting program execution")
    try:
        print("Creating evaluator instance...")

        evaluator = SegmentationEvaluator(
            model_path="(0.47)mask_rcnn_model.002-1.068161.h5",
            class_names=["BG", "tumor"],
            detection_confidence=0.90,
            backbone = "resnet101"
        )

        print("\nBeginning dataset evaluation...")
        evaluator.process_dataset(".")
        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback

        traceback.print_exc()