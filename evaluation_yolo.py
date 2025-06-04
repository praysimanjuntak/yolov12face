"""
WiderFace evaluation for Ultralytics YOLO models
Adapted from original WiderFace evaluation code
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
from evaluation_base import BaseWiderFaceEvaluator


class WiderFaceYOLOEvaluator(BaseWiderFaceEvaluator):
    def __init__(self, model_path, confidence_threshold=0.001):
        """
        Initialize YOLO model for WiderFace evaluation
        
        Args:
            model_path: Path to YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections
        """
        super().__init__(confidence_threshold)
        self.model = YOLO(model_path)
    
    def detect_faces(self, image_path):
        """
        YOLO implementation of face detection
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detections in format [x1, y1, width, height, confidence]
        """
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                # Convert to x1, y1, w, h, confidence format
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                
                detections.append([x1, y1, w, h, conf])
        
        return detections


def main():
    parser = argparse.ArgumentParser(description='WiderFace evaluation with YOLO')
    parser.add_argument('--model', '-m', required=True, help='Path to YOLO model (.pt file)')
    parser.add_argument('--images', '-i', required=True, help='Path to WiderFace validation images')
    parser.add_argument('--gt', '-g', required=True, help='Path to ground truth directory')
    parser.add_argument('--output', '-o', default='./predictions', help='Output directory for predictions')
    parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for evaluation')
    parser.add_argument('--skip-prediction', action='store_true', help='Skip prediction step (use existing predictions)')
    
    args = parser.parse_args()
    
    evaluator = WiderFaceYOLOEvaluator(args.model, args.conf)
    
    if not args.skip_prediction:
        print("Running YOLO predictions on WiderFace validation set...")
        evaluator.predict_on_widerface(args.images, args.output)
    
    print("Evaluating predictions...")
    evaluator.evaluate(args.output, args.gt, args.iou)


if __name__ == '__main__':
    main()
