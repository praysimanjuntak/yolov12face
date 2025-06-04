"""
General WiderFace evaluation framework
Base class with template detection function
"""

import os
import numpy as np
import argparse
from pathlib import Path
import tqdm
from scipy.io import loadmat
from abc import ABC, abstractmethod


class BaseWiderFaceEvaluator(ABC):
    def __init__(self, confidence_threshold=0.001):
        """
        Initialize base evaluator
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.conf_threshold = confidence_threshold
    
    @abstractmethod
    def detect_faces(self, image_path):
        """
        Template method for face detection - MUST BE IMPLEMENTED
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detections, each detection should be:
            [x1, y1, width, height, confidence]
            where (x1, y1) is top-left corner
            
        Example:
            return [
                [100, 200, 50, 60, 0.95],  # face at (100,200) with size 50x60, conf 0.95
                [300, 400, 40, 45, 0.87]   # another face
            ]
        """
        pass
    
    def predict_on_widerface(self, image_dir, output_dir):
        """
        Run face detection on WiderFace validation set
        Uses the detect_faces template method
        
        Args:
            image_dir: Path to WiderFace validation images
            output_dir: Directory to save prediction files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all event directories
        event_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        
        for event_name in tqdm.tqdm(event_dirs, desc="Processing events"):
            event_path = os.path.join(image_dir, event_name)
            output_event_dir = os.path.join(output_dir, event_name)
            os.makedirs(output_event_dir, exist_ok=True)
            
            # Get all images in event directory
            image_files = [f for f in os.listdir(event_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(event_path, img_file)
                
                # Run face detection using template method
                try:
                    detections = self.detect_faces(img_path)
                    
                    # Save predictions in WiderFace format
                    stem = Path(img_file).stem
                    output_file = os.path.join(output_event_dir, f"{stem}.txt")
                    
                    with open(output_file, 'w') as f:
                        f.write(f"{event_name}/{img_file}\n")  # Image path
                        f.write(f"{len(detections)}\n")  # Number of detections
                        
                        for det in detections:
                            f.write(f"{det[0]:.1f} {det[1]:.1f} {det[2]:.1f} {det[3]:.1f} {det[4]:.3f}\n")
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    # Create empty prediction file
                    stem = Path(img_file).stem
                    output_file = os.path.join(output_event_dir, f"{stem}.txt")
                    with open(output_file, 'w') as f:
                        f.write(f"{event_name}/{img_file}\n")
                        f.write("0\n")

    def evaluate(self, pred_dir, gt_dir, iou_thresh=0.5):
        """
        Evaluate predictions using WiderFace evaluation protocol
        """
        pred = self.get_preds(pred_dir)
        self.norm_score(pred)
        
        facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = self.get_gt_boxes(gt_dir)
        
        event_num = len(event_list)
        thresh_num = 1000
        settings = ['easy', 'medium', 'hard']
        setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
        aps = []
        
        for setting_id in range(3):
            gt_list = setting_gts[setting_id]
            count_face = 0
            pr_curve = np.zeros((thresh_num, 2)).astype('float')
            
            pbar = tqdm.tqdm(range(event_num))
            for i in pbar:
                pbar.set_description('Processing {}'.format(settings[setting_id]))
                event_name = str(event_list[i][0][0])
                img_list = file_list[i][0]
                
                if event_name not in pred:
                    continue
                    
                pred_list = pred[event_name]
                sub_gt_list = gt_list[i][0]
                gt_bbx_list = facebox_list[i][0]

                for j in range(len(img_list)):
                    raw_name = str(img_list[j][0][0])
                    key = Path(raw_name).stem
                    if key not in pred_list: 
                        continue
                    pred_info = pred_list[key]
                    gt_boxes = gt_bbx_list[j][0].astype('float')
                    keep_index = sub_gt_list[j][0]
                    count_face += len(keep_index)

                    if len(gt_boxes) == 0 or len(pred_info) == 0:
                        continue
                        
                    ignore = np.zeros(gt_boxes.shape[0])
                    if len(keep_index) != 0:
                        ignore[keep_index-1] = 1
                        
                    pred_recall, proposal_list = self.image_eval(pred_info, gt_boxes, ignore, iou_thresh)
                    _img_pr_info = self.img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
                    pr_curve += _img_pr_info
                    
            pr_curve = self.dataset_pr_info(thresh_num, pr_curve, count_face)
            propose = pr_curve[:, 0]
            recall = pr_curve[:, 1]
            ap = self.voc_ap(recall, propose)
            aps.append(ap)

        print("==================== Results ====================")
        print("Easy   Val AP: {:.4f}".format(aps[0]))
        print("Medium Val AP: {:.4f}".format(aps[1]))
        print("Hard   Val AP: {:.4f}".format(aps[2]))
        print("=================================================")
        
        return aps

    # Helper functions for evaluation
    def get_gt_boxes(self, gt_dir):
        gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
        hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
        medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
        easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

        facebox_list = gt_mat['face_bbx_list']
        event_list = gt_mat['event_list']
        file_list = gt_mat['file_list']

        hard_gt_list = hard_mat['gt_list']
        medium_gt_list = medium_mat['gt_list']
        easy_gt_list = easy_mat['gt_list']

        return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

    def read_pred_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            img_file = lines[0].rstrip('\n\r')
            lines = lines[2:]

        boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
        return img_file.split('/')[-1], boxes

    def get_preds(self, pred_dir):
        events = os.listdir(pred_dir)
        boxes = dict()
        pbar = tqdm.tqdm(events)

        for event in pbar:
            pbar.set_description('Reading Predictions ')
            event_dir = os.path.join(pred_dir, event)
            if not os.path.isdir(event_dir):
                continue
                
            event_images = os.listdir(event_dir)
            current_event = dict()
            for imgtxt in event_images:
                imgname, _boxes = self.read_pred_file(os.path.join(event_dir, imgtxt))
                key = Path(imgname).stem
                current_event[key] = _boxes
            boxes[event] = current_event
        return boxes

    def norm_score(self, pred):
        max_score = 0
        min_score = 1

        for _, k in pred.items():
            for _, v in k.items():
                if len(v) == 0:
                    continue
                _min = np.min(v[:, -1])
                _max = np.max(v[:, -1])
                max_score = max(_max, max_score)
                min_score = min(_min, min_score)

        diff = max_score - min_score
        for _, k in pred.items():
            for _, v in k.items():
                if len(v) == 0:
                    continue
                v[:, -1] = (v[:, -1] - min_score)/diff

    def bbox_overlaps(self, boxes, query_boxes):
        """Calculate IoU between boxes"""
        N = boxes.shape[0]
        K = query_boxes.shape[0]
        overlaps = np.zeros((N, K), dtype=np.float32)
        
        for k in range(K):
            box_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                       (query_boxes[k, 3] - query_boxes[k, 1]))
            for n in range(N):
                iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                     max(boxes[n, 0], query_boxes[k, 0]))
                if iw > 0:
                    ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                         max(boxes[n, 1], query_boxes[k, 1]))
                    if ih > 0:
                        ua = float((boxes[n, 2] - boxes[n, 0]) *
                                  (boxes[n, 3] - boxes[n, 1]) +
                                  box_area - iw * ih)
                        overlaps[n, k] = iw * ih / ua
        return overlaps

    def image_eval(self, pred, gt, ignore, iou_thresh):
        _pred = pred.copy()
        _gt = gt.copy()
        pred_recall = np.zeros(_pred.shape[0])
        recall_list = np.zeros(_gt.shape[0])
        proposal_list = np.ones(_pred.shape[0])

        _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
        _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

        overlaps = self.bbox_overlaps(_pred[:, :4], _gt)

        for h in range(_pred.shape[0]):
            gt_overlap = overlaps[h]
            max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
            if max_overlap >= iou_thresh:
                if ignore[max_idx] == 0:
                    recall_list[max_idx] = -1
                    proposal_list[h] = -1
                elif recall_list[max_idx] == 0:
                    recall_list[max_idx] = 1

            r_keep_index = np.where(recall_list == 1)[0]
            pred_recall[h] = len(r_keep_index)
        return pred_recall, proposal_list

    def img_pr_info(self, thresh_num, pred_info, proposal_list, pred_recall):
        pr_info = np.zeros((thresh_num, 2)).astype('float')
        for t in range(thresh_num):
            thresh = 1 - (t+1)/thresh_num
            r_index = np.where(pred_info[:, 4] >= thresh)[0]
            if len(r_index) == 0:
                pr_info[t, 0] = 0
                pr_info[t, 1] = 0
            else:
                r_index = r_index[-1]
                p_index = np.where(proposal_list[:r_index+1] == 1)[0]
                pr_info[t, 0] = len(p_index)
                pr_info[t, 1] = pred_recall[r_index]
        return pr_info

    def dataset_pr_info(self, thresh_num, pr_curve, count_face):
        _pr_curve = np.zeros((thresh_num, 2))
        for i in range(thresh_num):
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
        return _pr_curve

    def voc_ap(self, rec, prec):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


def main():
    """
    Generic main function for WiderFace evaluation
    Note: This requires a concrete implementation of BaseWiderFaceEvaluator
    """
    parser = argparse.ArgumentParser(description='WiderFace evaluation framework')
    parser.add_argument('--evaluator', '-e', required=True, 
                       help='Evaluator class module (e.g., evaluation_yolo.WiderFaceYOLOEvaluator)')
    parser.add_argument('--model', '-m', help='Path to model file (if required by evaluator)')
    parser.add_argument('--images', '-i', required=True, help='Path to WiderFace validation images')
    parser.add_argument('--gt', '-g', required=True, help='Path to ground truth directory')
    parser.add_argument('--output', '-o', default='./predictions', help='Output directory for predictions')
    parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for evaluation')
    parser.add_argument('--skip-prediction', action='store_true', help='Skip prediction step (use existing predictions)')
    
    args = parser.parse_args()
    
    # Dynamically import and instantiate the evaluator
    try:
        module_name, class_name = args.evaluator.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        evaluator_class = getattr(module, class_name)
        
        # Initialize evaluator (handle different constructor signatures)
        if args.model:
            evaluator = evaluator_class(args.model, args.conf)
        else:
            evaluator = evaluator_class(args.conf)
            
    except Exception as e:
        print(f"Error loading evaluator {args.evaluator}: {e}")
        print("Example usage: --evaluator evaluation_yolo.WiderFaceYOLOEvaluator")
        return
    
    if not args.skip_prediction:
        print("Running face detection predictions on WiderFace validation set...")
        evaluator.predict_on_widerface(args.images, args.output)
    
    print("Evaluating predictions...")
    evaluator.evaluate(args.output, args.gt, args.iou)


if __name__ == '__main__':
    main()
