#!/usr/bin/env python3
"""Yolo V3 Object Detection module"""
from tensorflow import keras as K
import numpy as np


class Yolo:
    """Class Yolo that uses Yolo v3 algorithm to perform detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet outputs into boundary boxes, confidences,
        and class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size
        input_h = int(self.model.input.shape[1])
        input_w = int(self.model.input.shape[2])

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            cx = np.tile(np.arange(0, grid_w),
                         (grid_h, 1)).reshape(grid_h, grid_w, 1)
            cy = np.tile(np.arange(0, grid_h),
                         (grid_w, 1)).T.reshape(grid_h, grid_w, 1)

            cx = np.repeat(cx, anchor_boxes, axis=-1)
            cy = np.repeat(cy, anchor_boxes, axis=-1)

            b_x = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + cy) / grid_h

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            b_w = (np.exp(t_w) * p_w) / input_w
            b_h = (np.exp(t_h) * p_h) / input_h

            x1 = (b_x - b_w / 2) * image_w
            y1 = (b_y - b_h / 2) * image_h
            x2 = (b_x + b_w / 2) * image_w
            y2 = (b_y + b_h / 2) * image_h

            box = np.zeros((grid_h, grid_w, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boundary boxes based on their box score.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to filter out overlapping bounding boxes.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            idxs = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            order = cls_scores.argsort()[::-1]
            keep = []

            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
                yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
                xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
                yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                area_i = ((cls_boxes[i, 2] - cls_boxes[i, 0]) *
                          (cls_boxes[i, 3] - cls_boxes[i, 1]))
                area_order = ((cls_boxes[order[1:], 2] -
                               cls_boxes[order[1:], 0]) *
                              (cls_boxes[order[1:], 3] -
                               cls_boxes[order[1:], 1]))

                union = area_i + area_order - inter
                iou = inter / union

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        if len(box_predictions) == 0:
            return (np.array([]), np.array([]), np.array([]))

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
