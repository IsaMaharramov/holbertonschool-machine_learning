#!/usr/bin/env python3
"""Yolo V3 Object Detection module"""
import cv2
import numpy as np
import os
from tensorflow import keras as K

glob = __import__('glob')


class Yolo:
    """Class Yolo that uses Yolo v3 algorithm to perform object detection"""

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

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            box = np.zeros(output[..., :4].shape)

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

            cx = np.tile(np.arange(0, grid_width), (grid_height, 1))
            cx = np.tile(cx, (anchor_boxes, 1, 1)).transpose(1, 2, 0)

            cy = np.tile(np.arange(0, grid_height), (grid_width, 1)).T
            cy = np.tile(cy, (anchor_boxes, 1, 1)).transpose(1, 2, 0)

            bx = (1 / (1 + np.exp(-t_x)) + cx) / grid_width
            by = (1 / (1 + np.exp(-t_y)) + cy) / grid_height

            bw = (pw * np.exp(t_w)) / input_width
            bh = (ph * np.exp(t_h)) / input_height

            box[..., 0] = (bx - (bw / 2)) * image_width
            box[..., 1] = (by - (bh / 2)) * image_height
            box[..., 2] = (bx + (bw / 2)) * image_width
            box[..., 3] = (by + (bh / 2)) * image_height

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
        Applies Non-max Suppression to filtered bounding boxes.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = (box_classes == cls)

            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                iou = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        if len(box_predictions) > 0:
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0
            )
            predicted_box_scores = np.concatenate(
                predicted_box_scores, axis=0
            )

        return box_predictions, predicted_box_classes, predicted_box_scores

    def load_images(self, folder_path):
        """
        Loads images from a given folder path.
        """
        image_paths = glob.glob(folder_path + '/*', recursive=False)
        images = [cv2.imread(p) for p in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses images by resizing with inter-cubic interpolation
        and rescaling pixel values to [0, 1].
        """
        pimages = []
        image_shapes = []

        input_w = int(self.model.input.shape[1])
        input_h = int(self.model.input.shape[2])

        for img in images:
            image_shapes.append([img.shape[0], img.shape[1]])
            resized = cv2.resize(
                img,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            pimages.append(resized / 255.0)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
