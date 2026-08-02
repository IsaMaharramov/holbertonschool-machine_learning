#!/usr/bin/env python3
"""Yolo V3 Object Detection module"""
from tensorflow import keras as K
import numpy as np


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

        image_h, image_w = image_size
        input_h = int(self.model.input.shape[1])
        input_w = int(self.model.input.shape[2])

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Sigmoid for confidences and class probabilities
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            # Reshape grid offsets for automatic numpy broadcasting
            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)

            # Center coordinates relative to grid cells
            b_x = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + cy) / grid_h

            # Anchor dimensions for output scale i
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Box width and height normalized to model input size
            b_w = (np.exp(t_w) * pw) / input_w
            b_h = (np.exp(t_h) * ph) / input_h

            # Convert to corner coordinates relative to original image size
            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            box = np.zeros((grid_h, grid_w, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
