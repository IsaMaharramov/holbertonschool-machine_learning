#!/usr/bin/env python3
"""Yolo V3 Object Detection module"""
import numpy as np
from tensorflow import keras as K


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
