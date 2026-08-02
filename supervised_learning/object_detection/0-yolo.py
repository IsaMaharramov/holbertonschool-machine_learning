#!/usr/bin/env python3
"""
Module for Yolo V3 Object Detection initialization
"""
from tensorflow import keras as K


class Yolo:
    """
    Yolo class that uses the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.

        Args:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to where the list of class names used for
                          the Darknet model, listed in order of index,
                          can be found
            class_t: float representing the box score threshold for the
                     initial filtering step
            nms_t: float representing the IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                     containing all of the anchor boxes
        """
        # Load the Darknet Keras model
        self.model = K.models.load_model(model_path)

        # Load the class names into a list
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        # Set thresholds and anchors
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
