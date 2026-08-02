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

            # Extract box transformation predictions
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # 1. Apply Sigmoid to confidences and class probabilities
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            # 2. Build grid cell offsets (cx, cy)
            # cx varies across columns (width), cy varies across rows (height)
            cx = np.tile(np.arange(0, grid_w), (grid_h, 1)).reshape(grid_h, grid_w, 1)
            cy = np.tile(np.arange(0, grid_h), (grid_w, 1)).T.reshape(grid_h, grid_w, 1)

            # 3. Calculate bounding box center coordinates (b_x, b_y) normalized by grid dimensions
            b_x = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + cy) / grid_h

            # 4. Reshape anchor dimensions for 3D broadcasting: (1, 1, anchor_boxes)
            p_w = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes)
            p_h = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes)

            # 5. Calculate bounding box width and height normalized by input image dimensions
            b_w = (np.exp(t_w) * p_w) / input_w
            b_h = (np.exp(t_h) * p_h) / input_h

            # 6. Convert (center_x, center_y, width, height) to corners (x1, y1, x2, y2)
            # scaled to original image dimensions
            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            # Combine coordinates into (grid_h, grid_w, anchor_boxes, 4)
            box = np.zeros((grid_h, grid_w, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
