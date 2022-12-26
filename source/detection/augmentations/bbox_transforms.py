import torch

class BoxOrder(DualTransform):
    """
    Bounding boxes reorder
    """
    
    def __init__(
        self,
        order
    ):
        """
        Class construstor
        :param order: bbox format
        """
        super(BoxOrder, self).__init__(always_apply=True, p=1.0)  # Initialize parent class
        self.order = order
        
    def apply(self, image, **params):
        """
        Applies the reorder augmentation on the given image
        
        :param image: The image to be augmented
        :returns augmented image
        """
        return image

    def apply_to_bbox(self, bbox, **params):

        """
        Change box order between (xmin, ymin, xmax, ymax) and (xcenter, ycenter, width, height).
        :param boxes: (tensor) or {np.array) bounding boxes, sized [N, 4]
        :param order: (str) ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcy', 'cxcy2xyxy']
        :return: (tensor) converted bounding boxes, size [N, 4]
        """

        assert self.order   in ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcy', 'cxcy2xyxy', 'yxyx2xyxy', 'xyxy2yxyx']

        # Convert 1-d to a 2-d tensor of boxes, which first dim is 1
        if isinstance(boxes, torch.Tensor):
            if len(boxes.shape) == 1:
                boxes = boxes.unsqueeze(0)

            if self.order   == 'xyxy2xywh':
                return torch.cat([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
            elif self.order   ==  'xywh2xyxy':
                return torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], 1)
            elif self.order   == 'xyxy2cxcy':
                return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,  # c_x, c_y
                                boxes[:, 2:] - boxes[:, :2]], 1)  # w, h
            elif self.order   == 'cxcy2xyxy':
                return torch.cat([boxes[:, :2] - (boxes[:, 2:] *1.0 / 2),  # x_min, y_min
                                boxes[:, :2] + (boxes[:, 2:] *1.0 / 2)], 1)  # x_max, y_max
            elif self.order   == 'xyxy2yxyx' or order == 'yxyx2xyxy':
                return boxes[:,[1,0,3,2]]
            
        else:
            # Numpy
            new_boxes = boxes.copy()
            if self.order == 'xywh2xyxy':
                new_boxes[:,2] = boxes[:,0] + boxes[:,2]
                new_boxes[:,3] = boxes[:,1] + boxes[:,3]
                return new_boxes
            elif self.order == 'xyxy2xywh':
                new_boxes[:,2] = boxes[:,2] - boxes[:,0]
                new_boxes[:,3] = boxes[:,3] - boxes[:,1]
                return new_boxes

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('order',  'always_apply', 'p')