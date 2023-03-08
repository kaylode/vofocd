# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------




import math
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn
from source.detection.models.detr_utils import box_ops
from source.detection.models.detr_utils.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)

from .dn_components import prepare_for_dn, dn_post_process
from source.detection.models.dyhead.dyhead import DyHead
from source.detection.models.dyhead2.dyhead import FPNDyHead

class DABDETR(nn.Module):
    """ This is the DAB-DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, 
                    iter_update=True,
                    query_dim=4, 
                    bbox_embed_diff_each_layer=False,
                    random_refpoints_xy=False,
                    num_image_classes = None
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for False. (shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
            

        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        # leave one dim for indicator
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)
        self.num_classes = num_classes

        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for i in range(6)])
        else:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        

        # setting query dim
        self.query_dim = query_dim
        assert query_dim in [2, 4]

        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.iter_update = iter_update

        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed
            self.transformer.decoder.class_embed = self.class_embed


        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # import ipdb; ipdb.set_trace()
        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # self.dyhead = DyHead(
        #     in_channels_list=self.backbone.num_channels,
        #     out_channels=self.hidden_dim,
        #     num_convs=3
        # )

        self.dyhead = FPNDyHead(
            S=60*60, # median shape of resnet+fpn layers (temporaly hardcoded)
            num_blocks=3,
            fpn_returned_layers=[1,2,3]
        )

        self.num_image_classes = num_image_classes
        if self.num_image_classes is not None:
            self.pooling = torch.nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )

            self.global_classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(
                    in_features=self.hidden_dim, 
                    out_features=self.num_image_classes
                )
            )

    def forward(self, samples: NestedTensor, dn_args=None, bin_masks=None):
        """
            Add two functions prepare_for_dn and dn_post_process to implement dn
            The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # if isinstance(samples, (list, torch.Tensor)):
            # samples = nested_tensor_from_tensor_list(samples)
        samples = NestedTensor(samples, bin_masks)

        # features = self.backbone(samples)
        # src, mask = features[-1].decompose()

        # bs = src.shape[0]
        # bs = src.shape[0]
        # dy_feats = self.dyhead(features)
        bs = samples.tensors.shape[0]
        dy_feats = self.dyhead(samples)
       
        if self.num_image_classes is not None:
            last_fmap = dy_feats[-1].tensors
            pooled_outputs = self.pooling(last_fmap)
            global_outputs = self.global_classifier(pooled_outputs)

        # default pipeline
        embedweight = self.refpoint_embed.weight

        # prepare for dn
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_dn(dn_args, embedweight, bs, self.training, self.num_queries, self.num_classes,
                           self.hidden_dim, self.label_enc)

        outputs_class, outputs_coord = self.transformer(dy_feats, input_query_bbox, tgt=input_query_label,
                                         attn_mask=attn_mask)
                                         
        # dn post process
        outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.num_image_classes is not None:
            return out, mask_dict, global_outputs
        else:
            return out, mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        if isinstance(outputs, dict) and 'pred_logits' in outputs.keys():

            num_select = self.num_select
            out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

            assert len(out_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2

            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.reshape(out_logits.shape[0], -1), num_select, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
            
            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            
            boxes = boxes * scale_fct[:, None, :]

            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        else:
            labels = [i['labels'] for i in outputs]
            boxes = [i['boxes'] for i in outputs]
            boxes = [box_ops.box_cxcywh_to_xyxy(box) for box in boxes]
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            new_boxes = [i*scale for i, scale in zip(boxes, scale_fct)]
            results = [{'labels': l, 'boxes': b} for l, b in zip(labels, new_boxes)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


