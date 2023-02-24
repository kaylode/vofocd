import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from theseus.base.utilities.cuda import move_to, detach
from source.detection.models.detr_utils import box_ops
from source.detection.models.detr_utils.misc import (
    NestedTensor, nested_tensor_from_tensor_list
)
from theseus.base.utilities.logits import logits2labels
from source.detection.models.detr_components.transformer import Transformer
from source.detection.models.detr_components.backbone import build_backbone
from source.detection.models.detr_components.detr import MLP
from theseus.cv.classification.models.timm_models import BaseTimmModel
from theseus.base.utilities.loading import load_state_dict
from .detr_components.detr import DETR, PostProcess

class DETRExtractor(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone_name, **kwargs):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()

        backbone = build_backbone(
            backbone_name, 
            hidden_dim=kwargs.get('detr_hidden_dim', 256), 
            position_embedding=kwargs.get('detr_position_embedding', 'sine'), 
            freeze_backbone=kwargs.get('detr_freeze_backbone', False), 
            dilation=kwargs.get('detr_dilation', True),
            return_interm_layers=False
        )

        transformer = Transformer(
            d_model=kwargs.get('detr_hidden_dim', 256),
            dropout=kwargs.get('detr_dropout', 0.1),
            nhead=kwargs.get('detr_nheads', 8),
            dim_feedforward=kwargs.get('detr_dim_feedforward', 2048),
            num_encoder_layers=kwargs.get('detr_enc_layers', 6),
            num_decoder_layers=kwargs.get('detr_dec_layers', 6),
            normalize_before=kwargs.get('detr_pre_norm', True),
            return_intermediate_dec=True,
        )

        self.backbone_hidden_dim = backbone.num_channels
        self.num_queries = kwargs.get('detr_num_queries', 10)
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.backbone = backbone

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)


    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        assert mask is not None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        
        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        import pdb
        pdb.set_trace()

        #hs shape: [hidden_state, batch, num_queries, hidden_dim]
        return hs, (src, mask)


class VOFO_DETR(nn.Module):
    def __init__(
        self, 
        num_classes: int,
        num_img_classes,
        detr_name, 
        embed_dim: int = 256,
        clf_name: str = None, 
        clf_checkpoint: str = None,
        clf_freeze: bool = False,
        detr_freeze: bool = False,
        detr_checkpoint: str = None,
        pooling_type: str = 'attn',
        num_heads: int = 4,
        classnames: Optional[List] = None,
        **kwargs) -> None:
        super().__init__()

        self.pooling_type = pooling_type
        self.num_heads = num_heads
        self.classnames = classnames
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.detr_extractor = DETRExtractor(
            backbone_name=detr_name,
            **kwargs
        )

        if clf_name is not None:
            self.clf_extractor = BaseTimmModel(
                clf_name, num_classes=0, from_pretrained=False, freeze = clf_freeze
            )
            if clf_checkpoint is not None:
                clf_state_dict = torch.load(clf_checkpoint)['model']
                load_state_dict(self.clf_extractor, state_dict=clf_state_dict, strict=False)
            self.global_dim = self.clf_extractor.feature_dim
        else:
            self.clf_extractor = None
            self.global_dim = self.detr_extractor.backbone_hidden_dim

        self.local_dim = self.detr_extractor.hidden_dim
        self.num_queries = self.detr_extractor.num_queries

        if detr_checkpoint is not None:
            detr_state_dict = torch.load(detr_checkpoint)['model']
            self.load_pretrained_detr(detr_state_dict)

        if detr_freeze:
            self.detr_extractor.freeze()

        self.multihead_attn = nn.MultiheadAttention(
            self.local_dim, 
            self.num_heads,
            kdim=225, vdim=225 # based on input shape
        )
        self.ffn = nn.Linear(self.embed_dim, num_img_classes)

        # Localization error
        self.class_embed = nn.Linear(self.local_dim, num_classes + 1)
        self.bbox_embed = MLP(self.local_dim, self.local_dim, 4, 3)

        self.postprocessor = PostProcess(min_conf=kwargs.get('detr_min_conf'))

    def load_pretrained_detr(self, state_dict):
        def overwrite_key(state_dict):
            for key in list(state_dict.keys()):
                state_dict[key.replace('model.', '')] = state_dict.pop(key)
            return state_dict
        load_state_dict(self.detr_extractor, state_dict=overwrite_key(state_dict), strict=False)

    def get_model(self):
        return self

    def forward_batch(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)

        if self.clf_extractor is not None:
            global_feats = self.clf_extractor.forward_batch(batch, device)['outputs']
            local_feats, (_, masks) = self.detr_extractor(x)
        else:
            local_feats, (global_feats, masks) = self.detr_extractor(x)

        logits, outputs_class, outputs_coord, attn_map = self.pooling(local_feats, global_feats, masks)
        return {
            'img_outputs': logits, 
            'attn_map': attn_map, 
            'obj_outputs': {
                'pred_logits': outputs_class,  
                'pred_boxes': outputs_coord
            }
        }


    def pooling(self, local_feats, global_feats, global_masks=None):
        # local_feats: [H, B, Q, D_local]
        # global_feats: [B, C, H, W]
        local_feats = torch.mean(local_feats, dim=0).permute(1, 0, 2) # [Q, B, D_local]  [10, 32, 256]
        global_feats = global_feats.flatten(2).permute(1, 0, 2) # [C, B, HxW]      1280, 32, 64
        attn_output, attn_output_weights = self.multihead_attn(
            query=local_feats, 
            key=global_feats, 
            value=global_feats,
        )
        # [Q, B, D_local], [B, Q, HxW]

        outputs_class = self.class_embed(attn_output) # [Q, B, num_obj+1]
        outputs_coord = self.bbox_embed(attn_output).sigmoid() # [Q, B, 4]

        outputs_class = outputs_class.permute(1,0,2)
        outputs_coord = outputs_coord.permute(1,0,2)

        logits = self.ffn(
            torch.mean(attn_output, dim=0) # mean over bboxes
        )

        return logits, outputs_class, outputs_coord, attn_output_weights

    def postprocess(self, outputs: Dict, batch: Dict):
        batch_size = outputs['obj_outputs']['pred_logits'].shape[0]
        target_sizes = torch.Tensor([batch['inputs'].shape[-2:]]).repeat(batch_size, 1)

        results = self.postprocessor(
            outputs = outputs['obj_outputs'],
            target_sizes=target_sizes
        )

        denormalized_targets = batch['obj_targets']
        denormalized_targets = self.postprocessor(
            outputs = denormalized_targets,
            target_sizes=target_sizes
        )

        batch['obj_targets'] = denormalized_targets
        outputs['obj_outputs'] = results
        return outputs, batch
    

class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[torch.Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights