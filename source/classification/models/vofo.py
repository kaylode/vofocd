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
from theseus.cv.classification.models.timm_models import BaseTimmModel
from theseus.base.utilities.loading import load_state_dict

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
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        #hs shape: [hidden_state, batch, num_queries, hidden_dim]
        return hs, (src, mask)


class VOFO_DETR(nn.Module):
    def __init__(
        self, 
        num_classes,
        detr_name, 
        embed_dim: int = 256,
        clf_name: str = None, 
        clf_checkpoint: str = None,
        clf_freeze: bool = False,
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
                load_state_dict(self.clf_extractor, state_dict=clf_state_dict)
            self.global_dim = self.clf_extractor.feature_dim
        else:
            self.clf_extractor = None
            self.global_dim = self.detr_extractor.backbone_hidden_dim

        self.local_dim = self.detr_extractor.hidden_dim
        self.num_queries = self.detr_extractor.num_queries


        if detr_checkpoint is not None:
            detr_state_dict = torch.load(detr_checkpoint)['model']
            load_state_dict(self.detr_extractor, state_dict=detr_state_dict)

        if self.pooling_type == 'mean':
            self.ffn_local = nn.Linear(self.local_dim*self.num_queries, self.local_dim)
            self.ffn_global = nn.Linear(self.global_dim, self.local_dim)
            self.ffn = nn.Linear(self.local_dim*2, num_classes)

        elif self.pooling_type == 'attn':
            self.multihead_attn = nn.MultiheadAttention(
                self.local_dim, 
                self.num_heads,
                kdim=self.global_dim, vdim=self.global_dim
            )
            self.ffn = nn.Linear(self.embed_dim, num_classes)

    def get_model(self):
        return self

    def forward_batch(self, batch: Dict, device: torch.device):
        x = move_to(batch['inputs'], device)

        if self.clf_extractor is not None:
            global_feats = self.clf_extractor.forward_batch(batch, device)['outputs']
            local_feats, (_, masks) = self.detr_extractor(x)
        else:
            local_feats, (global_feats, masks) = self.detr_extractor(x)
            global_feats = global_feats.flatten(2).permute(2, 0, 1) # [HxW, B, Dg]      64, 32, 128

        logits = self.pooling(local_feats, global_feats, masks)
        return {
            'outputs': logits,
        }

    def pooling(self, local_feats, global_feats, global_masks):
        # local_feats: [hidden_state, batch, num_queries, hidden_dim1]
        # global_feats: [batch, hidden_dim2]
        if self.pooling_type == 'attn':
            local_feats = torch.mean(local_feats, dim=0).permute(1, 0, 2) # [Q, N, Dl]  10, 32, 256
            
            attn_output, attn_output_weights = self.multihead_attn(
                query=local_feats, 
                key=global_feats, 
                value=global_feats,
            )

            # [num_queries, batch, hidden_dim1], [batch,num_queries, 64]
            logits = self.ffn(
                torch.mean(attn_output, dim=0)
            )

        elif self.pooling_type == 'mean':
            batch_size = local_feats.shape[1]
            local_feats = torch.mean(local_feats, dim=0).reshape(batch_size, -1)

            local_feats = self.ffn_local(local_feats)

            if len(global_feats.shape) == 3:
                global_feats = torch.mean(global_feats, dim=0)

            global_feats = self.ffn_global(global_feats)
            logits = self.ffn(torch.cat([global_feats, local_feats], dim=1))
        
        else:
            raise ValueError()

        return logits

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.

        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device
        """
        outputs = self.forward_batch(adict, device)["outputs"]

        if not adict.get("multilabel"):
            outputs, probs = logits2labels(
                outputs, label_type="multiclass", return_probs=True
            )
        else:
            outputs, probs = logits2labels(
                outputs,
                label_type="multilabel",
                threshold=adict["threshold"],
                return_probs=True,
            )

            if adict.get("no-zeroes"):
                argmaxs = torch.argmax(probs, dim=1)
                tmp = torch.sum(outputs, dim=1)
                one_hots = F.one_hot(argmaxs, outputs.shape[1])
                outputs[tmp == 0] = one_hots[tmp == 0].bool()

        probs = move_to(detach(probs), torch.device("cpu")).numpy()
        classids = move_to(detach(outputs), torch.device("cpu")).numpy()

        if self.classnames and not adict.get("multilabel"):
            classnames = [self.classnames[int(clsid)] for clsid in classids]
        elif self.classnames and adict.get("multilabel"):
            classnames = [
                [self.classnames[int(i)] for i, c in enumerate(clsid) if c]
                for clsid in classids
            ]
        else:
            classnames = []

        return {
            "labels": classids,
            "confidences": probs,
            "names": classnames,
        }