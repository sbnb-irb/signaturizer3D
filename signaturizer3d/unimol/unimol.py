# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import argparse
import logging
import pathlib

import numpy as np
import torch
import torch.nn as nn

from signaturizer3d.unimol.pad import pad_1d_tokens, pad_2d, pad_coords
from signaturizer3d.unicore.dictionary import Dictionary
from signaturizer3d.unicore.transformer_encoder import init_bert_params
from signaturizer3d.unicore.unicore_model import BaseUnicoreModel
from signaturizer3d.unicore.utils import get_activation_fn

from .transformers import TransformerEncoderWithPair

logger = logging.getLogger(__name__)

BACKBONE = {
    "transformer": TransformerEncoderWithPair,
}

WEIGHT_DIR = pathlib.Path(__file__).resolve().parents[2] / "weights"


class UniMolModel(BaseUnicoreModel):
    def __init__(
        self,
        classification_head_name: str,
        remove_hs: bool,
        output_dim: int,
        model_file_name: str = None,
        model_file_URL: str = None,
    ):
        super().__init__()
        self.args = finetuned_architecture()

        self.model_file_name = model_file_name
        self.classification_head_name = classification_head_name
        self.remove_hs = remove_hs
        self.output_dim = output_dim

        current_dir = pathlib.Path(__file__).resolve().parents[0]
        self.dictionary = Dictionary.load((current_dir / "dict.txt").as_posix())

        self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = self.dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(self.dictionary), self.args.encoder_embed_dim, self.padding_idx
        )
        self.encoder = BACKBONE[self.args.backbone](
            encoder_layers=self.args.encoder_layers,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.encoder_ffn_embed_dim,
            attention_heads=self.args.encoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.delta_pair_repr_norm_loss < 0,
        )
        K = 128
        n_edge_type = len(self.dictionary) * len(self.dictionary)
        self.gbf_proj = NonLinearHead(
            K, self.args.encoder_attention_heads, self.args.activation_fn
        )
        if self.args.kernel == "gaussian":
            self.gbf = GaussianLayer(K, n_edge_type)
        else:
            raise ValueError("Current not support kernel: {}".format(self.args.kernel))

        self.classification_heads = nn.ModuleDict()
        self.classification_heads[self.classification_head_name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.encoder_embed_dim,
            num_classes=self.output_dim,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

        self.apply(init_bert_params)
        if model_file_name:
            self.pretrain_path = (WEIGHT_DIR / self.model_file_name).as_posix()
            self.load_pretrained_weights(path=self.pretrain_path)
        elif model_file_URL:
            self.download_and_load_pretrained_weights(model_file_URL)
        else:
            raise ValueError("Please provide either model_file_name or model_file_URL")

    def load_pretrained_weights(self, path):
        logger.info(f"Loading pretrained weights from {path}")
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict["model"], strict=False)

    def download_and_load_pretrained_weights(self, url):
        logger.info(f"Loading pretrained weights from {url}")
        state_dict = torch.hub.load_state_dict_from_url(url)
        self.load_state_dict(state_dict["model"], strict=False)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        **kwargs,
    ):
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            _,
            _,
            _,
            _,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        cls_repr = encoder_rep[:, 0, :]  # CLS token repr
        all_repr = encoder_rep[:, :, :]  # all token repr

        filtered_tensors = []
        for tokens in src_tokens:
            filtered_tensor = tokens[
                (tokens != 0) & (tokens != 1) & (tokens != 2)
            ]  # filter out BOS(0), EOS(1), PAD(2)
            filtered_tensors.append(filtered_tensor)

        lengths = [
            len(filtered_tensor) for filtered_tensor in filtered_tensors
        ]  # Compute the lengths of the filtered tensors

        cls_atomic_reprs = []
        for i in range(len(all_repr)):
            atomic_repr = encoder_rep[i, 1 : lengths[i] + 1, :]
            cls_atomic_reprs.append(atomic_repr)

        logits = self.classification_heads[self.classification_head_name](cls_repr)

        return logits

    def batch_collate_fn(self, samples):
        batch = {}
        for k in samples[0][0].keys():
            if k == "src_coord":
                v = pad_coords(
                    [torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0
                )
            elif k == "src_edge_type":
                v = pad_2d(
                    [torch.tensor(s[0][k]).long() for s in samples],
                    pad_idx=self.padding_idx,
                )
            elif k == "src_distance":
                v = pad_2d(
                    [torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0
                )
            elif k == "src_tokens":
                v = pad_1d_tokens(
                    [torch.tensor(s[0][k]).long() for s in samples],
                    pad_idx=self.padding_idx,
                )
            batch[k] = v
        try:
            label_np = np.array([s[1] for s in samples])
            label = torch.tensor(label_np)
        except:
            label = None
        return batch, label


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


def finetuned_architecture():
    args = argparse.ArgumentParser()
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", "gaussian")
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    return args
