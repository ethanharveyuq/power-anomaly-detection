"""
From George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""
# --- Standard library ---
from typing import Optional

# --- Third‑party ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from einops import rearrange

# --- Transformers ---
from transformers import (
    GPT2ForSequenceClassification,
    BertTokenizer,
    BertModel,
    BertConfig,
    LlamaModel,
    LlamaConfig,
    PhiModel,
    GemmaModel,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

# --- Local modules ---
from .embed import DataEmbedding, DataEmbedding_wo_time



class gpt4ts(nn.Module):
    
    def __init__(self, config, data):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['patch stride']
        self.gpt_layers = 6
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], dropout=config['dropout'])


        # Added different llm support
        self._load_backbone(config["model"])
        self._freeze_backbone(config["model"])

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.backbone.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        self.head_dropout = nn.Dropout(config['dropout'])
        self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes)
    
    # Changed func definition to have default none value
    def forward(self, x_enc, x_mark_enc=None):
        B, L, M = x_enc.shape
        
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        
        outputs = self.enc_embedding(input_x, None)
        outputs = self.backbone(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.head_dropout(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs

    def _load_backbone(self, backbone: str) -> None:
        """
        Added by Ethan Harvey, this initialises model backbone based on config["model"]
        Allows support for gpt-2, BERT, Mini Llama, Phi-2, Gemma-2b
        """
        match backbone:
            case "gpt2":
                self.backbone = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
                self.backbone.h = self.backbone.h[:self.gpt_layers]
            case "bert":
                self.backbone = BertModel.from_pretrained("bert-base-uncased", output_attentions=True, output_hidden_states=True)
                self.backbone.encoder.layer = self.backbone.encoder.layer[:self.gpt_layers]
            case "llama":
                self.backbone = LlamaModel.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                output_hidden_states=True,
                output_attentions=True)
                self.backbone.layers = self.backbone.layers[:self.gpt_layers]
            case "phi":
                self.backbone = PhiModel.from_pretrained(
                    "microsoft/phi-2",
                    output_hidden_states=True,
                    output_attentions=True,
                )
                self.backbone.layers = self.backbone.layers[:self.gpt_layers]
                self.backbone = self.backbone.to(torch.float32)
            case "gemma":
                self.backbone = GemmaModel.from_pretrained("google/gemma-2b", output_hidden_states=True, output_attentions=True)
                self.backbone.layers = self.backbone.layers[:self.gpt_layers]

    def _freeze_backbone(self, backbone: str) -> None:
        """
        Freezes the relevant backbone params based on the config["model"]
        """
        match backbone:
            case "gpt2":
                for i, (name, param) in enumerate(self.backbone.named_parameters()):
                    if 'ln' in name or 'wpe' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            case "bert":
                for i, (name, param) in enumerate(self.backbone.named_parameters()):
                    if "LayerNorm" in name or "position_embeddings" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            case "llama":
                for i, (name, param) in enumerate(self.backbone.named_parameters()):
                    if "norm" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            case "phi":
                for i, (name, param) in enumerate(self.backbone.named_parameters()):
                    if "layernorm" in name.lower():
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            case "gemma":
                for i, (name, param) in enumerate(self.backbone.named_parameters()):
                    if "norm" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

