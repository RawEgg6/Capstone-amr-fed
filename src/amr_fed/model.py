"""AMR-SAGE: heterogeneous GraphSAGE for per-test resistance prediction.

Encoder: HeteroConv of SAGEConv over every (incl. reverse) edge type -> node
embeddings for patient / organism / antibiotic. Decoder: an MLP over the
concatenated (h_patient, h_organism, h_antibiotic) for each supervision triple
-> a single resistance logit.

Runs on Colab (torch). Architecture is identical across clients (federation
requires it) — keep hyperparameters in one place.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


class AMRSAGE(nn.Module):
    def __init__(self, edge_types, hidden: int = 64, layers: int = 2,
                 decoder_hidden: int = 64, dropout: float = 0.3, aggr: str = "sum"):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for _ in range(layers):
            # (-1, -1) = lazy input dims, inferred per relation on first forward.
            # aggr = how a node combines messages from its DIFFERENT edge types
            # ("sum" lets high-degree relations dominate; "mean" balances them).
            conv = HeteroConv({et: SAGEConv((-1, -1), hidden) for et in edge_types}, aggr=aggr)
            self.convs.append(conv)
        self.decoder = nn.Sequential(
            nn.Linear(3 * hidden, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, 1),
        )

    def encode(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                      for k, v in x_dict.items()}
        return x_dict

    def decode(self, h, triple_index):
        # triple_index: [3, N] rows = patient, organism, antibiotic node ids
        z = torch.cat([h["patient"][triple_index[0]],
                       h["organism"][triple_index[1]],
                       h["antibiotic"][triple_index[2]]], dim=-1)
        return self.decoder(z).squeeze(-1)

    def forward(self, x_dict, edge_index_dict, triple_index):
        return self.decode(self.encode(x_dict, edge_index_dict), triple_index)
