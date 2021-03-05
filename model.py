import torch as t
from torch import nn
import numpy as np


class Model(nn.Module):
    def __init__(self, n_vprot, n_hprot, hidden_dim, seq_emb_dim = 1900):
        super(Model, self).__init__()

        self.n_vprot = n_vprot
        self.n_hprot = n_hprot
        self.hidden_dim = hidden_dim
        self.seq_emb_dim = seq_emb_dim

        self.vemb = nn.Embedding(n_vprot, seq_emb_dim)
        self.hemb = nn.Embedding(n_hprot, seq_emb_dim)
        self.vlinear = nn.Linear(seq_emb_dim, hidden_dim)
        self.hlinear = nn.Linear(seq_emb_dim, hidden_dim)
        self.assoc_clf = nn.Linear(hidden_dim, 1)
        self.ppi_clf = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, vindex_tensor, hindex_tensor, vh_pairs, hppi_pairs):
        vemb = self.vemb(vindex_tensor)
        hemb = self.hemb(hindex_tensor)

        vhid = self.vlinear(vemb)
        hhid = self.hlinear(hemb)

        vghid = self.relu(vhid)
        hghid = self.relu(hhid)

        assoc_out = self.sigmoid(self.assoc_clf(vghid[vh_pairs[:,0]] * hghid[vh_pairs[:, 1]]))
        hhp_out = self.sigmoid(self.ppi_clf(hghid[hppi_pairs[:,0]] * hghid[hppi_pairs[:, 1]]))
        return assoc_out, hhp_out

