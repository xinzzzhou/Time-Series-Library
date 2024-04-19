import torch
from torch import nn
from model.PatchTST import Model

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        pass
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        pass