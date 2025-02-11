#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import math
import time,os,sys
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import zipfile
from ..utils.utils2 import (
    log_std_denorm_dataset,
    cos_date,
    sin_date,
    adjust_learning_rate,
)
random.seed('a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class EncoderLSTM(nn.Module):
    def __init__(self, opt):
        super().__init__()        
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.lstm_dim = opt.hidden_dim
        self.opt = opt
        
        self.lstm1 = nn.LSTM(2, self.lstm_dim, self.layer_dim, batch_first=True)
        
    def forward(self, x, h, c):
        # Initialize hidden and cell state with zeros   

        h0 = torch.zeros(self.layer_dim, x.size(0), self.lstm_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.lstm_dim).to(device)    
        x0 = x[:,-480:,0:2]                    
        hn = []
        cn = []
        out, (hn1, cn1) = self.lstm1(x0, (h0,c0))

        hn.append(hn1)
        cn.append(cn1)
        
        return hn, cn
    
class DecoderLSTM(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hidden_dim = opt.hidden_dim
        self.lstm_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.output_len = opt.output_len
        self.opt = opt
        self.dim_in = 2 + opt.r_shift
            
        self.fc_embed1 = nn.Linear(self.dim_in, 512)   
        self.fc_embed2 = nn.Linear(512, self.hidden_dim) 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)
        self.emb_layer = nn.Linear(2, self.hidden_dim)
        self.attn0 = nn.MultiheadAttention(self.hidden_dim, 4)
        self.L_out10 = nn.Linear(self.hidden_dim, 512)  
        self.L_out20 = nn.Linear(512, self.hidden_dim) 
        self.L_out30 = nn.Linear(self.hidden_dim, 1) 
        self.attn3 = nn.MultiheadAttention(self.hidden_dim, 4)
        self.bn = nn.BatchNorm1d(self.output_len)
     
        self.lstm01 = nn.LSTM(2, self.lstm_dim, self.layer_dim, dropout=0, batch_first=True)
        self.L_out01 = nn.Linear(self.lstm_dim, 1) 
        
    def forward(self, x1, x3, encoder_h, encoder_c):  # x1: time sin & cos; x3: input sequence
        # Initialize hidden and cell state with zeros
        h0 = encoder_h
        c0 = encoder_c
        
        Lrelu = nn.LeakyReLU(0.1)
        relu = nn.ReLU()
        T = nn.Tanh()

        # FC encoder embedding
        src = self.fc_embed2(relu(self.fc_embed1(x3[:,:,:self.dim_in])))
        
        # LSTM decoder embedding
        out, (hn, cn) = self.lstm01(x1, (h0[0],c0[0]))  
        out02 = self.L_out01(out)
        
        # Transformer encoder        
        memory = self.transformer_encoder(src)
        memory = memory[:,-1*self.output_len:,:]
        
        # Cross attention decoder
        ww0 = out 
        ww00,_ = self.attn0(ww0,memory,memory)
        ww0 = self.bn(ww0 + ww00)  # add & norm
        ww00,_ = self.attn3(ww0,ww0,ww0)        
        ww00 = self.bn(ww0 + ww00)  # add & norm    
        ww0 = self.L_out20(relu(self.L_out10(ww00)))# feed forward layer        
        out01 = self.L_out30(self.bn(ww0 + ww00)) # add & norm &linear
        
        # output
        out01 = torch.squeeze(out01)
        out02 = torch.squeeze(out02) 
        out01 = out01  + out02

        return out01, out02


