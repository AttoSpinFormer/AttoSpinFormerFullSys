
#!/usr/bin/env python3
"""
###############################################################################
# Module:        TransformerBasic.py
# Description:   The Transformer model (encoder-decoder) for sequence-to-sequence tasks.
#
# Synopsis:      This module implements the full Transformer model, consisting of 
#                stacked Encoder and Decoder components. It manages the token padding 
#                indices and, critically, contains the logic to generate the necessary 
#                **source mask (padding mask)** and the **target mask (combined 
#                padding and look-ahead mask)** used to control the attention flow 
#                during training and inference.
#
###############################################################################
"""


import torch
from torch import nn
from torch.utils.data import DataLoader,random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from EncDecLayer import Encoder,Decoder


class Transformer(nn.Module):

	def __init__(self,src_pad_idx,trg_pad_idx,trg_sos_idx,enc_voc_size,max_len,d_model,ffn_hidden,n_head, n_layers,drop_prob,device,dec_voc_size,mode,bit_width, cycle_res, var):
		super().__init__()
		#these are the padding tokens added to input sequences to make the length of the inputs to be the same. Ob, we need everything to be of the same length. However, we dont want the model to focus on these padding tokens. So, we use the masking part to prevent it from focussing on them. 
		self.src_pad_idx=src_pad_idx
		self.trg_pad_idx=trg_pad_idx
		self.trg_sos_idx=trg_sos_idx
		self.device=device	
		self.encoder=Encoder(enc_voc_size=enc_voc_size,
			max_len = max_len,
			d_model = d_model,
			ffn_hidden = ffn_hidden,
			n_head = n_head,
			n_layers = n_layers,
			drop_prob = drop_prob,
			mode = mode,
			bit_width = bit_width,
			cycle_res = cycle_res,
			var = var,
			device = device)
		self.decoder=Decoder(dec_voc_size=dec_voc_size,
			max_len = max_len,
			d_model = d_model,
			ffn_hidden = ffn_hidden,
			n_head = n_head,
			n_layers = n_layers,
			drop_prob = drop_prob,
			mode = mode,
			bit_width = bit_width,
			cycle_res = cycle_res,
			var = var,
			device = device)


	def forward(self,src,trg):
		src_mask=self.make_src_mask(src)
		trg_mask=self.make_trg_mask(trg)
		encOut=self.encoder(src,src_mask)
		Out=self.decoder(trg,encOut,trg_mask,src_mask)
		return Out	
	
	def make_src_mask(self,src):
		src_mask= (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
		return src_mask


	def make_trg_mask(self,trg):
		trg_pad_mask=(trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
		trg_len=trg.shape[1]
		trg_sub_mask=torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
		trg_mask=trg_pad_mask & trg_sub_mask
		return trg_mask


