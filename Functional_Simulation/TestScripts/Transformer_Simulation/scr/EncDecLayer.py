

#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        EncDecLayer.py
# Description:   Core building blocks (layers and stacks) of the Transformer Encoder-Decoder architecture.
#
# Synopsis:      This module implements the fundamental units of the Transformer: 
#                the EncoderLayer and DecoderLayer. It manages residual 
#                connections, Layer Normalization, and Dropout around Multi-Head 
#                Attention and Feed-Forward Networks. It also defines the full 
#                Encoder and Decoder stacks by combining multiple layers with 
#                initial embedding/projection layers.
# Created:       2025-11-11
# Last Modified: 2026-04-15
###############################################################################
"""

import torch
from torch import nn
from Attention import MultiHeadAttention,FFN
from LayerNorm import LayerNorm
from Embeddings import TransformerEmbedding


class EncoderLayer(nn.Module):
	
	def __init__(self, d_model, ffn_hidden, n_head, drop_prob, mode, bit_width, cycle_res, var):
		super(EncoderLayer,self).__init__()
		self.attention=MultiHeadAttention(d_model, n_head, mode, bit_width, cycle_res, var)
		self.norm1=LayerNorm(d_model)
		self.dropout1=nn.Dropout(p=drop_prob)
		self.ffn=FFN(d_model,ffn_hidden,d_model,drop_prob)
		self.norm2=LayerNorm(d_model)
		self.dropout2=nn.Dropout(p=drop_prob)


	def forward(self,x,src_mask):
		_x=x
		x=self.attention(q=x,k=x,v=x,mask=src_mask)
		x=self.dropout1(x)
		x=self.norm1(x+_x)
		_x=x
		x=self.ffn(x)
		x=self.dropout2(x)
		x=self.norm2(x+_x)
		return x
		




class DecoderLayer(nn.Module):
	
	def __init__(self, d_model, ffn_hidden, n_head, drop_prob,mode, bit_width, cycle_res, var):
		super(DecoderLayer,self).__init__()
		self.attention=MultiHeadAttention(d_model, n_head, mode, bit_width, cycle_res, var)
		self.norm1=LayerNorm(d_model)
		self.dropout1=nn.Dropout(p=drop_prob)


		self.crossattention=MultiHeadAttention(d_model, n_head, mode, bit_width, cycle_res, var)
		self.norm2=LayerNorm(d_model)
		self.dropout2=nn.Dropout(p=drop_prob)

		self.ffn=FFN(d_model,ffn_hidden,d_model,drop_prob)
		self.norm3=LayerNorm(d_model)
		self.dropout3=nn.Dropout(p=drop_prob)


	def forward(self,x,enc,trg_mask,src_mask):
		_x=x
		x=self.attention(q=x,k=x,v=x,mask=trg_mask)
		x=self.dropout1(x)
		x=self.norm1(x+_x)
		if enc is not None:
			_x=x
			x=self.crossattention(q=x,k=enc,v=enc,mask=src_mask)
			x=self.dropout2(x)
			x=self.norm2(x+_x)
		_x=x
		x=self.ffn(x)
		x=self.dropout3(x)
		x=self.norm3(x+_x)
		return x



class Encoder(nn.Module):
	def __init__(self,enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,drop_prob,mode,bit_width,cycle_res, var, device):
		super().__init__()
		self.emb=TransformerEmbedding(d_model=d_model,max_len=max_len,vocab_size=enc_voc_size,drop_prob=drop_prob,device=device)
		self.layers=nn.ModuleList([EncoderLayer(d_model,ffn_hidden,n_head,drop_prob,mode,bit_width,cycle_res,var) for _ in range(n_layers)])

	def forward(self,x,src_mask):
		x=self.emb(x)
		for layer in self.layers:
			x=layer(x,src_mask)
		return x


class Decoder(nn.Module):
	def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,drop_prob,mode,bit_width,cycle_res,var,device):
		super().__init__()
		self.emb=TransformerEmbedding(d_model=d_model,max_len=max_len,vocab_size=dec_voc_size,drop_prob=drop_prob,device=device)
		self.layers=nn.ModuleList([DecoderLayer(d_model,ffn_hidden,n_head,drop_prob,mode,bit_width,cycle_res,var) for _ in range(n_layers)])
		self.linear = nn.Linear(d_model, dec_voc_size)

	def forward(self,trg,src,trg_mask,src_mask):
		x=self.emb(trg)
		for layer in self.layers:
			x=layer(x,src,trg_mask,src_mask)
		x=self.linear(x)
		return x

















