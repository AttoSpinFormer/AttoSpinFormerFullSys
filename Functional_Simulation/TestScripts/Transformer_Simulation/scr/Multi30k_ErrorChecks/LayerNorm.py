

#!/usr/bin/env python3
"""
###############################################################################
# Module:        LayerNorm.py
# Description:   PyTorch implementation of the Layer Normalization module (LayerNorm).
#
# Synopsis:      This module applies normalization across the last dimension (the feature 
#                or embedding dimension) of the input tensor. Unlike Batch Normalization, 
#                it computes mean and variance statistics independently for each sample 
#                and feature across the layer, ensuring consistent performance regardless 
#                of the batch size.
#
###############################################################################
"""

import torch
from torch import nn




class LayerNorm(nn.Module):
	
	def __init__(self,d_model,eps=1e-12):
		super(LayerNorm,self).__init__()
		self.gamma=nn.Parameter(torch.ones(d_model))
		self.beta=nn.Parameter(torch.zeros(d_model))
		self.eps=eps
		#This is for running mean. However, transformers that deal with sequential data donot use batchNorm that normalizes over a batch. Instead, they only normalize over each layer. 
		#self.mean=torch.zeros()
		#self.var=torch.zeros()
		#self.momentum=momentum

	def forward(self,x):
		#the original code is not for running mean. Again, it is simple layer normalization that transformers use. However, for typical CNNs, we can use running mean and running var. However, the updates only during training for batch norm and hence, we need to check if its training or not.
		#self.mean=self.mean*(1-self.momentum)+x.mean(-1,keepdim=True)*self.momentum
		#self.var=self.var*(1-self.momentum)+x.var(-1,correction=0,keepdim=True)*self.momentum
		mean=x.mean(-1,keepdim=True)
		var=x.var(-1,correction=0,keepdim=True)
		out=(x-mean)/ torch.sqrt(var + self.eps)
		out=self.gamma*out+self.beta
		return out