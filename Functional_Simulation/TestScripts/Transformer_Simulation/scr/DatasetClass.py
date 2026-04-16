

#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        DatasetClass.py
# Description:   Custom PyTorch Dataset class for loading the ImageNet (ILSVRC) dataset from a specific directory structure.
#
# Synopsis:      This class implements a map-style dataset interface (using __init__, 
#                __len__, and __getitem__) to handle the complex hierarchy and 
#                metadata of the ImageNet dataset. It maps WordNet Synset IDs to 
#                integer class labels and generates a list of file paths and 
#                corresponding targets for both training and validation splits.
#
# Created:       2025-11-11
# Last Modified: 2025-12-08
###############################################################################
"""


import os
from torch.utils.data import Dataset
from PIL import Image
import json


#This is defining a dataset class. The three essential parts are init, len and getitem - compulsory. ImageNet is a map-style dataset. 

class ImageNetKaggle(Dataset):
	def __init__(self, root, split, transform=None):
		#creating empty sets for samples, and targets. 
		self.samples = []
		self.targets = []
		self.transform = transform
		self.syn_to_class = {}
		with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
			json_file = json.load(f)
			for class_id, v in json_file.items(): 
				self.syn_to_class[v[0]] = int(class_id)
	
		with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
			self.val_to_syn = json.load(f)
		samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
		for entry in os.listdir(samples_dir):
			if split == "train":
				syn_id = entry
				target = self.syn_to_class[syn_id]
				syn_folder = os.path.join(samples_dir, syn_id)
				for sample in os.listdir(syn_folder):
					sample_path = os.path.join(syn_folder, sample)
					self.samples.append(sample_path)
					self.targets.append(target)
			elif split == "val":
				syn_id = self.val_to_syn[entry]
				target = self.syn_to_class[syn_id]
				sample_path = os.path.join(samples_dir, entry)
				self.samples.append(sample_path)
				self.targets.append(target)
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		x = Image.open(self.samples[idx]).convert("RGB")
		if self.transform:
			x = self.transform(x)
		return x, self.targets[idx]