

#!/usr/bin/env python3
"""
###############################################################################
# Module:        DatasetClass.py
# Description:   Custom PyTorch Dataset class for loading the ImageNet (ILSVRC) dataset from a specific directory structure.
#
# Synopsis:      This class implements a map-style dataset interface (using __init__, 
#                __len__, and __getitem__) to handle the complex hierarchy and 
#                metadata of the ImageNet dataset. It maps WordNet Synset IDs to 
#                integer class labels and generates a list of file paths and 
#                corresponding targets for both training and validation splits.
#
###############################################################################
"""


import os
from torch.utils.data import Dataset
from PIL import Image
import json


#This is defining a dataset class. The three essential parts are init, len and getitem - compulsory. ImageNet is a map-style dataset. 

class ImageNetKaggle(Dataset):
	#root is the main directory location, split being train or validation set. transform is basically how the tensors need to be modified.
	def __init__(self, root, split, transform=None):
		#creating empty sets for samples, and targets. 
		self.samples = []
		self.targets = []
		self.transform = transform
		#not sure what syn_to_class is. 
		self.syn_to_class = {}
		#opens the given file for reading binary. The json file is given in text. We iterate through the json file and create a data table that can be used by our program.
		with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
			json_file = json.load(f)
			for class_id, v in json_file.items():
				#defining the file type to a particular class_id. This is effectively the 0-9 values in MNIST database. Each image is mapped to a particular class. The image has a weird number in its name, which is then mapped to class through this variable. 
				self.syn_to_class[v[0]] = int(class_id)
	
		#opens the given file that gives the labels for the validation data?
		with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
			self.val_to_syn = json.load(f)
		samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
		#os.listdir - gives a list of all the directories within the present directory.
		for entry in os.listdir(samples_dir):
			if split == "train":
				#entry is the folder name. This is effectively used to map the photos in a particular folder to a particular class. Since syn_to_class has a mapping of the name to the class, we use that to generate the required target class for the images currently being processed. 
				syn_id = entry
				target = self.syn_to_class[syn_id]
				syn_folder = os.path.join(samples_dir, syn_id)
				#Here, we are creating a list with the path to each of the photos and the targets. Same as the initial example we saw, where we give the path to the image we are interested in classifying. 
				for sample in os.listdir(syn_folder):
					sample_path = os.path.join(syn_folder, sample)
					self.samples.append(sample_path)
					self.targets.append(target)
			elif split == "val":
				#val doesnt have the directory structure that we used for the training database. So, we use the second file to determine what the target ought to be. 
				syn_id = self.val_to_syn[entry]
				target = self.syn_to_class[syn_id]
				sample_path = os.path.join(samples_dir, entry)
				self.samples.append(sample_path)
				self.targets.append(target)
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		#effectively open the image and convert it to RGB structure. 
		x = Image.open(self.samples[idx]).convert("RGB")
		if self.transform:
			x = self.transform(x)
		#return the values of the image after applying the transform and the target. 
		return x, self.targets[idx]