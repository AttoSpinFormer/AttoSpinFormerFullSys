

#!/usr/bin/env python3
"""
###############################################################################
# Module:        DataSetLoads.py
# Description:   Data processing and utility module for the Multi30k machine translation dataset.
#
# Synopsis:      This module manages the full data pipeline setup for sequence-to-sequence 
#                (Seq2Seq) experiments. It configures the device, loads the SpaCy 
#                language models, defines tokenizers and Field objects, loads the 
#                Multi30k corpus, builds the source (German) and target (English) 
#                vocabularies, and creates padded, batched BucketIterators for training.
#
###############################################################################

Usage and Interface:

    This script is intended to be imported to provide necessary data utilities and iterators.

Exported Objects:
    train_iter (BucketIterator): Training data iterator, batched and padded.
    valid_iter (BucketIterator): Validation data iterator, batched and padded.
    test_iter (BucketIterator):  Test data iterator, batched and padded.
    vocab_de (torchtext.Vocab):  Vocabulary object for the German (Source) language.
    vocab_en (torchtext.Vocab):  Vocabulary object for the English (Target) language.
    DEVICE (torch.device):       The determined PyTorch device ('mps' or 'cpu').

Dependencies:
    Requires pre-configured model parameters (e.g., 'batch_size') from Config.py.
    Requires SpaCy models ('en_core_web_sm', 'de_core_news_sm') to be downloaded.
"""



from Config import *
import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy 
import os


# Load SpaCy models
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

def tokenize_de(text):
	return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]

# Define Field objects for German (Source)
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>',lower=True, batch_first=True)

# Define Field objects for English (Target)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>',lower=True, batch_first=True)


# Standard approach: Source first, then Target
# German is Source, English is Target
# Match fields to extensions
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),  fields=(SRC, TRG), path='../../datas/Multi30k')


print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(valid_data)}")
print(f"Test examples: {len(test_data)}")


def build_or_load_vocab(src_vocab=None, trg_vocab=None):
	"""
	Build vocabularies or use provided ones from checkpoint.
    
	Args:
		src_vocab: Pre-built source vocabulary (from checkpoint)
		trg_vocab: Pre-built target vocabulary (from checkpoint)
	"""
	print("\n" + "="*60)
	print("VOCABULARY LOADING")
	print("="*60)
    
	if src_vocab is not None and trg_vocab is not None:
		# Use vocabularies from checkpoint
		SRC.vocab = src_vocab
		TRG.vocab = trg_vocab
		print("Using vocabularies from checkpoint")
	else:
		# Build new vocabularies
		print("Building new vocabularies (this may take a moment)...")
		SRC.build_vocab(train_data, min_freq=2)
		TRG.build_vocab(train_data, min_freq=2)
		print("Vocabularies built from scratch")
    
	print(f"Unique tokens in Source (DE): {len(SRC.vocab)}")
	print(f"Unique tokens in Target (EN): {len(TRG.vocab)}")
	print(f"Special tokens: {SRC.vocab.itos[:10]}")
	print("="*60 + "\n")
	return SRC.vocab, TRG.vocab


def create_iterators():
	"""Create BucketIterators after vocabularies are loaded."""
	global train_iter, valid_iter, test_iter
    
	train_iter, valid_iter, test_iter = BucketIterator.splits(
		(train_data, valid_data, test_data),
		batch_size=batch_size,
		sort_key=lambda x: len(x.src),
		device=DEVICE
		)
    
	return train_iter, valid_iter, test_iter
	

train_iter=None
valid_iter=None
test_iter=None

# Diagnostic: Check first training example
print("="*60)
print("DATA LOADING VERIFICATION")
print("="*60)

first_example = train_data[0]
print(f"\nFirst training example:")
print(f"  SRC (should be German): {first_example.src[:10]}")
print(f"  TRG (should be English): {first_example.trg[:10]}")

for i in range(3):
	example = train_data[i]
	print(f"\nExample {i+1}:")
	print(f"  SRC tokens: {example.src}")
	print(f"  TRG tokens: {example.trg}")
	print(f"  SRC length: {len(example.src)}")
	print(f"  TRG length: {len(example.trg)}")

print("="*60 + "\n")

# Export vocab objects
vocab_de = None 
vocab_en = None

src_pad_idx = None
trg_pad_idx = None
trg_sos_idx = None
enc_voc_size = None
dec_voc_size = None


def finalize_vocab_exports():
	"""Call this after vocabularies are loaded to set all export variables."""
	global vocab_de, vocab_en, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size
    
	vocab_de = SRC.vocab
	vocab_en = TRG.vocab
	src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
	trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]
	trg_sos_idx = TRG.vocab.stoi[TRG.init_token]
	enc_voc_size = len(SRC.vocab)
	dec_voc_size = len(TRG.vocab)
    
	return src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size

