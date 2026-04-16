#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        ModelSetupManager.py
# Description:   Centralized model, vocabulary, and data loader initialization
#
# Synopsis:      This module handles the complete setup pipeline for transformer
#                training, including:
#                - Checkpoint detection and vocabulary loading
#                - Vocabulary building/loading from checkpoint
#                - Model creation with correct vocab sizes
#                - Checkpoint weight loading
#                - Data iterator creation
#
# Created:       2026-01-21
# Last Modified: 2026-01-22
###############################################################################
"""

import torch
import os
from torch import nn, optim
from torch.optim import Adam

from DatasetLoads import build_or_load_vocab, finalize_vocab_exports, create_iterators, DEVICE, SRC, TRG
from TransformerBasic import Transformer
from CheckPointManager import (find_best_checkpoint, find_best_checkpoint_with_config, load_checkpoint, 
                                print_checkpoint_summary)


class TransformerLRScheduler:
	def __init__(self, optimizer, d_model, warmup):
		self.optimizer = optimizer
		self.d_model = d_model
		self.warmup = warmup
		self.step_num = 0

	def step(self):
		self.step_num += 1
		lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup ** -1.5))
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr
		return lr

	def state_dict(self):
		"""Return the state of the scheduler as a dict."""
		return {
			'step_num': self.step_num,
			'd_model': self.d_model,
			'warmup': self.warmup
			}
    
	def load_state_dict(self, state_dict):
		"""Load the scheduler state."""
		self.step_num = state_dict.get('step_num', 0)
		self.d_model = state_dict.get('d_model', self.d_model)
		self.warmup = state_dict.get('warmup', self.warmup)
		print(f"Restored TransformerLRScheduler: step_num={self.step_num}")


def initialize_weights(m):
	if hasattr(m,'weight') and m.weight.dim()>1:
		#nn.init.kaiming_uniform_(m.weight.data)
		nn.init.xavier_uniform_(m.weight.data)


def setup_model_and_data(config):
	"""
	Complete setup pipeline for model, vocabularies, and data loaders.
    
	Args:
		config (dict): Configuration dictionary containing all necessary parameters:
			- resume_training (bool): Whether to resume from checkpoint
			- strict_config (bool): Whether to enforce strict config validation
			- d_model (int): Model dimension
			- n_heads (int): Number of attention heads
			- n_layers (int): Number of transformer layers
			- ffn_hidden (int): Feed-forward network hidden dimension
			- drop_prob (float): Dropout probability
			- batch_size (int): Batch size
			- max_len (int): Maximum sequence length
			- mode (int): Execution mode (0=CMOS, 1=MESO IMC)
			- bit_width (int): Bit width for MESO IMC
			- init_lr (float): Initial learning rate
			- weight_decay (float): Weight decay
			- adam_eps (float): Adam epsilon
			- warmup (int): Warmup steps
			- label_smoothing (float): Label smoothing factor
			- factor (float): LR reduction factor
			- patience (int): Patience for LR scheduler
			- inf (float): Infinity value
            
	Returns:
		dict: Dictionary containing all initialized components:
			- model: Initialized Transformer model
			- optimizer: Adam optimizer
			- lr_scheduler: Custom Transformer LR scheduler
			- scheduler: ReduceLROnPlateau scheduler
			- criterion: Loss criterion
			- train_iter: Training data iterator
			- valid_iter: Validation data iterator
			- test_iter: Test data iterator
			- vocab_de: German vocabulary
			- vocab_en: English vocabulary
			- src_pad_idx: Source padding index
			- trg_pad_idx: Target padding index
			- trg_sos_idx: Target SOS index
			- enc_voc_size: Encoder vocabulary size
			- dec_voc_size: Decoder vocabulary size
			- start_epoch: Starting epoch number
			- best_loss: Best validation loss so far
			- train_losses: List of training losses
			- test_losses: List of validation losses
			- bleus: List of BLEU scores
	"""
    
	# Extract config parameters
	resume_training = config['resume_training']
	strict_config = config['strict_config']
    
	# ============================================================================
	# STEP 1: Check if we're resuming and find checkpoint
	# ============================================================================
	print("\n" + "="*80)
	print("MODEL SETUP PIPELINE")
	print("="*80)
    
	start_epoch = 0
	best_loss = config['inf']
	train_losses, test_losses, bleus = [], [], []
	src_vocab_loaded = None
	trg_vocab_loaded = None
	best_checkpoint = None


	if resume_training:
		print("\n[STEP 1/5] Searching for config-matching checkpoint...")
		print("-" * 80)
    
		current_config = {
			'd_model': config['d_model'],
			'n_heads': config['n_heads'],
			'n_layers': config['n_layers'],
			'ffn_hidden': config['ffn_hidden'],
			'drop_prob': config['drop_prob'],
			'batch_size': config['batch_size'],
			'max_len': config['max_len'],
			'mode': config['mode']
			}

		best_checkpoint, config_matches, warnings = find_best_checkpoint_with_config(current_config=current_config,
			checkpoint_dir='saved', strict=strict_config)
    
		if best_checkpoint:
			if config_matches:
				print(f"\n Found config-matching checkpoint: {os.path.basename(best_checkpoint)}")
			else:
				print(f"\n Using non-matching checkpoint: {os.path.basename(best_checkpoint)}")
				for warning in warnings:
					print(f"  {warning}")
		else:
			print("\n No suitable checkpoint found - will start fresh training")
			if warnings:
				for warning in warnings:
					print(f"  {warning}")

	#if resume_training:
	#	print("\n[STEP 1/5] Checking for existing checkpoint...")
	#	print("-" * 80)
        #
	#	print_checkpoint_summary('saved')
	#	best_checkpoint = find_best_checkpoint('saved')
        #
	#	if best_checkpoint:
	#		print(f"Found checkpoint: {best_checkpoint}")
	#	else:
	#		print("No checkpoint found - will start fresh training")
	#else:
	#	print("\n[STEP 1/5] Starting fresh training (resume_training=False)")
	#	print("-" * 80)

	# ============================================================================
	# STEP 2: Load vocabularies from checkpoint OR build new ones
	# ============================================================================
	print("\n[STEP 2/5] Loading vocabularies...")
	print("-" * 80)
    
	if best_checkpoint and resume_training:
		print("Attempting to extract vocabularies from checkpoint...")
		try:
			checkpoint = torch.load(best_checkpoint, map_location=DEVICE, weights_only=False)
			if isinstance(checkpoint, dict):
				src_vocab_loaded = checkpoint.get('src_vocab', None)
				trg_vocab_loaded = checkpoint.get('trg_vocab', None)
				if src_vocab_loaded and trg_vocab_loaded:
					print("Vocabularies found in checkpoint")
				else:
					print("No vocabularies in checkpoint (old format - will build new)")
		except Exception as e:
			print(f"Warning: Could not pre-load vocabularies: {e}")

	# Build or load vocabularies
	vocab_de, vocab_en = build_or_load_vocab(src_vocab=src_vocab_loaded, trg_vocab=trg_vocab_loaded)

	# Finalize vocab exports
	src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size = finalize_vocab_exports()

	print(f"Vocabulary ready")
	#print(f"  Source vocab size: {enc_voc_size}")
	#print(f"  Target vocab size: {dec_voc_size}")
	#print(f"  Pad idx: {src_pad_idx}, SOS idx: {trg_sos_idx}")

	# ============================================================================
	# STEP 3: Create the model with correct vocab sizes
	# ============================================================================
	print("\n[STEP 3/5] Creating model...")
	print("-" * 80)
    
	model = Transformer(
		src_pad_idx=src_pad_idx,
		trg_pad_idx=trg_pad_idx,
		trg_sos_idx=trg_sos_idx,
		enc_voc_size=enc_voc_size,
		max_len=config['max_len'],
		d_model=config['d_model'],
		ffn_hidden=config['ffn_hidden'],
		n_head=config['n_heads'],
		n_layers=config['n_layers'],
		drop_prob=config['drop_prob'],
		device=DEVICE,
		dec_voc_size=dec_voc_size,
		mode=config['mode'],
		bit_width=config['bit_width'],
		cycle_res=config['cycle_res'],
		var=config['var']
		).to(DEVICE)

	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'Model created with {num_params:,} trainable parameters')

	# Initialize weights (will be overwritten if loading checkpoint)
	model.apply(initialize_weights)

	# Create optimizer and schedulers
	criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=config['label_smoothing'])

	optimizer = Adam(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'], betas=(0.9, 0.98), eps=config['adam_eps'])

	lr_scheduler = TransformerLRScheduler(optimizer, d_model=config['d_model'], warmup=config['warmup'])

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=config['factor'], patience=config['patience'])

	# ============================================================================
	# STEP 4: Load checkpoint weights into the model
	# ============================================================================
	if best_checkpoint and resume_training:
		print("\n[STEP 4/5] Loading checkpoint weights...")
		print("-" * 80)
        
		current_config = {
			'd_model': config['d_model'],
			'n_heads': config['n_heads'],
			'n_layers': config['n_layers'],
			'ffn_hidden': config['ffn_hidden'],
			'drop_prob': config['drop_prob'],
			'batch_size': config['batch_size'],
			'max_len': config['max_len'],
			'mode': config['mode']
			}
        
		start_epoch, best_loss, train_losses, test_losses, bleus, _, _ = load_checkpoint(model=model,
			checkpoint_path=best_checkpoint,
			optimizer=optimizer,
			lr_scheduler=lr_scheduler,
			device=DEVICE,
			current_config=current_config,
			strict_config=strict_config
			)
        
		if start_epoch == 0 and best_loss == float('inf'):
			print("✗ Failed to load checkpoint - starting fresh instead")
			model.apply(initialize_weights)
			train_losses, test_losses, bleus = [], [], []
		else:
			print(f"✓ Successfully resumed from epoch {start_epoch}")
			print(f"  Will continue training from epoch {start_epoch + 1}")
	else:
		print("\n[STEP 4/5] Skipping checkpoint loading (fresh training)")
		print("-" * 80)

	# ============================================================================
	# STEP 5: Create data iterators
	# ============================================================================
	print("\n[STEP 5/5] Creating data iterators...")
	print("-" * 80)
    
	train_iter, valid_iter, test_iter = create_iterators()
	print("Data iterators created")
    
	print("\n" + "="*80)
	print("SETUP COMPLETE - Ready to train!")
	print("="*80 + "\n")

	# Return everything in a dictionary
	return {
		'model': model,
		'optimizer': optimizer,
		'lr_scheduler': lr_scheduler,
		'scheduler': scheduler,
		'criterion': criterion,
		'train_iter': train_iter,
		'valid_iter': valid_iter,
		'test_iter': test_iter,
		'vocab_de': vocab_de,
		'vocab_en': vocab_en,
		'src_pad_idx': src_pad_idx,
		'trg_pad_idx': trg_pad_idx,
		'trg_sos_idx': trg_sos_idx,
		'enc_voc_size': enc_voc_size,
		'dec_voc_size': dec_voc_size,
		'start_epoch': start_epoch,
		'best_loss': best_loss,
		'train_losses': train_losses,
		'test_losses': test_losses,
		'bleus': bleus
		}