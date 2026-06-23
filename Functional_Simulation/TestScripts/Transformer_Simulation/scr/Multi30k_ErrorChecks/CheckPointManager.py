#!/usr/bin/env python3
"""
###############################################################################
# Module:        CheckpointManager.py
# Description:   Checkpoint management utilities for saving and loading model states
#
# Synopsis:      This module provides functions to save, load, and manage model
#                checkpoints during training. It handles finding the best checkpoint,
#                loading model and optimizer states, and saving training progress.
#
###############################################################################
"""

import os
import glob
import torch
from datetime import datetime


def clear_vocab_cache(vocab_dir='saved/vocab'):
	"""
	Clear cached vocabulary files.
	Useful when you want to rebuild vocabularies from scratch.
	
	Args:
		vocab_dir (str): Directory containing cached vocabulary files
	"""
	if not os.path.exists(vocab_dir):
		print(f"No vocab directory found at {vocab_dir}")
		return
	
	vocab_files = glob.glob(os.path.join(vocab_dir, '*.pkl'))
	
	if not vocab_files:
		print("No cached vocabularies found")
		return
	
	for filepath in vocab_files:
		try:
			os.remove(filepath)
			print(f"Removed: {os.path.basename(filepath)}")
		except Exception as e:
			print(f"Warning: Could not remove {filepath}: {e}")
	
	print(f"Cleared {len(vocab_files)} vocabulary cache file(s)")


def print_vocab_info(vocab_dir='saved/vocab'):
	"""
	Print information about cached vocabularies.
	
	Args:
		vocab_dir (str): Directory containing cached vocabulary files
	"""
	if not os.path.exists(vocab_dir):
		print(f"No vocab directory found at {vocab_dir}")
		return
	
	vocab_files = glob.glob(os.path.join(vocab_dir, '*.pkl'))
	
	if not vocab_files:
		print("No cached vocabularies found")
		return
	
	print("\n" + "="*60)
	print("CACHED VOCABULARIES")
	print("="*60)
	
	for filepath in vocab_files:
		filename = os.path.basename(filepath)
		file_size = os.path.getsize(filepath)
		modified_time = os.path.getmtime(filepath)
		
		
		mod_time_str = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
		
		print(f"{filename}")
		print(f"  Size: {file_size / 1024:.2f} KB")
		print(f"  Modified: {mod_time_str}")
	
	print("="*60 + "\n")


def find_best_checkpoint(checkpoint_dir='saved'):
	"""
	Find the checkpoint with the lowest validation loss.
    
	Args:
		checkpoint_dir (str): Directory containing checkpoint files
        
	Returns:
		str or None: Path to the best checkpoint file, or None if not found
	"""
	if not os.path.exists(checkpoint_dir):
		print(f"No checkpoint directory found at {checkpoint_dir}")
		return None
    
	# Get all saved model files
	checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model-*.pt'))
    
	if not checkpoint_files:
		print("No checkpoints found")
		return None
    
	# Extract loss values from filenames (format: model-{loss}.pt)
	checkpoints_with_loss = []
	for filepath in checkpoint_files:
		try:
			# Extract the loss value from filename
			filename = os.path.basename(filepath)
			loss_str = filename.split('-loss')[-1].replace('.pt', '')
			loss_value = float(loss_str)
			checkpoints_with_loss.append((filepath, loss_value))
		except ValueError:
			print(f"Skipping invalid checkpoint filename: {filepath}")
			continue
    
	if not checkpoints_with_loss:
		print("No valid checkpoints found")
		return None
    
	# Sort by loss and get the best one
	checkpoints_with_loss.sort(key=lambda x: x[1])
	best_checkpoint, best_loss = checkpoints_with_loss[0]
    
	print(f"\nFound {len(checkpoints_with_loss)} checkpoint(s):")
	for path, loss in checkpoints_with_loss[:5]:  # Show top 5
		print(f"  - {os.path.basename(path)} (loss: {loss:.4f})")
	if len(checkpoints_with_loss) > 5:
		print(f"  ... and {len(checkpoints_with_loss) - 5} more")
	print(f"\nBest checkpoint: {os.path.basename(best_checkpoint)} (loss: {best_loss:.4f})")
    
	return best_checkpoint



def find_best_checkpoint_with_config(current_config, checkpoint_dir='saved', strict=True):
	"""
	Find the checkpoint with the lowest validation loss that matches the current config.
	
	Args:
		current_config (dict): Current model configuration
		checkpoint_dir (str): Directory containing checkpoint files
		strict (bool): If True, only return checkpoints with matching config.
		              If False, return best checkpoint even if config doesn't match.
	
	Returns:
		tuple: (checkpoint_path, config_matches, warnings)
		       Returns (None, False, []) if no suitable checkpoint found
	"""
	if not os.path.exists(checkpoint_dir):
		print(f"No checkpoint directory found at {checkpoint_dir}")
		return None, False, []
	
	# Get all saved model files
	checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model-*.pt'))
	
	if not checkpoint_files:
		print("No checkpoints found")
		return None, False, []
	
	# Critical parameters that MUST match
	critical_params = ['d_model', 'n_heads', 'n_layers', 'ffn_hidden']
	
	# Analyze all checkpoints
	matching_checkpoints = []
	non_matching_checkpoints = []
	
	for filepath in checkpoint_files:
		# Skip if can't extract loss from filename
		try:
			filename = os.path.basename(filepath)
			if filename == 'model-latest.pt':
				continue
			loss_str = filename.split('-loss')[-1].replace('.pt', '')
			loss_value = float(loss_str)
		except (ValueError, IndexError):
			print(f"Skipping invalid checkpoint filename: {filepath}")
			continue
		
		# Load checkpoint to check config
		try:
			checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
		except Exception as e:
			print(f"Skipping checkpoint {filename}: Could not load ({e})")
			continue
		
		# Check if checkpoint has config
		if not isinstance(checkpoint, dict) or 'config' not in checkpoint:
			print(f"Skipping {filename}: No config found (old format)")
			non_matching_checkpoints.append((filepath, loss_value, None))
			continue
		
		checkpoint_config = checkpoint['config']
		
		# Check if config matches
		is_valid, warnings, errors = validate_config(checkpoint_config, current_config)
		
		if is_valid:
			matching_checkpoints.append((filepath, loss_value, checkpoint_config))
		else:
			non_matching_checkpoints.append((filepath, loss_value, checkpoint_config))
	
	# Report findings
	print(f"\nFound {len(matching_checkpoints)} checkpoint(s) matching current config")
	print(f"Found {len(non_matching_checkpoints)} checkpoint(s) with different config")
	
	# If we have matching checkpoints, use the best one
	if matching_checkpoints:
		matching_checkpoints.sort(key=lambda x: x[1])  # Sort by loss
		best_checkpoint, best_loss, best_config = matching_checkpoints[0]
		
		print(f"\n{'='*60}")
		print("MATCHING CHECKPOINTS")
		print(f"{'='*60}")
		for i, (path, loss, cfg) in enumerate(matching_checkpoints[:5], 1):
			marker = "  <-- SELECTED" if i == 1 else ""
			print(f"  {i}. {os.path.basename(path)} (loss: {loss:.4f}){marker}")
		if len(matching_checkpoints) > 5:
			print(f"  ... and {len(matching_checkpoints) - 5} more")
		print(f"{'='*60}")
		
		print(f"\n✓ Best matching checkpoint: {os.path.basename(best_checkpoint)} (loss: {best_loss:.4f})")
		return best_checkpoint, True, []
	
	# No matching checkpoints found
	if strict:
		print(f"\n{'='*60}")
		print("⚠ WARNING: No checkpoints found matching current config!")
		print(f"{'='*60}")
		print("  Options:")
		print("  1. Adjust Config.py to match an existing checkpoint")
		print("  2. Set strict_config=False to load best checkpoint regardless of config")
		print("  3. Start training from scratch (resume_training=False)")
		
		if non_matching_checkpoints:
			print(f"\n{'='*60}")
			print("NON-MATCHING CHECKPOINTS (not loaded in strict mode)")
			print(f"{'='*60}")
			non_matching_checkpoints.sort(key=lambda x: x[1])
			for i, (path, loss, cfg) in enumerate(non_matching_checkpoints[:5], 1):
				print(f"  {i}. {os.path.basename(path)} (loss: {loss:.4f})")
				if cfg:
					# Show what doesn't match
					for param in critical_params:
						if param in cfg and param in current_config:
							if cfg[param] != current_config[param]:
								print(f"  -> {param}: checkpoint={cfg[param]}, current={current_config[param]}")
			print(f"{'='*60}")
		
		return None, False, ["No config-matching checkpoints found"]
	
	# Non-strict mode: return best checkpoint regardless of config
	if non_matching_checkpoints:
		non_matching_checkpoints.sort(key=lambda x: x[1])
		best_checkpoint, best_loss, best_config = non_matching_checkpoints[0]
		
		warnings = [f"Loading checkpoint with config mismatch (strict_config=False)"]
		
		print(f"\n{'='*60}")
		print("WARNING: Loading best checkpoint despite config mismatch")
		print(f"{'='*60}")
		print(f"Best checkpoint: {os.path.basename(best_checkpoint)} (loss: {best_loss:.4f})")
		
		# Show config differences
		if best_config:
			print("\nCONFIG DIFFERENCES:")
			for param in critical_params:
				if param in best_config and param in current_config:
					if best_config[param] != current_config[param]:
						print(f"  {param}: checkpoint={best_config[param]}, current={current_config[param]}")
		print(f"{'='*60}")
		
		return best_checkpoint, False, warnings
	
	print("\n ERROR: No valid checkpoints found at all!")
	return None, False, ["No valid checkpoints found"]


def validate_config(checkpoint_config, current_config):
	"""
	Validate that checkpoint config matches current config.
    
	Args:
		checkpoint_config (dict): Config saved in checkpoint
		current_config (dict): Current model configuration
        
	Returns:
		tuple: (is_valid, warnings, errors)
	"""
	if checkpoint_config is None:
		return True, ["No config saved in checkpoint - cannot validate"], []
    
	warnings = []
	errors = []
    
	# Critical parameters that MUST match
	critical_params = ['d_model', 'n_heads', 'n_layers', 'ffn_hidden']
    
	# Parameters that should match but won't break loading
	warning_params = ['drop_prob', 'batch_size', 'max_len', 'mode']
    
	for param in critical_params:
		if param in checkpoint_config and param in current_config:
			if checkpoint_config[param] != current_config[param]:
				errors.append(
					f"CRITICAL: {param} mismatch! "
					f"Checkpoint: {checkpoint_config[param]}, "
					f"Current: {current_config[param]}"
					)
    
	for param in warning_params:
		if param in checkpoint_config and param in current_config:
			if checkpoint_config[param] != current_config[param]:
				warnings.append(
					f"WARNING: {param} changed. "
					f"Checkpoint: {checkpoint_config[param]}, "
					f"Current: {current_config[param]}"
					)
    
	is_valid = len(errors) == 0
	return is_valid, warnings, errors



def load_checkpoint(model, checkpoint_path, optimizer=None, lr_scheduler=None, device=None, current_config=None, strict_config=True):
	"""
	Load model weights from checkpoint, optionally load optimizer and scheduler state.
    
	Args:
		model (nn.Module): The model to load weights into
		checkpoint_path (str): Path to the checkpoint file
		optimizer (Optimizer, optional): Optimizer to load state into
		lr_scheduler (Scheduler, optional): Learning rate scheduler to load state into
		device (torch.device, optional): Device to load the checkpoint on
        
	Returns:
		tuple: (start_epoch, best_loss, train_losses, test_losses, bleus, src_vocab, trg_vocab)
	"""
	if not os.path.exists(checkpoint_path):
		print(f"Checkpoint not found: {checkpoint_path}")
		return 0, float('inf'), [], [], []
    
	print(f"\nLoading checkpoint from: {checkpoint_path}")
    
	# Load the state dict
	if device:
		checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
	else:
		checkpoint = torch.load(checkpoint_path, weights_only=False)
    
	# Initialize return values
	start_epoch = 0
	best_loss = float('inf')
	train_losses = []
	test_losses = []
	bleus = []
	src_vocab = None
	trg_vocab = None
    
	# Check if this is an enhanced checkpoint or just model weights
	if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:


		# Validate configuration if provided
		if current_config is not None:
			checkpoint_config = checkpoint.get('config', None)
			is_valid, warnings, errors = validate_config(checkpoint_config, current_config)
            
			# Print warnings
			if warnings:
				print("\n Configuration Warnings:")
				for warning in warnings:
					print(f"  {warning}")
            
			# Handle errors
			if errors:
				print("\n Configuration Errors:")
				for error in errors:
					print(f"  {error}")
                
				if strict_config:
					print("\n ABORTING: Config mismatch detected!")
					print("  Options:")
					print("  1. Change your Config.py to match the checkpoint")
					print("  2. Set resume_training=False to start fresh")
					print("  3. Set strict_config=False to load anyway (risky!)")
					return 0, float('inf'), [], [], []
				else:
					print("\n WARNING: Loading anyway (strict_config=False)")
					print("  This may cause crashes or incorrect results!")
            
			if not warnings and not errors and checkpoint_config:
				print("Configuration validated successfully")
        
		# Enhanced checkpoint with full training state
		try:
			# Enhanced checkpoint with full training state
			model.load_state_dict(checkpoint['model_state_dict'])
			print("Loaded model weights")
		except RuntimeError as e:
			print(f"\n ERROR loading model weights: {e}")
			print("  The checkpoint model architecture doesn't match current model!")
			return 0, float('inf'), [], [], []
			
        
		if optimizer and 'optimizer_state_dict' in checkpoint:
			try:
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
				print("Loaded optimizer state")
			except Exception as e:
				print(f"Warning: Could not load optimizer state: {e}")
				
        
		if lr_scheduler and 'scheduler_state_dict' in checkpoint:
			try:
				lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
				print("Loaded learning rate scheduler state")
			except Exception as e:
				print(f"Warning: Could not load scheduler state: {e}")
        
		start_epoch = checkpoint.get('epoch', 0)
		best_loss = checkpoint.get('best_loss', float('inf'))
		train_losses = checkpoint.get('train_losses', [])
		test_losses = checkpoint.get('test_losses', [])
		bleus = checkpoint.get('bleus', [])


		# Load vocabularies if present
		src_vocab = checkpoint.get('src_vocab', None)
		trg_vocab = checkpoint.get('trg_vocab', None)
		
		if src_vocab is not None and trg_vocab is not None:
			print("Loaded vocabularies from checkpoint")
			print(f"  Source vocab size: {len(src_vocab)}")
			print(f"  Target vocab size: {len(trg_vocab)}")
		else:
			print("Note: No vocabularies found in checkpoint (old format)")
        
		print(f"Resumed from epoch {start_epoch}")
		print(f"Best loss so far: {best_loss:.4f}")
		if bleus:
			print(f"Best BLEU so far: {max(bleus):.4f}")
        
	else:
		try:
			# Simple checkpoint with only model weights
			model.load_state_dict(checkpoint)
			print("Loaded model weights (simple checkpoint)")
			print("Note: Optimizer and training history not available")
			print("  Note: Cannot validate config (old checkpoint format)")
		except RuntimeError as e:
			print(f"\n ERROR loading model weights: {e}")
			print("  The checkpoint model architecture doesn't match current model!")
			return 0, float('inf'), [], [], []
			
    
	return start_epoch, best_loss, train_losses, test_losses, bleus, src_vocab, trg_vocab


def save_checkpoint(model, optimizer, lr_scheduler, epoch, best_loss, train_loss, val_loss, bleu, train_losses, test_losses, bleus, config=None, src_vocab=None, trg_vocab=None, checkpoint_dir='saved', save_latest=True):
	"""
	Save a checkpoint with full training state.
    
	Args:
		model (nn.Module): The model to save
		optimizer (Optimizer): The optimizer state to save
		lr_scheduler (Scheduler): The learning rate scheduler state to save
		epoch (int): Current epoch number
		best_loss (float): Best validation loss achieved so far
		train_loss (float): Current training loss
		val_loss (float): Current validation loss
		bleu (float): Current BLEU score
		train_losses (list): List of all training losses
		test_losses (list): List of all validation losses
		bleus (list): List of all BLEU scores
		config (dict, optional): Model configuration to save
		src_vocab (Vocab, optional): Source vocabulary to save
		trg_vocab (Vocab, optional): Target vocabulary to save
		checkpoint_dir (str): Directory to save checkpoints
		save_latest (bool): Whether to also save as 'latest' checkpoint
        
	Returns:
		str: Path to the saved checkpoint
	"""
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
		print(f"Created checkpoint directory: {checkpoint_dir}")
    
	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None, 
		'best_loss': best_loss,
		'train_loss': train_loss,
		'val_loss': val_loss,
		'bleu': bleu,
		'train_losses': train_losses,
		'test_losses': test_losses,
		'bleus': bleus,
		'config': config, 
		'src_vocab': src_vocab,
		'trg_vocab': trg_vocab
		}
    
	# Save with validation loss in filename
	checkpoint_path = os.path.join(checkpoint_dir, f'model-epoch{epoch}-loss{val_loss:.4f}.pt')
	torch.save(checkpoint, checkpoint_path)

	# Also save as latest
	if save_latest:
		latest_path = os.path.join(checkpoint_dir, 'model-latest.pt')
		torch.save(checkpoint, latest_path)

	return checkpoint_path


def load_training_history(checkpoint_dir='saved'):
	"""
	Load training history from text files (legacy support).
    
	Args:
		checkpoint_dir (str): Directory containing the history files
        
	Returns:
		tuple: (train_losses, test_losses, bleus)
	"""
	train_losses = []
	test_losses = []
	bleus = []
    
	train_loss_file = os.path.join(checkpoint_dir, 'train_loss.txt')
	test_loss_file = os.path.join(checkpoint_dir, 'test_loss.txt')
	bleu_file = os.path.join(checkpoint_dir, 'bleu.txt')
    
	if os.path.exists(train_loss_file):
		with open(train_loss_file, 'r') as f:
			try:
				train_losses = eval(f.read())
				print(f"Loaded {len(train_losses)} training loss values")
			except:
				print("Warning: Could not load training loss history")
    
	if os.path.exists(test_loss_file):
		with open(test_loss_file, 'r') as f:
			try:
				test_losses = eval(f.read())
				print(f"Loaded {len(test_losses)} validation loss values")
			except:
				print("Warning: Could not load validation loss history")
    
	if os.path.exists(bleu_file):
		with open(bleu_file, 'r') as f:
			try:
				bleus = eval(f.read())
				print(f"Loaded {len(bleus)} BLEU score values")
			except:
				print("Warning: Could not load BLEU history")
    
	return train_losses, test_losses, bleus


def save_training_history(train_losses, test_losses, bleus, checkpoint_dir='saved'):
	"""
	Save training history to text files (legacy support).
    
	Args:
		train_losses (list): List of training losses
		test_losses (list): List of validation losses
		bleus (list): List of BLEU scores
		checkpoint_dir (str): Directory to save the history files
	"""
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
    
	with open(os.path.join(checkpoint_dir, 'train_loss.txt'), 'w') as f:
		f.write(str(train_losses))
    
	with open(os.path.join(checkpoint_dir, 'test_loss.txt'), 'w') as f:
		f.write(str(test_losses))
    
	with open(os.path.join(checkpoint_dir, 'bleu.txt'), 'w') as f:
		f.write(str(bleus))


def cleanup_old_checkpoints(checkpoint_dir='saved', keep_best_n=5):
	"""
	Remove old checkpoints, keeping only the best N.
    
	Args:
		checkpoint_dir (str): Directory containing checkpoints
		keep_best_n (int): Number of best checkpoints to keep
	"""
	checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model-*.pt'))
    
	# Don't delete the latest checkpoint
	latest_path = os.path.join(checkpoint_dir, 'model-latest.pt')
	checkpoint_files = [f for f in checkpoint_files if f != latest_path]
    
	if len(checkpoint_files) <= keep_best_n:
		return
    
	# Extract loss values and sort
	checkpoints_with_loss = []
	for filepath in checkpoint_files:
		try:
			filename = os.path.basename(filepath)
			loss_str = filename.split('-loss')[-1].replace('.pt', '')
			loss_value = float(loss_str)
			checkpoints_with_loss.append((filepath, loss_value))
		except (ValueError, IndexError):
			print(f"Skipping invalid checkpoint filename: {filepath}")
			continue
    
	# Sort by loss (best first)
	checkpoints_with_loss.sort(key=lambda x: x[1])
    
	# Delete checkpoints beyond keep_best_n
	for filepath, loss in checkpoints_with_loss[keep_best_n:]:
		try:
			os.remove(filepath)
			print(f"Removed old checkpoint: {os.path.basename(filepath)}")
		except:
			print(f"Warning: Could not remove {filepath}")


def print_checkpoint_summary(checkpoint_dir='saved'):
	"""
		Print a summary of available checkpoints.
    
		Args:
			checkpoint_dir (str): Directory containing checkpoints
	"""
	if not os.path.exists(checkpoint_dir):
		print(f"No checkpoint directory found at {checkpoint_dir}")
		return
    
	checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model-*.pt'))
    
	if not checkpoint_files:
		print("No checkpoints found")
		return
    
	print("\n" + "="*60)
	print("CHECKPOINT SUMMARY")
	print("="*60)
    
	checkpoints_with_loss = []
	for filepath in checkpoint_files:
		try:
			filename = os.path.basename(filepath)
			if filename == 'model-latest.pt':
				continue
			loss_str = filename.split('-loss')[-1].replace('.pt', '')
			loss_value = float(loss_str)
			checkpoints_with_loss.append((filepath, loss_value))
		except (ValueError, IndexError):
			print(f"Skipping invalid checkpoint filename: {filepath}")
			continue
    
	checkpoints_with_loss.sort(key=lambda x: x[1])
    
	print(f"Total checkpoints: {len(checkpoints_with_loss)}")
	print("\nBest checkpoints:")
	for i, (path, loss) in enumerate(checkpoints_with_loss[:10], 1):
		print(f"  {i}. {os.path.basename(path)} - Loss: {loss:.4f}")
    
	latest_path = os.path.join(checkpoint_dir, 'model-latest.pt')
	if os.path.exists(latest_path):
		print(f"\nLatest checkpoint available: model-latest.pt")
    
	print("="*60 + "\n")

