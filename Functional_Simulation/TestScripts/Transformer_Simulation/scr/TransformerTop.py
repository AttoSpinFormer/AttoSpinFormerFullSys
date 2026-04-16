

#!/usr/bin/env python3
"""
###############################################################################
#Related Paper:	 AttoSpinFormer: An Energy-efficient Magneto Electric Spin Orbit Logic-based Compute-in-Memory Transformer Architecture
# Module:        TransformerTop.py
# Description:   Verification script for a **language model** mapped onto the MESO IMC architecture.
#
# Synopsis:      This module performs functional validation and quantitative error
#                analysis of the MESO IMC pipeline specifically targeting **Transformer** 
#                workloads. The architecture utilizes the MESO IMC engine for 
#                the critical matrix multiplications within the attention mechanism: 
#                specifically, between Queries and Keys, and between Softmax outputs and Values.
#
# Created:       2025-11-11
# Last Modified: 2026-01-21
###############################################################################

Execution Command:
    
    This script is executed directly from the terminal:
    python3 TransformerTop.py 

Note: The module evaluates the fidelity and performance impact of mapping **Transformer attention layers** onto a 
      MESO IMC infrastructure (128 x 16 MESO arrays).
"""

import os
import torch
import math
import time
import sacrebleu
import re

from torch import nn,optim
from torch.optim import Adam

from TransformerBasic import Transformer
from bleu import idx_to_word,get_bleu
#from epoch_timer import epoch_timer
from CheckPointManager import save_checkpoint, save_training_history, cleanup_old_checkpoints
from DatasetLoads import SRC, TRG
from ModelSetupManager import setup_model_and_data, TransformerLRScheduler



# ============================================================================
# Config sanity check functions 
# ============================================================================

from ConfigSanityCheck import get_validated_config

# Get validated configuration (auto-corrects invalid values)
validated = get_validated_config(verbose=True)

# Unpack all validated parameters
DEVICE = validated['DEVICE']
device = validated['device']
batch_size = validated['batch_size']
max_len = validated['max_len']
d_model = validated['d_model']
n_layers = validated['n_layers']
n_heads = validated['n_heads']
ffn_hidden = validated['ffn_hidden']
drop_prob = validated['drop_prob']
init_lr = validated['init_lr']
factor = validated['factor']
adam_eps = validated['adam_eps']
patience = validated['patience']
warmup = validated['warmup']
epoch = validated['epoch']
clip = validated['clip']
weight_decay = validated['weight_decay']
label_smoothing = validated['label_smoothing']
mode = validated['mode']
bit_width = validated['bit_width']
cycle_res = validated['cycle_res']
var = validated['var']
resume_training = validated['resume_training']
strict_config = validated['strict_config']
specials = validated['specials']
inf = validated['inf']


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Setup everything using ModelSetupManager
# ============================================================================
config = {
	'resume_training': resume_training,
	'strict_config': strict_config,
	'd_model': d_model,
	'n_heads': n_heads,
  	'n_layers': n_layers,
	'ffn_hidden': ffn_hidden,
	'drop_prob': drop_prob,
	'batch_size': batch_size,
	'max_len': max_len,
	'mode': mode,
	'bit_width': bit_width,
	'cycle_res': cycle_res, 
	'var' : var,
	'init_lr': init_lr,
	'weight_decay': weight_decay,
	'adam_eps': adam_eps,
	'warmup': warmup,
	'label_smoothing': label_smoothing,
	'factor': factor,
	'patience': patience,
	'inf': inf
	}

# This single call handles all setup steps 1-5
setup = setup_model_and_data(config)

# Unpack everything we need
model = setup['model']
optimizer = setup['optimizer']
lr_scheduler = setup['lr_scheduler']
scheduler = setup['scheduler']
criterion = setup['criterion']
train_iter = setup['train_iter']
valid_iter = setup['valid_iter']
test_iter = setup['test_iter']
vocab_de = setup['vocab_de']
vocab_en = setup['vocab_en']
src_pad_idx = setup['src_pad_idx']
trg_pad_idx = setup['trg_pad_idx']
trg_sos_idx = setup['trg_sos_idx']
enc_voc_size = setup['enc_voc_size']
dec_voc_size = setup['dec_voc_size']
start_epoch = setup['start_epoch']
best_loss = setup['best_loss']
train_losses = setup['train_losses']
test_losses = setup['test_losses']
bleus = setup['bleus']

# ============================================================================
# Training functions 
# ============================================================================

def train(model, iterator, optimizer, criterion, clip):
	model.train()
	epoch_loss = 0
	for i, batch in enumerate(iterator):
		src = batch.src
		trg = batch.trg
		optimizer.zero_grad()
		output = model(src, trg[:, :-1])
		output_reshape = output.contiguous().view(-1, output.shape[-1])
		trg_flat = trg[:, 1:].contiguous().view(-1)

		loss = criterion(output_reshape, trg_flat)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()

		if lr_scheduler is not None and lr_scheduler.step_num <warmup :
			lr_scheduler.step()
			
		current_lr=optimizer.param_groups[0]['lr']
		epoch_loss += loss.item()
		print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
		print(f"LR:{current_lr}")

	return epoch_loss / len(iterator)



# ============================================================================
# Loss and BLEU score evaluation functions 
# ============================================================================

def detokenize(text):
	"""Clean up tokenized text for BLEU evaluation."""
	text = re.sub(r'\s+([,.!?;:])', r'\1', text)
	text = re.sub(r"\s+'\s+", r"'", text) # Fixes contractions like "don ' t"
	return text



def evaluate_inference(model, iterator, criterion):
	"""Evaluate model using inference mode (autoregressive generation)."""
	model.eval()
	epoch_loss = 0
	all_hypotheses = []
	all_references = []

	total_tokens = 0
	pad_tokens = 0
	sos_tokens = 0
	eos_tokens = 0
	unk_tokens = 0
    
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			src = batch.src
			trg = batch.trg 
            
			
			output_loss = model(src, trg[:, :-1])
			output_reshape = output_loss.contiguous().view(-1, output_loss.shape[-1])
			trg_flat = trg[:, 1:].contiguous().view(-1)
			loss = criterion(output_reshape, trg_flat)
			epoch_loss += loss.item()

			batch_size = src.shape[0]
			decoded_trg = torch.full((batch_size, 1), trg_sos_idx, dtype=torch.long, device=DEVICE)
            
			for _ in range(max_len - 1):
				output = model(src, decoded_trg) # [batch_size, current_len, voc_size]
				next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1) # [batch_size, 1]
				decoded_trg = torch.cat((decoded_trg, next_token), dim=1)

			for j in range(batch_size):
				actual_trg_indices = trg[j, 1:]
				trg_words = idx_to_word(actual_trg_indices, TRG.vocab).split()
				trg_sentence = idx_to_word(actual_trg_indices, TRG.vocab)
				generated_indices = decoded_trg[j, 1:] 
				eos_idx = TRG.vocab.stoi['<eos>']
				indices_list = []
				for idx in generated_indices:
					val = idx.item()
					indices_list.append(val)
					total_tokens += 1
					if val == trg_pad_idx: pad_tokens += 1
					elif val == trg_sos_idx: sos_tokens += 1
					elif val == eos_idx: 
						eos_tokens += 1
						break # Stop collecting words for this sentence after <eos>
					elif val == TRG.vocab.stoi['<unk>']: unk_tokens += 1

				output_sentence = idx_to_word(torch.tensor(indices_list), TRG.vocab)
                
				all_hypotheses.append(output_sentence)
				all_references.append(trg_sentence)

	if total_tokens > 0:
		print(f"\n=== Token Statistics (Inference Mode) ===")
		print(f"Total tokens generated: {total_tokens}")
		print(f"Padding tokens: {pad_tokens} ({100*pad_tokens/total_tokens:.2f}%)")
		print(f"SOS tokens: {sos_tokens} ({100*sos_tokens/total_tokens:.2f}%)")
		print(f"EOS tokens: {eos_tokens} ({100*eos_tokens/total_tokens:.2f}%)")
		print(f"UNK tokens: {unk_tokens} ({100*unk_tokens/total_tokens:.2f}%)")
		print(f"Real tokens: {total_tokens - pad_tokens - sos_tokens - eos_tokens - unk_tokens} ({100*(total_tokens - pad_tokens - sos_tokens - eos_tokens - unk_tokens)/total_tokens:.2f}%)")
		print("========================\n")

	if not all_hypotheses:
		return epoch_loss / len(iterator), 0.0


	detokenized_hypotheses = [detokenize(h) for h in all_hypotheses]
	detokenized_references = [detokenize(r) for r in all_references]

	bleu_result = sacrebleu.corpus_bleu(detokenized_hypotheses, [detokenized_references])
	total_bleu = bleu_result.score

	return epoch_loss / len(iterator), total_bleu


# ============================================================================
# Debug functions 
# ============================================================================


def inspect_outputs(model, iterator, num_samples=5):
	"""Print sample translations to see what the model is generating"""
	model.eval()
    
	with torch.no_grad():
		batch = next(iter(iterator))
        
		# DEBUG: Print what's actually in the batch
		print("\n=== DEBUGGING BATCH STRUCTURE ===")
		print(f"Batch attributes: {dir(batch)}")
		print(f"batch.src shape: {batch.src.shape}")
		print(f"batch.trg shape: {batch.trg.shape}")
		print("================================\n")


		src = batch.src
		trg = batch.trg
        
		output = model(src, trg[:, :-1])
        
		print("\n=== Sample Translations ===")
		for i in range(min(num_samples, src.shape[0])):
			# Source sentence
			src_words = idx_to_word(src[i], SRC.vocab)
			print(f"\nSource (DE): {src_words}")
            
			# Target sentence
			actual_trg_indices = trg[i, 1:] 
			trg_words = idx_to_word(actual_trg_indices, TRG.vocab)
			print(f"Target (EN): {trg_words}")
            
			# Predicted sentence
			output_indices = output[i].max(dim=1)[1]
			pred_words = idx_to_word(output_indices, TRG.vocab)
			print(f"Predicted:   {pred_words}")
            
			# Raw indices (to see if it's just padding)
			print(f"Raw indices: {output_indices[:20].tolist()}...")  # First 20 tokens
			print(f"Pad index: {trg_pad_idx}, SOS: {trg_sos_idx}, EOS: {TRG.vocab.stoi['<eos>']}")
        
		print("===========================\n")


# ============================================================================
# Top function that runs the simulation
# ============================================================================

def run(total_epoch, best_loss, start_epoch=0, train_losses=None, test_losses=None, bleus=None):
	if not os.path.exists('saved'):
		os.makedirs('saved')
	
	if train_losses is None:
		train_losses=[]
	if test_losses is None:
		test_losses=[]
	if bleus is None:
		bleus=[]


	print(f"\nStarting training from epoch {start_epoch + 1} to {total_epoch}")
	print(f"Current best loss: {best_loss:.4f}")
	if bleus:
		print(f"Current best BLEU: {max(bleus):.4f}\n")

	print("\n"+"-"*60+"\n")

	if mode == 0:
		print(f"Chosen mode for current run {mode}: Typical CMOS Execution (GPU Reference).")
	else:
		print(f"Chosen mode for current run {mode}: MESO IMC Execution (In-Memory Computing).")
		print(f"Chosen bit-width for current run: {bit_width} bits.")
		print(f"Chosen per cycle resolution of the input applied through the CMOS gate for current run: {cycle_res} bits.")
		print(f"Chosen device variability for current run: {var*100}%.")

	print("\n"+"-"*60+"\n")

	for step in range(start_epoch, total_epoch):
		start_time = time.time()
		train_loss = train(model, train_iter, optimizer, criterion, clip)
		valid_loss, bleu = evaluate_inference(model, valid_iter, criterion)
		end_time = time.time()

		if step % 5 == 0:  # Every 5 epochs
			inspect_outputs(model, valid_iter, num_samples=3)

		if lr_scheduler.step_num >= warmup:
			scheduler.step(valid_loss)

		train_losses.append(train_loss)
		test_losses.append(valid_loss)
		bleus.append(bleu)
		epoch_secs=end_time-start_time
		epoch_mins=float(epoch_secs)/60

		if valid_loss < best_loss:
			best_loss = valid_loss

			# Save enhanced checkpoint with full state
			config_dict = {
				'd_model': d_model,
				'n_heads': n_heads,
				'n_layers': n_layers,
				'ffn_hidden': ffn_hidden,
				'drop_prob': drop_prob,
				'batch_size': batch_size,
				'max_len': max_len,
				'mode': mode
				}
            
			checkpoint_path = save_checkpoint(
				model=model,
				optimizer=optimizer,
				lr_scheduler=lr_scheduler,
				epoch=step + 1,
				best_loss=best_loss,
				train_loss=train_loss,
				val_loss=valid_loss,
				bleu=bleu,
				train_losses=train_losses,
				test_losses=test_losses,
				bleus=bleus,
				config=config_dict,
				src_vocab=vocab_de, 
				trg_vocab=vocab_en,
				checkpoint_dir='saved',
				save_latest=True
				)
			print(f"*** New best model saved! ***")
			print(f"    Checkpoint: {os.path.basename(checkpoint_path)}")
			print(f"    Loss: {best_loss:.4f}, BLEU: {bleu:.4f}")


		# Also save training history to text files (for backward compatibility)
		save_training_history(train_losses, test_losses, bleus, 'saved')
        
		# Clean up old checkpoints (keep only best 5)
		if (step + 1) % 10 == 0:  # Every 10 epochs
			cleanup_old_checkpoints('saved', keep_best_n=5)


		print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
		print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
		print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
	run(total_epoch=epoch, best_loss=best_loss,start_epoch=start_epoch,train_losses=train_losses,test_losses=test_losses,bleus=bleus)
	print("\n"+"-"*60+"\n")
	print("Testing the model on Test dataset\n")
	test_loss, bleu = evaluate_inference(model, test_iter, criterion)
	print(f'\tTest Loss: {test_loss:.3f} |  Test PPL: {math.exp(test_loss):7.3f}')
	print(f'\tBLEU Score: {bleu:.3f}')
	print("\n"+"-"*60+"\n")



