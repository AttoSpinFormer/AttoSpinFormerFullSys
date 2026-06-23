
#!/usr/bin/env python3
"""
###############################################################################
# Module:        bleu.py
# Description:   Implementation of the BLEU (Bilingual Evaluation Understudy) metric for text generation quality assessment.
#
# Synopsis:      This module provides the necessary functions to calculate the aggregated 
#                BLEU score (up to n-gram order 4) across multiple hypothesis-reference pairs. 
#                It utilizes numpy for efficient statistic aggregation and standard Python 
#                math libraries for the final geometric mean calculation.
#
###############################################################################
"""

import math
from collections import Counter

import numpy as np
import torch



# In bleu.py
def idx_to_word(x, vocab):
	words = []
	# Explicitly list tokens to ignore
	#ignore_tokens = {vocab.stoi['<sos>'], vocab.stoi['<eos>'], vocab.stoi['<pad>'], vocab.stoi['<unk>']}

	#using the following because i want the model to be penalized for unknown tokens too. 
	#If you dont want this, uncomment the command above this line and comment out the one below this line
	ignore_tokens = {vocab.stoi['<sos>'], vocab.stoi['<eos>'], vocab.stoi['<pad>']}
    
	for i in x:
		idx = i.item() if torch.is_tensor(i) else i
		if idx not in ignore_tokens:
			words.append(vocab.itos[idx])
            
	return " ".join(words)


def bleu_stats(hypothesis, reference):
	stats = []
	stats.append(len(hypothesis))
	stats.append(len(reference))
	for n in range(1, 5):
		s_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)])
		r_ngrams = Counter([tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)])

		stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
		stats.append(max([len(hypothesis) + 1 - n, 0]))
	return stats


def bleu_old(stats):
	if len(list(filter(lambda x: x == 0, stats))) > 0:
		return 0
	(c, r) = stats[:2]
	log_bleu_prec = sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
	return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def bleu(stats):

	if stats[0] == 0: return 0  # If hypothesis is empty
    
	(c, r) = stats[:2]
	log_bleu_prec = 0
	for i in range(2, 10, 2):
		matched = stats[i]
		total = stats[i+1]

		if matched == 0:
			log_bleu_prec += math.log(0.1 / total) if total > 0 else 0
		else:
			log_bleu_prec += math.log(float(matched) / total)
            
	log_bleu_prec /= 4.
	return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
	stats = np.zeros(10)
	for hyp, ref in zip(hypotheses, reference):
		stats += np.array(bleu_stats(hyp, ref))
	return 100 * bleu(stats)
