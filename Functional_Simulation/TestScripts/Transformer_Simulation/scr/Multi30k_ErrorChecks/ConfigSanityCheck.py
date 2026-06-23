#!/usr/bin/env python3
"""
###############################################################################
# Module:	   ConfigSanityCheck.py
# Description:     Validation and correction module for Config.py parameters
###############################################################################

Usage:
	from ConfigSanityCheck import get_validated_config
	
	config = get_validated_config()
	# Now use config['batch_size'], config['d_model'], etc.
	
This module is imported by TransformerTop.py and automatically validates
all configuration parameters, correcting invalid values to safe defaults.
"""

import torch

# Default safe values (matching original Config.py)
DEFAULTS = {
	'batch_size': 64,
	'max_len': 128,
	'd_model': 512,
	'n_layers': 6,
	'n_heads': 8,
	'ffn_hidden': 2048,
	'drop_prob': 0.1,
	'init_lr': 1e-4,
	'factor': 0.1,
	'adam_eps': 1e-9,
	'patience': 3,
	'warmup': 4000,
	'epoch': 50,
	'clip': 1.0,
	'weight_decay': 0.0001,
	'label_smoothing': 0.1,
	'mode': 1,
	'bit_width': 8,
	'cycle_res': 1,
	'var': 0.0,
	'resume_training': True,
	'strict_config': True,
}

class ConfigValidator:
	"""Validates and corrects configuration parameters"""
	
	def __init__(self, verbose=True):
		self.corrections = []
		self.warnings = []
		self.validated_config = {}
		self.verbose = verbose
		
	def _print(self, *args, **kwargs):
		"""Print only if verbose mode is on"""
		if self.verbose:
			print(*args, **kwargs)
	
	def validate_and_fix(self, param_name, value, validator_func, default_value):
		"""
		Validate a parameter and fix if invalid
		
		Args:
			param_name: Name of the parameter
			value: Current value
			validator_func: Function that returns (is_valid, message)
			default_value: Default value to use if invalid
		
		Returns:
			Corrected value
		"""
		is_valid, message = validator_func(value)
		
		if is_valid:
			if message:  # Warning message
				self.warnings.append(f"{param_name}: {message}")
				self._print(f"⚠ {param_name:20s} = {value:15s} - {message}")
			return value
		else:
			self.corrections.append(f"{param_name}: {value} -> {default_value} ({message})")
			self._print(f" {param_name:20s} = {value} -> {default_value} - {message}")
			return default_value
	
	def validate_positive_int(self, value, min_val=1):
		"""Validate positive integer"""
		if not isinstance(value, int) or value < min_val:
			return False, f"Must be integer >= {min_val}"
		return True, ""
	
	def validate_positive_float(self, value):
		"""Validate positive float"""
		if not isinstance(value, (int, float)) or value <= 0:
			return False, "Must be positive number"
		return True, ""
	
	def validate_range(self, value, min_val, max_val, inclusive_min=True, inclusive_max=True):
		"""Validate value (scalar or 1D tensor) is in range [min_val, max_val] with inclusivity options."""

		# Convert to torch.Tensor (stays on same device if already tensor)
		if torch.is_tensor(value):
			t = value
		elif isinstance(value, (int, float)):
			t = torch.tensor(value)
		elif isinstance(value, (list, tuple)):
			t = torch.tensor(value)
		else:
			return False, "Must be a torch Tensor, a number, or a 1D list/tuple of numbers"

		# Ensure numeric dtype (reject bool and non-numeric)
		if t.dtype == torch.bool:
			return False, "Must be numeric (not bool)"
		if not (torch.is_floating_point(t) or t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
			return False, "Must be a numeric tensor"

		# Ensure scalar (0-d) or 1D
		if t.ndim > 1:
			return False, "Must be a scalar or 1D tensor"

		# Build range check
		if inclusive_min:
			lower_ok = t >= min_val
		else:
			lower_ok = t > min_val

		if inclusive_max:
			upper_ok = t <= max_val
		else:
			upper_ok = t < max_val

		ok = lower_ok & upper_ok

		# ok can be a scalar bool tensor or 1D bool tensor
		if not torch.all(ok).item():
			# Optional: give a more informative message
			# Find first failing element (works for scalar and 1D)
			if t.ndim == 0:
				return False, f"Must be in the specified range"
			bad_idx = torch.nonzero(~ok, as_tuple=False).flatten()
			i0 = bad_idx[0].item()
			return False, f"Element {i0} (value={t[i0].item()}) is out of range"

		# If you want to keep exact bracket text like before:
		if inclusive_min and inclusive_max:
			msg = f"Must be in [{min_val}, {max_val}]"
		elif inclusive_min and not inclusive_max:
			msg = f"Must be in [{min_val}, {max_val})"
		elif not inclusive_min and inclusive_max:
			msg = f"Must be in ({min_val}, {max_val}]"
		else:
			msg = f"Must be in ({min_val}, {max_val})"

		return True, ""	


	def run_validation(self):
		"""Main validation function"""
		
		try:
			import Config
			self._print("\n" + "="*70)
			self._print("  CONFIGURATION VALIDATION")
			self._print("="*70)
		except ImportError as e:
			self._print(f"\n ERROR: Could not import Config.py: {e}")
			self._print("Using all default values...")
			return DEFAULTS.copy()

		self.validated_config['DEVICE'] = Config.device
		self.validated_config['device'] = Config.device
		
		# ====================================================================
		# MODEL PARAMETERS
		# ====================================================================
		
		# Batch size
		self.validated_config['batch_size'] = self.validate_and_fix(
			'batch_size', Config.batch_size,
			lambda v: self.validate_positive_int(v, 1),
			DEFAULTS['batch_size']
		)
		
		# Max length
		self.validated_config['max_len'] = self.validate_and_fix(
			'max_len', Config.max_len,
			lambda v: self.validate_positive_int(v, 80),
			DEFAULTS['max_len']
		)
		
		# Number of heads (validate this first, before d_model)
		self.validated_config['n_heads'] = self.validate_and_fix(
			'n_heads', Config.n_heads,
			lambda v: self.validate_positive_int(v, 1),
			DEFAULTS['n_heads']
		)
		
		# Model dimension (must be divisible by n_heads)
		def validate_d_model(v):
			valid, msg = self.validate_positive_int(v, 1)
			if not valid:
				return False, msg
			if v % self.validated_config['n_heads'] != 0:
				return False, f"Must be divisible by n_heads ({self.validated_config['n_heads']})"
			return True, ""
		
		self.validated_config['d_model'] = self.validate_and_fix(
			'd_model', Config.d_model,
			validate_d_model,
			DEFAULTS['d_model']
		)
		
		# Number of layers
		self.validated_config['n_layers'] = self.validate_and_fix(
			'n_layers', Config.n_layers,
			lambda v: self.validate_positive_int(v, 1),
			DEFAULTS['n_layers']
		)
		
		# FFN hidden dimension
		def validate_ffn(v):
			valid, msg=self.validate_positive_int(v,1)
			if not valid:
				return False, msg
			#if not v / self.validated_config['d_model'] != 4 or not v % self.validated_config['d_model'] == 0 :
			#	return False, f"Must be 4*d_model"
			return True, ""

		self.validated_config['ffn_hidden'] = self.validate_and_fix(
			'ffn_hidden', Config.ffn_hidden,
			validate_ffn,
			DEFAULTS['ffn_hidden']
		)
		
		# Dropout probability
		self.validated_config['drop_prob'] = self.validate_and_fix(
			'drop_prob', Config.drop_prob,
			lambda v: self.validate_range(v, 0, 1),
			DEFAULTS['drop_prob']
		)
		
		# ====================================================================
		# OPTIMIZER PARAMETERS
		# ====================================================================
		
		# Learning rate
		self.validated_config['init_lr'] = self.validate_and_fix(
			'init_lr', Config.init_lr,
			lambda v: self.validate_positive_float(v),
			DEFAULTS['init_lr']
		)
		
		# Factor
		self.validated_config['factor'] = self.validate_and_fix(
			'factor', Config.factor,
			lambda v: self.validate_range(v, 0, 1, inclusive_min=False),
			DEFAULTS['factor']
		)
		
		# Adam epsilon
		self.validated_config['adam_eps'] = self.validate_and_fix(
			'adam_eps', Config.adam_eps,
			lambda v: self.validate_positive_float(v),
			DEFAULTS['adam_eps']
		)
		
		# Patience
		self.validated_config['patience'] = self.validate_and_fix(
			'patience', Config.patience,
			lambda v: self.validate_positive_int(v, 0),
			DEFAULTS['patience']
		)
		
		# Warmup steps
		self.validated_config['warmup'] = self.validate_and_fix(
			'warmup', Config.warmup,
			lambda v: self.validate_positive_int(v, 0),
			DEFAULTS['warmup']
		)
		
		# Epochs
		self.validated_config['epoch'] = self.validate_and_fix(
			'epoch', Config.epoch,
			lambda v: self.validate_positive_int(v, 1),
			DEFAULTS['epoch']
		)
		
		# Gradient clipping
		self.validated_config['clip'] = self.validate_and_fix(
			'clip', Config.clip,
			lambda v: self.validate_positive_float(v),
			DEFAULTS['clip']
		)
		
		# Weight decay
		self.validated_config['weight_decay'] = self.validate_and_fix(
			'weight_decay', Config.weight_decay,
			lambda v: self.validate_range(v, 0, float('inf')),
			DEFAULTS['weight_decay']
		)
		
		# Label smoothing
		self.validated_config['label_smoothing'] = self.validate_and_fix(
			'label_smoothing', Config.label_smoothing,
			lambda v: self.validate_range(v, 0, 1, inclusive_max=False),
			DEFAULTS['label_smoothing']
		)
		
		# ====================================================================
		# HARDWARE/IMC PARAMETERS
		# ====================================================================
		
		# Mode
		def validate_mode(v):
			if v not in (0, 1):
				return False, "Must be 0 (CMOS) or 1 (IMC)"
			return True, ""
		
		self.validated_config['mode'] = self.validate_and_fix(
			'mode', Config.mode,
			validate_mode,
			DEFAULTS['mode']
		)
		
		# Bit width
		self.validated_config['bit_width'] = self.validate_and_fix(
			'bit_width', Config.bit_width,
			lambda v: self.validate_range(v, 2, 32),
			DEFAULTS['bit_width']
		)


		# Cycle resolution
		self.validated_config['cycle_res'] = self.validate_and_fix(
			'cycle_res', Config.cycle_res,
			lambda v: self.validate_range(v, 1, 7),
			DEFAULTS['cycle_res']
		)

		# device variability
		self.validated_config['var'] = self.validate_and_fix(
			'var', Config.var,
			lambda v: self.validate_range(v, 0, 1),
			DEFAULTS['var']
		)
		
		# ====================================================================
		# OTHER PARAMETERS
		# ====================================================================
		
		# Resume training
		self.validated_config['resume_training'] = Config.resume_training if isinstance(Config.resume_training, bool) else DEFAULTS['resume_training']
		
		# Strict config
		self.validated_config['strict_config'] = Config.strict_config if isinstance(Config.strict_config, bool) else DEFAULTS['strict_config']
		
		# Special tokens
		self.validated_config['specials'] = Config.specials if isinstance(Config.specials, list) else ["<unk>", "<pad>", "<sos>", "<eos>"]
		
		# Infinity
		self.validated_config['inf'] = float('inf')
		
		# Print summary
		if self.corrections:
			self._print(f"\n Applied {len(self.corrections)} correction(s):")
			for corr in self.corrections:
				self._print(f"    {corr}")
		
		if self.warnings:
			self._print(f"\n {len(self.warnings)} warning(s):")
			for warn in self.warnings:
				self._print(f"    {warn}")
		
		if not self.corrections and not self.warnings:
			self._print("\n All configuration parameters valid!")
		
		self._print("="*70 + "\n")
		
		return self.validated_config


def get_validated_config(verbose=True):
	"""
	Main function to get validated configuration
	
	Args:
		verbose: If True, print validation messages
	
	Returns:
		Dictionary with validated configuration parameters
	"""
	validator = ConfigValidator(verbose=verbose)
	return validator.run_validation()


# For backward compatibility and standalone testing
if __name__ == "__main__":
	print("\n" + "="*70)
	print("  CONFIG VALIDATION TEST")
	print("="*70)
	
	config = get_validated_config(verbose=True)
	
	print("\n" + "="*70)
	print("  VALIDATED CONFIGURATION")
	print("="*70)
	print("\nModel Parameters:")
	print(f"  batch_size  = {config['batch_size']}")
	print(f"  max_len	 = {config['max_len']}")
	print(f"  d_model	 = {config['d_model']}")
	print(f"  n_heads	 = {config['n_heads']}")
	print(f"  n_layers	= {config['n_layers']}")
	print(f"  ffn_hidden  = {config['ffn_hidden']}")
	print(f"  drop_prob   = {config['drop_prob']}")
	
	print("\nOptimizer Parameters:")
	print(f"  init_lr	 = {config['init_lr']}")
	print(f"  epoch	   = {config['epoch']}")
	print(f"  warmup	  = {config['warmup']}")
	
	print("\nHardware Parameters:")
	print(f"  mode		= {config['mode']} ({'CMOS' if config['mode'] == 0 else 'IMC'})")
	print(f"  bit_width   = {config['bit_width']}")
	print(f"  cycle_res   = {config['cycle_res']}")
	print(f"  variability   = {config['var']}")
	
	print("\nDevice:")
	print(f"  DEVICE	  = {config['DEVICE']}")
	
	print("\n" + "="*70)
	print(" Validation complete - ready for training!")
	print("="*70 + "\n")