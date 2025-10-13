"""Utilities for extracting embeddings from Hugging Face transformer models.

This module provides small helpers to load a tokenizer+model, compute
batched embeddings with mean pooling, and save/load embeddings to disk.

Default model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

Usage example:
	from embedding_utils import load_model_and_tokenizer, embed_texts

	tokenizer, model, device = load_model_and_tokenizer()
	embs = embed_texts(['text1','text2'], tokenizer, model, device)

Notes:
 - For production/fast CPU inference, consider quantization (bitsandbytes) or
   convert the model with optimum. This file stays framework-agnostic and
   keeps the minimal PyTorch/transformers path.
"""

from typing import List, Optional, Tuple
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-v0.1"
# DEFAULT_MODEL_ID = "openbmb/MiniCPM-V-4_5"


def get_device(force_cpu: bool = False) -> torch.device:
	"""Return the torch device to use.

	If CUDA is available and force_cpu is False, returns cuda device.
	"""
	if not force_cpu and torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def load_model_and_tokenizer(
	model_id: str = DEFAULT_MODEL_ID,
	device: Optional[torch.device] = None,
	trust_remote_code: bool = False,
) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
	"""Load tokenizer and model from Hugging Face.

	Args:
		model_id: HF model id or local path.
		device: Optional torch.device. If None, auto selected.
		trust_remote_code: Pass to from_pretrained if model repo requires it.

	Returns:
		tokenizer, model, device
	"""
	if device is None:
		device = get_device()

	logger.info("Loading tokenizer for %s", model_id)
	# Load tokenizer (don't pass dtype to tokenizer)
	tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)

	# Ensure tokenizer has a pad token. Some causal LM tokenizers don't define one.
	tokens_added = False
	if tokenizer.pad_token is None:
		# Prefer using eos_token as pad if available (safe for causal models)
		if getattr(tokenizer, "eos_token", None) is not None:
			tokenizer.pad_token = tokenizer.eos_token
			logger.info("Tokenizer had no pad_token, falling back to eos_token as pad_token")
		else:
			# Add a new pad token and mark that we added tokens so model embeddings can be resized
			pad_tok = "[PAD]"
			tokenizer.add_special_tokens({"pad_token": pad_tok})
			tokens_added = True
			logger.info("Tokenizer had no pad_token; added special token '%s'", pad_tok)

	logger.info("Loading model for %s", model_id)
	# Choose torch_dtype only for CUDA to avoid CPU float16 issues
	torch_dtype = torch.float16 if device.type == "cuda" else None
	model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code, dtype=torch_dtype)
	model.to(device)

	# If we added tokenizer tokens (like pad), resize model token embeddings
	if tokens_added:
		try:
			model.resize_token_embeddings(len(tokenizer))
			logger.info("Resized model token embeddings to %d", len(tokenizer))
		except Exception as e:
			logger.warning("Could not resize model embeddings after adding tokens: %s", e)
	model.eval()

	logger.info("Loaded model on device: %s", device)
	return tokenizer, model, device


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
	"""Compute mean pooling of token embeddings, ignoring padding tokens.

	Args:
		hidden_states: (batch, seq_len, dim)
		attention_mask: (batch, seq_len)

	Returns:
		Tensor (batch, dim)
	"""
	mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
	masked = hidden_states * mask
	summed = masked.sum(dim=1)
	counts = mask.sum(dim=1).clamp(min=1e-9)
	return summed / counts


def embed_texts(
	texts: List[str],
	tokenizer: AutoTokenizer,
	model: AutoModel,
	device: torch.device,
	batch_size: int = 8,
	normalize: bool = True,
) -> np.ndarray:
	"""Compute embeddings for a list of texts and return a NumPy array.

	- Tokenizes with padding/truncation
	- Runs model in torch.no_grad()
	- Mean-pools token embeddings using the attention mask
	- Optionally L2-normalizes vectors (useful for cosine similarity)

	Returns:
		numpy array of shape (len(texts), dim)
	"""
	all_embs = []
	n = len(texts)
	if n == 0:
		return np.zeros((0, model.config.hidden_size), dtype=np.float32)

	for i in tqdm(range(0, n, batch_size), desc="Embedding texts", unit="batch"):
		batch_texts = texts[i : i + batch_size]
		enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
		input_ids = enc["input_ids"].to(device)
		attention_mask = enc["attention_mask"].to(device)

		with torch.no_grad():
			outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

			# last_hidden_state is standard for AutoModel
			last_hidden = outputs.last_hidden_state
			pooled = mean_pooling(last_hidden, attention_mask)

			if normalize:
				pooled = F.normalize(pooled, p=2, dim=1)

			all_embs.append(pooled.cpu().numpy())

	return np.vstack(all_embs)


def save_embeddings(path: str, embeddings: np.ndarray) -> None:
	"""Save embeddings to a .npy file (overwrites)."""
	dirname = os.path.dirname(path)
	if dirname and not os.path.exists(dirname):
		os.makedirs(dirname, exist_ok=True)
	np.save(path, embeddings)
	logger.info("Saved embeddings to %s", path)


def load_embeddings(path: str) -> np.ndarray:
	"""Load embeddings from a .npy file."""
	return np.load(path)


__all__ = [
	"get_device",
	"load_model_and_tokenizer",
	"embed_texts",
	"save_embeddings",
	"load_embeddings",
	"mean_pooling",
	"DEFAULT_MODEL_ID",
]


if __name__ == "__main__":
	# Small demo when executed directly. This will download the model if needed.
	logging.basicConfig(level=logging.INFO)
	tokenizer, model, device = load_model_and_tokenizer()

	demo_texts = [
		"Red cotton t-shirt with round neck",
		"Wireless Bluetooth headphones with noise cancellation",
		"Lightweight running shoes size 10",
	]

	emb = embed_texts(demo_texts, tokenizer, model, device, batch_size=4)
	print("Embeddings shape:", emb.shape)
	print("First vector (first 5 dims):", emb[0][:5])
	save_embeddings("embeddings_demo.npy", emb)
