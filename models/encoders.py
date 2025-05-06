import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class Encoder(nn.Module):
	def __init__(
			self, model_name, lyr_selector,
			lyr_tuning=False, emb_tuning=False,
			lyr_pooling=False, emb_pooling=None, tokenized=False, specials=False,
			model_revision='main', cache=None):
		super().__init__()
		# load transformer
		transformers.logging.set_verbosity_error()
		# auto-load appropriate model and tokenizer
		self._lm = transformers.AutoModel.from_pretrained(
			model_name, revision=model_revision,
			output_hidden_states=True, return_dict=True
		)
		self._tok = transformers.AutoTokenizer.from_pretrained(
			model_name, revision=model_revision, use_fast=True,
			add_prefix_space=True,
			model_max_length=self._lm.config.max_position_embeddings  # compensate for models missing max_length
		)
		# set PAD-token if none is configured (e.g., GPT models)
		if self._tok.pad_token is None:
			self._tok.pad_token = self._tok.eos_token

		# setup trainable parameters
		self._emb_tuning = emb_tuning
		# load cache
		self._cache = cache  # {hash: torch.tensor (num_layers, sen_len, emb_dim)}
		# internal variables
		self._lm_name = model_name
		self._lm_revision = model_revision
		self._tokenized = tokenized
		self._specials = specials
		self._emb_pooling = emb_pooling
		# layer selector parameters
		self._lyr_tuning = lyr_tuning
		self._lyr_pooling = lyr_pooling
		self._lyr_selector = nn.Parameter(lyr_selector) if self._lyr_tuning else lyr_selector
		# public variables
		self.emb_dim = self._lm.config.hidden_size
		self.num_layers = self._lm.config.num_hidden_layers + 1
		self.num_outputs = 1 if self._lyr_selector.dtype == torch.float else int(torch.sum(self._lyr_selector).detach().cpu())
		# move model to GPU if available
		if torch.cuda.is_available():
			self.to(torch.device('cuda'))

	def __repr__(self):
		return \
			f'<{self.__class__.__name__}:\n' \
			f'    (N,M)→(N,{self.num_layers},M,{self.emb_dim})→' \
			f'{f"[pool]→(N,{self.num_outputs},M,{self.emb_dim})→" if not self._lyr_pooling else ""}' \
			f'{f"[pool]→" if self._lyr_pooling else ""}' \
			f'(N,{self.num_outputs},M,{self.emb_dim}),\n' \
			f'    {self._lm.__class__.__name__} ("{self._lm_name} ({self._lm_revision})"),\n' \
			f'    {"tokenized" if self._tokenized else "raw"} inputs' \
			f', {"tunable" if self._emb_tuning else "static"} embeddings' \
			f', pooling {"disabled" if self._emb_pooling is None else "enabled"},\n' \
			f'    {"tunable" if self._lyr_tuning else "static"} layer selection' \
			f', {f"{torch.sum(self._lyr_selector != 0)}/{self._lyr_selector.shape[0]}"} active layers,\n' \
			f'    {f"w/" if self._cache is not None else "w/o"} cache\n>'

	def get_savable_objects(self):
		objects = {}
		objects['language_model_name'] = self._lm_name
		objects['language_model_revision'] = self._lm_revision
		objects['embedding_pooling'] = self._emb_pooling
		if self._emb_tuning:
			objects['language_model'] = self._lm.state_dict()
		if self._lyr_tuning:
			objects['layer_selector'] = self._lyr_selector
		return objects

	def get_trainable_parameters(self):
		parameters = []
		if self._emb_tuning:
			parameters += list(self._lm.parameters())
		if self._lyr_tuning:
			parameters.append(self._lyr_selector)
		return parameters

	def save(self, path):
		torch.save(self.get_savable_objects(), path)

	@staticmethod
	def load(
			path,
			lyr_selector=None,
			lyr_tuning=False, emb_tuning=False,
			lyr_pooling=False, emb_pooling=None, specials=False, cache=None):
		# objects = torch.load(path)
		# initially mapping parameters to CPU due to PyTorch bug when using MIG partitioning
		# https://github.com/pytorch/pytorch/issues/90543
		objects = torch.load(path, map_location=torch.device('cpu'))

		# load necessary components
		lm_name = objects['language_model_name']
		lm_revision = objects.get('language_model_revision', 'main')
		emb_pooling = objects.get('embedding_pooling', emb_pooling)
		lyr_selector = objects.get('layer_selector', lyr_selector)

		# construct model
		encoder = Encoder(
			model_name=lm_name, model_revision=lm_revision,
			lyr_selector=lyr_selector,
			lyr_tuning=lyr_tuning, emb_tuning=emb_tuning,
			lyr_pooling=lyr_pooling, emb_pooling=emb_pooling, specials=specials, cache=cache
		)

		# load fine-tuned language model (if available)
		if 'language_model' in objects:
			encoder.load_language_model(path)

		return encoder

	def load_language_model(self, path):
		objects = torch.load(path, map_location=torch.device('cpu'))
		assert objects['language_model_name'] == self._lm_name, \
			f"[Error] Checkpoint LM '{objects['language_model_name']}' does not match LM '{self._lm_name}'."
		missing_parameters, unexpected_parameters = self._lm.load_state_dict(objects['language_model'], strict=False)
		assert len(missing_parameters) == 0, \
			f"[Error] Missing {len(missing_parameters)} parameters while loading pre-trained LM weights from '{path}': " \
			f"{missing_parameters}."

	def train(self, mode=True):
		super().train(mode)
		# disable LM training (incl. dropout)
		if not self._emb_tuning:
			self._lm.eval()
		return self

	def forward(self, sentences):
		# embed sentences returning (batch_size, num_layers, max_len, emb_dim) + (batch_size, max_len)
		if self._emb_tuning:
			emb_tokens, att_tokens = self.embed(sentences)
		else:
			with torch.no_grad():
				emb_tokens, att_tokens = self.embed(sentences)

		# apply layer selection
		if not self._lyr_pooling:
			# apply layer selection (bool: mask, float: weighted sum)
			if self._lyr_selector.dtype == torch.bool:
				emb_tokens = emb_tokens[:, self._lyr_selector, :, :]
			else:
				# compute weighted sum over layers
				emb_tokens = torch.einsum('blth, l->bth', emb_tokens, torch.softmax(self._lyr_selector, dim=0))[:, None, :]

		# apply layer pooling
		if self._lyr_pooling:
			emb_tokens = torch.einsum('blth, l->bth', emb_tokens, torch.softmax(self._lyr_selector, dim=0))[:, None, :]

		# pool token embeddings
		if self._emb_pooling is not None:
			# prepare sentence embedding tensor (batch_size, num_active_layers, 1, emb_dim)
			emb_pooled = torch.zeros((emb_tokens.shape[0], emb_tokens.shape[1], 1, emb_tokens.shape[3]), device=emb_tokens.device)
			# iterate over sentences and pool relevant tokens
			for sidx in range(emb_tokens.shape[0]):
				for lidx in range(emb_tokens.shape[1]):
					emb_pooled[sidx, lidx, 0, :] = self._emb_pooling(emb_tokens[sidx, lidx, :torch.sum(att_tokens[sidx]), :])
			emb_tokens = emb_pooled
			# set embedding attention mask to cover each sentence embedding
			att_tokens = torch.ones((att_tokens.shape[0], 1), dtype=torch.bool)

		return emb_tokens, att_tokens

	def embed(self, sentences):
		# try retrieving embeddings from cache
		emb_cache = self.retrieve(sentences)
		if emb_cache is not None:
			emb_tokens, att_tokens = emb_cache
			return emb_tokens, att_tokens

		# compute embeddings if not in cache
		tok_sentences = self.tokenize(sentences)
		model_inputs = {
			k: tok_sentences[k] for k in ['input_ids', 'attention_mask']  # no segment embeddings for GPT ['input_ids', 'token_type_ids', 'attention_mask']
			if k in tok_sentences
		}

		# perform embedding forward pass
		model_outputs = self._lm(**model_inputs)
		hidden_states = model_outputs.hidden_states  # (num_layers * (batch_size, max_len, hidden_dim))
		emb_tokens = torch.stack(hidden_states, dim=1)
		# (batch_size, num_layers, max_len, emb_dim) + (batch_size, max_len)
		emb_tokens, att_tokens = emb_tokens, tok_sentences['attention_mask'].bool()

		# remove special tokens
		if (not self._specials) and (self._emb_pooling is None):
			# mask any non-padding token that is special
			non_special_mask = ~tok_sentences['special_tokens_mask'].bool() & att_tokens  # (batch_size, max_len)
			# find first + last token index that does not solely consist of specials (e.g., exclude first [CLS])
			non_special_start_idx, non_special_end_idx = torch.flatten(torch.sum(non_special_mask, dim=0).nonzero())[[0, -1]]
			# truncate embeddings into range of non-special tokens (non-max-length sequences still keep specials at end)
			emb_tokens = emb_tokens[:, :, non_special_start_idx:non_special_end_idx+1, :]  # (batch_size, num_layers, new_max_len, emb_dim)
			# truncate and remap attention map to only contain non-padding + non-special tokens
			att_tokens = non_special_mask[:, non_special_start_idx:non_special_end_idx+1]  # (batch_size, new_max_len)

		# store embeddings in cache (if cache is enabled)
		if self._cache is not None:
			self.cache(sentences, emb_tokens, att_tokens)

		return emb_tokens, att_tokens

	def tokenize(self, sentences, tokenized=None):
		tokenized = self._tokenized if tokenized is None else tokenized
		# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]], special_tokens_mask: [[]]}
		# check for multi-sentence inputs
		if (len(sentences) > 0) and (type(sentences[0]) is tuple):
			sentences1, sentences2 = [s[0] for s in sentences], [s[1] for s in sentences]
			tok_sentences = self._tok(
				sentences1, sentences2,
				padding=True, truncation=True,
				return_tensors='pt', return_special_tokens_mask=True,
				return_offsets_mapping=True, is_split_into_words=tokenized
			)
		# use standard tokenization otherwise
		else:
			tok_sentences = self._tok(
				sentences,
				padding=True, truncation=True,
				return_tensors='pt', return_special_tokens_mask=True,
				return_offsets_mapping=True, is_split_into_words=tokenized
			)

		# move input to GPU (if available)
		if torch.cuda.is_available():
			for k, v in tok_sentences.items():
				tok_sentences[k] = v.to(torch.device('cuda'))

		return tok_sentences

	def retrieve(self, sentences):
		# check if cache is disabled or empty
		if (self._cache is None) or (len(self._cache) < 1):
			return None

		emb_retrieved = []
		max_len = 0

		# iterate over sentences
		for sidx, sentence in enumerate(sentences):
			# retrieve sentence embedding using string hash
			sen_hash = hashlib.md5(' '.join(sentence).encode('utf-8')).hexdigest()
			# skip batch if not all sentences are in cache
			if sen_hash not in self._cache:
				return None

			# retrieve embeddings from cache
			emb_cached = self._cache[sen_hash]  # (num_layers, sen_len, emb_dim)
			emb_retrieved.append(emb_cached)
			max_len = max_len if max_len > emb_cached.shape[1] else emb_cached.shape[1]

		emb_tokens = torch.zeros((len(sentences), self.num_layers, max_len, self.emb_dim))
		att_tokens = torch.zeros((len(sentences), max_len), dtype=torch.bool)

		for sidx, emb_cached in enumerate(emb_retrieved):
			emb_tokens[sidx, :, :emb_cached.shape[1], :] = emb_cached
			att_tokens[sidx, :emb_cached.shape[1]] = True

		# move input to GPU (if available)
		if torch.cuda.is_available():
			emb_tokens = emb_tokens.to(torch.device('cuda'))
			att_tokens = att_tokens.to(torch.device('cuda'))

		return emb_tokens, att_tokens

	def cache(self, sentences, emb_tokens, att_tokens):
		# detach, duplicate and move embeddings to CPU
		emb_tokens = emb_tokens.detach().cpu()

		# iterate over sentences
		for sidx, sentence in enumerate(sentences):
			# compute sentence hash
			sen_hash = hashlib.md5(' '.join(sentence).encode('utf-8')).hexdigest()
			# store cache entry
			self._cache[sen_hash] = emb_tokens[sidx, :, :torch.sum(att_tokens[sidx]), :]  # (num_layers, sen_len, emb_dim)


#
# Pooling Functions
#


def get_mean_embedding(token_embeddings):
	return torch.mean(token_embeddings, dim=0)


def get_first_embedding(token_embeddings):
	return token_embeddings[0]
