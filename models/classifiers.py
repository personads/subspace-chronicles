import torch
import torch.nn as nn

from models.bayesian import *
from models.encoders import *

#
# Base Classifier
#


class EmbeddingClassifier(nn.Module):
	def __init__(self, emb_model: Encoder, lbl_models, classes):
		super().__init__()
		# internal models
		self._emb_models = emb_model
		# clone model for each embedding output
		self._lbl_models = lbl_models
		# internal variables
		self._classes = classes
		# public variables
		self.num_models = len(self._lbl_models)
		# move model to GPU if available
		if torch.cuda.is_available():
			self.to(torch.device('cuda'))
			for mdl_idx in range(len(self._lbl_models)):
				self._lbl_models[mdl_idx] = self._lbl_models[mdl_idx].to(torch.device('cuda'))

	def __repr__(self):
		return \
			f'<{self.__class__.__name__}:\n' \
			f'    emb_model = {"    ".join(repr(self._emb_models).splitlines(keepends=True))},\n' \
			f'    num_classifiers = {self.num_models},\n' \
			f'    num_classes = {len(self._classes)}\n>'

	def get_trainable_parameters(self):
		return list(self._emb_models.get_trainable_parameters()) + [p for m in self._lbl_models for p in m.parameters()]

	def get_savable_objects(self):
		objects = self._emb_models.get_savable_objects()
		objects['classifiers'] = self._lbl_models
		objects['classes'] = self._classes
		return objects

	def save(self, path):
		torch.save(self.get_savable_objects(), path)

	@classmethod
	def load(cls, path, emb_model):
		# objects = torch.load(path)
		# initially mapping parameters to CPU due to PyTorch bug when using MIG partitioning
		# https://github.com/pytorch/pytorch/issues/90543
		objects = torch.load(path, map_location=torch.device('cpu'))
		classes = objects['classes']
		lbl_models = objects['classifiers']
		# instantiate class using pre-trained label model and fixed encoder
		return cls(
			emb_model=emb_model, lbl_models=lbl_models, classes=classes
		)

	def forward(self, sentences):
		# embed sentences (batch_size, seq_length) -> (batch_size, num_active_layers, max_length, emb_dim)
		emb_sentences, att_sentences = self._emb_models(sentences)

		# logits for all tokens in all sentences + padding -inf (batch_size, num_active_layers, max_len, num_labels)
		logits = torch.ones(
			(emb_sentences.shape[0], emb_sentences.shape[1], emb_sentences.shape[2], len(self._classes)),
			device=emb_sentences.device
		) * float('-inf')
		# iterate over classifiers for each layer output
		for lyr_idx, classifier in enumerate(self._lbl_models):
			# get token embeddings of all sentences (total_tokens, emb_dim)
			emb_tokens = emb_sentences[:, lyr_idx, :, :][att_sentences]
			# pass through classifier
			flat_logits = classifier(emb_tokens)  # (num_words * num_layers, num_labels)
			logits[:, lyr_idx, :, :][att_sentences] = flat_logits  # (batch_size, num_layers, max_len, num_labels)

		results = {
			'embeddings': emb_sentences,
			'attentions': att_sentences,
			'logits': logits
		}
		# supply additional return values
		results = self.add_results(results)

		return results

	def get_labels(self, predictions):
		# return list of classifier-wise predictions per sentence (batch_size, num_classifiers, sen_len (var))
		labels = []

		logits = predictions['logits'].detach()
		# get predicted labels with maximum probability (padding should have -inf)
		max_labels = torch.argmax(logits, dim=-1)  # (batch_size, num_layers, max_len)

		# iterate over inputs items
		for sidx in range(logits.shape[0]):
			# gather non-padding (and non-special) label indices (num_layers, sen_len)
			label_idcs = max_labels[sidx, :, predictions['attentions'][sidx]]
			# append as list of string labels
			labels.append([[self._classes[l] for l in cls_labels] for cls_labels in label_idcs])

		return labels

	def add_results(self, results):
		return results


#
# Linear Classifiers
#


class MdlLinearClassifier(EmbeddingClassifier):
	def __init__(self, emb_model: Encoder, classes, lbl_models=None):
		# instantiate variational linear classifier
		if lbl_models is None:
			lbl_models = [
				LinearGroupNJ(emb_model.emb_dim, len(classes), clip_var=0.04)  # default from Voita and Titov (2020)
				for _ in range(emb_model.num_outputs)
			]

		super().__init__(
			emb_model=emb_model, lbl_models=lbl_models, classes=classes
		)

	def get_kld(self):
		# return list of cumulative KL divergences
		return torch.stack([m.kl_divergence() for m in self._lbl_models])

	def add_results(self, results):
		results['kld'] = self.get_kld()
		return results
