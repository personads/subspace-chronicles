import torch
import torch.nn as nn

#
# Loss Functions
#


class LabelLoss(nn.Module):
	def __init__(self, dataset, classes):
		super().__init__()
		self._xe_loss = nn.CrossEntropyLoss(ignore_index=-1)
		self._dataset = dataset
		self._classes = classes
		self._class2id = {c:i for i, c in enumerate(self._classes)}

	def __repr__(self):
		return \
			f'<{self.__class__.__name__}: XEnt (num_classes={len(self._classes)})>'

	def _gen_target_labels(self, logits, targets):
		# convert and generate target label indices as [s0l0t0 s0l1t0, ...]
		target_labels = [self._class2id.get(l, -1) for l in targets]
		target_labels = torch.tensor(target_labels, dtype=torch.long, device=logits.device)
		target_labels = torch.repeat_interleave(target_labels, logits.shape[1])
		return target_labels

	def forward(self, predictions, targets):
		assert 'logits' in predictions, f"[Error] LabelLoss requires predictions to contain logits."
		logits = predictions['logits']
		# gather target labels
		target_labels = self._gen_target_labels(logits, targets)
		# flatten logits
		flat_logits = logits[torch.sum(logits, dim=-1) != float('-inf'), :]  # (sum_over_seq_lens * num_layers,)

		return {'loss': self._xe_loss(flat_logits, target_labels)}

	def get_accuracy(self, predictions, targets):
		logits = predictions['logits'].detach()

		# gather target labels
		target_labels = self._gen_target_labels(logits, targets)

		# get labels from logits
		flat_logits = logits[torch.sum(logits, dim=-1) != float('-inf'), :]  # (sum_over_seq_lens * num_layers,)
		labels = torch.argmax(flat_logits, dim=-1)

		# compute label accuracy
		num_label_matches = torch.sum(labels == target_labels)
		accuracy = float(num_label_matches / labels.shape[0])

		return accuracy


class MdlLoss(LabelLoss):
	def __init__(self, dataset, classes):
		super().__init__(dataset, classes)
		self._xe_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')  # following Voita and Titov (2020)
		self._total_targets = len(list(self._dataset.get_flattened_labels()))

	def __repr__(self):
		return \
			f'<{self.__class__.__name__}: KLD + XEnt (num_classes={len(self._classes)})>'

	def forward(self, predictions, targets):
		assert ('logits' in predictions) and ('kld' in predictions), \
			f"[Error] MdlLoss requires output to contain logits and KL divergence."
		logits = predictions['logits']
		kld = torch.mean(predictions['kld'])  # mean over KL divergence across layers

		num_layers = predictions['logits'].shape[1]

		# gather target labels
		target_labels = self._gen_target_labels(logits, targets)
		# flatten logits
		flat_logits = logits[torch.sum(logits, dim=-1) != float('-inf'), :]  # (sum_over_seq_lens * num_layers,)

		# compute cross-entropy loss
		xe_loss = self._xe_loss(flat_logits, target_labels) / num_layers

		# compute MDL loss
		num_targets = flat_logits.shape[0] / num_layers
		mdl_loss = kld * num_targets / self._total_targets

		# total loss
		loss = xe_loss + mdl_loss

		outputs = {
			'loss': loss,
			'loss/xe': xe_loss,
			'loss/mdl': mdl_loss,
			'loss/kld': kld
		}

		return outputs
