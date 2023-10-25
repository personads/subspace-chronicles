import sys

import torch
import numpy as np

from collections import defaultdict


def classify_dataset(
		classifier, criterion, optimizer,
		dataset, batch_size, mode='train',
		return_predictions=False):
	stats = defaultdict(list)

	# set model to training mode
	if mode == 'train':
		classifier.train()
		batch_generator = dataset.get_shuffled_batches
	# set model to eval mode
	elif mode == 'eval':
		classifier.eval()
		batch_generator = dataset.get_batches

	# iterate over batches
	for bidx, batch_data in enumerate(batch_generator(batch_size)):
		# set up batch data
		sentences, labels, num_remaining = batch_data

		# when training, perform both forward and backward pass
		if mode == 'train':
			# zero out previous gradients
			optimizer.zero_grad()

			# forward pass
			predictions = classifier(sentences)

			# propagate loss
			outputs = criterion(predictions, labels)
			loss = outputs['loss']
			loss.backward()
			optimizer.step()

		# when evaluating, perform forward pass without gradients
		elif mode == 'eval':
			with torch.no_grad():
				# forward pass
				predictions = classifier(sentences)
				# calculate loss
				outputs = criterion(predictions, labels)

		# calculate accuracy
		accuracy = criterion.get_accuracy(predictions, labels)

		# store statistics
		for stat in outputs:
			if not stat.startswith('loss'):
				continue
			stats[stat].append(float(outputs[stat].detach()))
		stats['accuracy'].append(float(accuracy))

		# convert logits to labels and store predictions
		if return_predictions:
			stats['predictions'] += classifier.get_labels(predictions)

		# print batch statistics
		pct_complete = (1 - (num_remaining / len(dataset._inputs))) * 100
		sys.stdout.write(
			f"\x1b[1K\r[{mode.capitalize()} | Batch {bidx + 1} | {pct_complete:.2f}%] "
			f"Acc: {np.mean(stats['accuracy']):.4f}, Loss: "
			f"({' | '.join(f'{stat}: {np.mean(val):.4f}' for stat, val in stats.items() if stat.startswith('loss'))})"
		)
		sys.stdout.flush()

	# clear line
	print("\x1b[1K\r", end='')

	return stats
