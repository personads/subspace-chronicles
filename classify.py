#!/usr/bin/python3

import argparse, logging, json, os, sys

import numpy as np

from collections import defaultdict

# local imports
from utils.setup import *
from utils.datasets import LabelledDataset
from utils.training import classify_dataset
from models.encoders import *
from models.classifiers import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Classifier Training')
	# data setup
	arg_parser.add_argument(
		'--train-path',
		help='path to training data')
	arg_parser.add_argument(
		'--valid-path', required=True,
		help='path to validation data')
	arg_parser.add_argument(
		'--repeat-labels', action='store_true', default=False,
		help='set flag to repeat sequence target label for each token in sequence (default: False)')
	arg_parser.add_argument(
		'--prediction', action='store_true', default=False,
		help='set flag to only perform prediction on the validation data without training (default: False)')

	# encoder setup
	arg_parser = setup_model_arguments(arg_parser)

	# experiment setup
	arg_parser.add_argument(
		'--exp-path', required=True,
		help='path to experiment directory')
	arg_parser.add_argument(
		'--epochs', type=int, default=30,
		help='maximum number of epochs (default: 50)')
	arg_parser.add_argument(
		'--early-stop', type=int, default=1,
		help='maximum number of epochs without improvement (default: 1)')
	arg_parser.add_argument(
		'--batch-size', type=int, default=32,
		help='maximum number of sentences per batch (default: 32)')
	arg_parser.add_argument(
		'--learning-rate', type=float, default=1e-3,
		help='learning rate (default: 1e-3)')
	arg_parser.add_argument(
		'--decay-rate', type=float, default=.5,
		help='learning rate decay (default: 0.5)')
	arg_parser.add_argument(
		'--random-seed', type=int,
		help='seed for probabilistic components (default: None)')

	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# set up experiment directory and logging
	setup_experiment(args.exp_path, prediction=args.prediction)

	model_path = None
	if args.prediction:
		logging.info("Running in prediction mode (no training).")
		model_path = os.path.join(args.exp_path, 'best.pt') if model_path is None else model_path
		if not os.path.exists(model_path):
			logging.error(f"[Error] No pre-trained model available at '{model_path}'. Exiting.")
			exit(1)

	# set random seeds
	if args.random_seed is not None:
		np.random.seed(args.random_seed)
		torch.random.manual_seed(args.random_seed)

	# set up data
	train_data = None
	valid_data = LabelledDataset.from_path(args.valid_path)
	label_types = sorted(set(valid_data.get_label_types()))
	logging.info(f"Loaded {valid_data} (dev).")
	if args.train_path is not None:
		train_data = LabelledDataset.from_path(args.train_path)
		logging.info(f"Loaded {train_data} (train).")
		# gather labels
		if set(train_data.get_label_types()) < set(valid_data.get_label_types()):
			logging.warning(f"[Warning] Validation data contains labels unseen in the training data.")
		label_types = sorted(set(train_data.get_label_types()) | set(valid_data.get_label_types()))

	# set up encoder
	encoder = setup_encoder(
		model_name=args.model_name, model_revision=args.model_revision,
		layers_id=args.layers, filter_id=args.filter,
		model_path=model_path, encoder_path=args.encoder_path,
		emb_caching=args.embedding_caching, emb_pooling=args.embedding_pooling, emb_tuning=args.embedding_tuning,
		lyr_pooling=args.layer_pooling
	)

	# align dataset labels with encoder tokenization
	if args.repeat_labels:
		valid_data.repeat_labels(encoder, batch_size=args.batch_size, verbose=True)
		if train_data is not None:
			train_data.repeat_labels(encoder, batch_size=args.batch_size, verbose=True)
		print("\x1b[1K\r", end="")
		logging.info("Aligned dataset labels with encoder tokenization (repeat: on).")

	# setup classifier and losses based on identifier
	classifier, train_criterion, valid_criterion = setup_classifier(
		args.classifier, encoder=encoder,
		train_data=train_data, valid_data=valid_data, classes=label_types,
		model_path=model_path
	)
	logging.info(f"{'Constructed' if model_path is None else 'Loaded pre-trained'} classifier:\n{classifier}")

	logging.info(f"Using validation criterion {valid_criterion}.")

	# main prediction call (when only predicting on validation data w/o training)
	if args.prediction:
		stats = classify_dataset(
			classifier, valid_criterion, None, valid_data,
			args.batch_size, mode='eval', return_predictions=True
		)
		logging.info(
			f"Prediction completed with Acc: {np.mean(stats['accuracy']):.4f}, Losses: "
			f"({' | '.join(f'{stat}: {np.mean(val):.4f}' for stat, val in stats.items() if stat.startswith('loss'))}) "
			f"(mean over batches and classifiers)."
		)

		# iterate over classifiers
		for cls_idx in range(classifier.num_models):
			# convert label indices back to string labels
			cls_pred_labels = [
				stats['predictions'][sen_idx][cls_idx]
				for sen_idx in range(len(stats['predictions']))
			]
			# construct dataset with predicted labels
			pred_data = LabelledDataset(valid_data._inputs, cls_pred_labels)
			pred_prefix = os.path.splitext(os.path.basename(args.valid_path))[0]
			pred_path = os.path.join(args.exp_path, f'{pred_prefix}-pred-{cls_idx}.csv')
			pred_data.save(pred_path)
			logging.info(f"Saved results from {pred_data} to '{pred_path}'.")

		# save prediction statistics
		stats_path = os.path.join(args.exp_path, f'{pred_prefix}-stats.json')
		with open(stats_path, 'w', encoding='utf8') as fp:
			compact_stats = dict(stats)
			del(compact_stats['predictions'])
			json.dump(compact_stats, fp, indent=4, sort_keys=True)
		logging.info(f"Saved prediction statistics to '{stats_path}'.")

		logging.info("Exiting.")
		exit()

	# setup training loss
	logging.info(f"Using training criterion {train_criterion}.")

	# setup optimizer and scheduler
	optimizer = torch.optim.Adam(params=classifier.get_trainable_parameters(), lr=args.learning_rate)
	logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {args.learning_rate}.")
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.decay_rate, patience=0)
	logging.info(f"Scheduler {scheduler.__class__.__name__} reduces learning rate by {args.decay_rate} after 1 epoch without improvement.")

	# main training loop
	stats = defaultdict(lambda: defaultdict(list))
	for ep_idx in range(args.epochs):
		# iterate over training batches and update classifier weights
		ep_stats = classify_dataset(
			classifier, train_criterion, optimizer, train_data,
			args.batch_size, mode='train'
		)
		# store and print statistics
		for stat in ep_stats:
			stats['train'][stat].append(np.mean(ep_stats[stat]))
		logging.info(
			f"[Epoch {ep_idx + 1}/{args.epochs}] Train completed with "
			f"Acc: {np.mean(stats['train']['accuracy']):.4f}, Loss: "
			f"({' | '.join(f'{stat}: {np.mean(val):.4f}' for stat, val in stats['train'].items() if stat.startswith('loss'))})"
		)

		# iterate over batches in dev split
		ep_stats = classify_dataset(
			classifier, valid_criterion, None, valid_data,
			args.batch_size, mode='eval'
		)

		# store and print statistics
		for stat in ep_stats:
			stats['valid'][stat].append(np.mean(ep_stats[stat]))
		logging.info(
			f"[Epoch {ep_idx + 1}/{args.epochs}] Validation completed with "
			f"Acc: {stats['valid']['accuracy'][-1]:.4f}, Loss: "
			f"({' | '.join(f'{stat}: {np.mean(val):.4f}' for stat, val in stats['valid'].items() if stat.startswith('loss'))})"
		)
		cur_eval_loss = stats['valid']['loss'][-1]

		# save most full recent model
		if args.embedding_tuning:
			path = os.path.join(args.exp_path, f'latest.pt')
		# save probes for all epochs
		else:
			path = os.path.join(args.exp_path, f'epoch-{ep_idx}.pt')
		classifier.save(path)
		logging.info(f"Saved model from epoch {ep_idx + 1} to '{path}'.")

		# save training statistics
		path = os.path.join(args.exp_path, 'stats.json')
		with open(path, 'w', encoding='utf8') as fp:
			json.dump(stats, fp, indent=4, sort_keys=True)
		logging.info(f"Saved training statistics to '{path}'.")

		# save best model
		if cur_eval_loss <= min(stats['valid']['loss']):
			path = os.path.join(args.exp_path, 'best.pt')
			classifier.save(path)
			logging.info(f"Saved model with best loss {cur_eval_loss:.4f} to '{path}'.")

		# check for early stopping
		if (ep_idx - stats['valid']['loss'].index(min(stats['valid']['loss']))) > args.early_stop:
			logging.info(f"No improvement since {args.early_stop + 1} epochs ({min(stats['valid']['loss']):.4f} loss). Early stop.")
			break

	logging.info(f"Training completed after {ep_idx + 1} epochs.")


if __name__ == '__main__':
	main()
