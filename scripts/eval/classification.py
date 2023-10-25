#!/usr/bin/python3

import argparse, json, os, sys

import torch

from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.encoders import Encoder
from utils.datasets import LabelledDataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Evaluate Classification Performance')
	arg_parser.add_argument('--target', required=True, help='path to target CSV')
	arg_parser.add_argument('--prediction', required=True, help='path to predicted CSV')
	arg_parser.add_argument('-t', '--tokenizer', help='name of HuggingFace tokenizer')
	arg_parser.add_argument('-o', '--output', help='path to output JSON')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	tgt_data = LabelledDataset.from_path(args.target)
	print(f"Loaded target data {tgt_data} from '{args.target}'.")
	prd_data = LabelledDataset.from_path(args.prediction)
	print(f"Loaded predicted data {prd_data} from '{args.prediction}'.")
	labels = tgt_data.get_label_types()

	if args.tokenizer:
		encoder = Encoder(args.tokenizer, None, torch.tensor([True], dtype=torch.bool))
		print(f"Loaded '{args.tokenizer}' for label repetition.")
		tgt_data.repeat_labels(encoder, batch_size=100, verbose=True)
		print()

	targets, predictions = [], []
	for idx in range(tgt_data.get_input_count()):
		tgt_text, tgt_labels = tgt_data[idx]
		prd_text, prd_labels = prd_data[idx]

		# case: both labels apply to the full sequence
		if (type(prd_labels) is not list) and (type(tgt_labels) is not list):
			tgt_labels, prd_labels = [tgt_labels], [prd_labels]
		# case: predictions are token-level, but targets are sequence-level
		elif (type(prd_labels) is list) and (type(tgt_labels) is not list):
			# repeat target label across sequence
			tgt_labels = [tgt_labels for _ in range(len(prd_labels))]

		targets += tgt_labels
		predictions += prd_labels

	# compute metrics
	metrics = {}

	# Accuracy
	num_correct = sum([1 for tl, pl in zip(targets, predictions) if tl == pl])
	accuracy = num_correct/len(predictions)
	metrics['accuracy_mean'] = accuracy
	print(f"Accuracy (mean): {accuracy * 100:.2f}%")

	# macro-F1
	f1_macro = f1_score(targets, predictions, average='macro')
	metrics['f1_macro'] = f1_macro
	print(f"F1 (macro): {f1_macro * 100:.2f}%")

	# micro-F1
	f1_micro = f1_score(targets, predictions, average='micro')
	metrics['f1_micro'] = f1_micro
	print(f"F1 (micro): {f1_micro * 100:.2f}%")

	# class-wise F1
	f1_class = f1_score(targets, predictions, average=None, labels=labels)
	metrics['f1'] = {}
	print("F1 (class-wise):")
	for label, f1_label in zip(labels, f1_class):
		metrics['f1'][label] = f1_label
		print(f"  {label}: {f1_label * 100:.2f}%")

	# store metrics in JSON
	if args.output is not None:
		# save updated metrics
		with open(args.output, 'w') as fp:
			json.dump(metrics, fp, indent=4, sort_keys=True)
		print(f"Saved metrics to '{args.output}'.")


if __name__ == '__main__':
	main()
