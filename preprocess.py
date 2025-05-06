#!/usr/bin/python3

import argparse, json, os, sys

from collections import defaultdict

# local imports
from utils.setup import *
from models.encoders import Encoder
from utils.datasets import LabelledDataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Dataset Preprocessing')
	# data setup
	arg_parser.add_argument('--data-path', required=True, help='path to original data')
	arg_parser.add_argument('--output-path', required=True, help='output path for processed data')

	# encoder setup
	arg_parser.add_argument('--model-name', required=True, help='language model identifier')
	arg_parser.add_argument('--model-revision', default='main', help='language model revision (default: main)')

	# preprocessing pipeline
	arg_parser.add_argument(
		'-rl', '--repeat-labels', action='store_true', default=False,
		help='set flag to repeat sequence target label for each token in sequence (default: False)')
	arg_parser.add_argument(
		'--tokenized', action='store_true', default=False,
		help='set flag to indicate pre-tokenized datasets (default: False)')

	# execution setup
	arg_parser.add_argument(
		'-bs', '--batch-size', type=int, default=64,
		help='maximum number of sentences per batch (default: 32)')

	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# set up data
	dataset = LabelledDataset.from_path(args.data_path, tokenized=args.tokenized)
	dataset_inputs = dataset.get_inputs()
	dataset_labels = dataset._labels
	logging.info(f"Loaded {dataset} from '{args.data_path}'.")

	# load encoder model
	encoder = Encoder(
		model_name=args.model_name, model_revision=args.model_revision,
		lyr_selector=torch.tensor([True], dtype=torch.bool)
	)
	print(f"Loaded '{args.model_name}':")
	print(encoder)

	# align dataset labels with encoder tokenization
	if args.repeat_labels:
		dataset.repeat_labels(encoder, batch_size=args.batch_size, verbose=True)
		dataset_labels = dataset._repeated_labels
		print("\x1b[1K\r", end="")
		print("Aligned dataset labels with encoder tokenization (repeat: on).")

	# save pre-processed dataset
	processed_dataset = LabelledDataset(dataset_inputs, dataset_labels)
	processed_dataset.save(args.output_path)
	print(f"Saved pre-processed {processed_dataset} to '{args.output_path}'.")


if __name__ == '__main__':
	main()
