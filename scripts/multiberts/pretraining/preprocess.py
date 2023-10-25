#!/usr/bin/python3

import argparse, csv, datetime, json, logging, os, pickle, random, sys, time

import multiprocessing as mp

import numpy as np
import spacy
import torch

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.setup import setup_logging


#
# Helper Classes
#

class Sentencizer():
	def __init__(self):
		self.model = spacy.load('en_core_web_sm')

	def sentencize(self, instance):
		# return sentences as strings (if they are non-empty)
		return [[span.text.strip()] for span in self.model(instance['text']).sents if span.text.strip()]


#
# Helper Functions
#

def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='MultiBERTs Training Data Preprocessing')

	arg_parser.add_argument(
		'--out-path', required=True,
		help='path to output directory')
	arg_parser.add_argument(
		'--continued', action='store_true', default=False,
		help='set flag to continue processing from existing files and appending (default: False)')
	arg_parser.add_argument(
		'--seed', type=int, default=0,
		help='random seed for masking and shuffling (default: 0)')
	arg_parser.add_argument(
		'--batch-size', type=int, default=256,
		help='batch size for final processing stage (default: 256)')

	return arg_parser.parse_args()


def generate_sentencized_dataset(sentencizer, dataset, out_path):
	with open(out_path, 'w', encoding='utf8', newline='') as out_file:
		csv_writer = csv.writer(out_file, quoting=csv.QUOTE_ALL, lineterminator='\n')
		csv_writer.writerow(['text'])
		num_sentences = 0
		with mp.Pool(min(64, mp.cpu_count())) as pool:
			for idx, sentences in enumerate(pool.imap(sentencizer.sentencize, dataset, chunksize=32)):
				print(f"\r[{idx}/{dataset.num_rows}] Sentencizing...", end='', flush=True)
				csv_writer.writerows(sentences)
				num_sentences += len(sentences)
		logging.info(f"\rSentencized {dataset.num_rows} documents.")
	logging.info(f"Saved {num_sentences} sentences to '{out_path}'.")


def generate_sentence_pairs(dataset):
	# generate correct sentence pairs ((current_sentence, next_sentence), num_instances/2)
	num_instances = dataset.num_rows if dataset.num_rows % 2 == 0 else dataset.num_rows - 1
	logging.info(f"Generating {num_instances // 2} next sentence pairs...")
	nsp_pairs = np.stack(
		(
			np.arange(start=0, stop=num_instances-1, step=2),
			np.arange(start=1, stop=num_instances, step=2)
		),
		axis=-1
	)
	nsp_labels = np.zeros(num_instances // 2, dtype=int)  # 0: matching sentences
	# sample pairs for which to shuffle the next sentence
	shuffle_idcs = random.sample(range(num_instances//2), k=num_instances//4)
	sorted_shuffle_idcs = sorted(shuffle_idcs)
	# reorder second sentence in each pair which should be shuffled
	nsp_pairs[sorted_shuffle_idcs, 1] = nsp_pairs[shuffle_idcs, 1]
	nsp_labels[sorted_shuffle_idcs] = 1

	return nsp_pairs, nsp_labels


#
# Main Function
#

def main():
	args = parse_arguments()

	# set up experiment directory and logging
	setup_logging(os.path.join(args.out_path, 'preprocess.log'))

	# stamp start time
	total_start_time = time.time()
	stage_start_time = time.time()
	logging.info(f"Started data pre-processing at {datetime.datetime.now()}.")

	# set random seeds
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.random.manual_seed(args.seed)
	logging.info(f"Set random seed to {args.seed}.")

	# load Wikipedia corpus
	sentencized_wiki_path = os.path.join(args.out_path, 'wiki-sentences.csv')
	# generate sentence-splitted Wikipedia articles
	if not os.path.exists(sentencized_wiki_path):
		dataset_wiki = load_dataset('wikipedia', '20220301.en')['train']
		logging.info(f"Loaded 'Wikipedia' dataset:\n{dataset_wiki}")
		sentencizer = Sentencizer()
		generate_sentencized_dataset(sentencizer, dataset_wiki, sentencized_wiki_path)
		dataset_wiki = None
		logging.info(f"Completed splitting 'Wikipedia' corpus into sentences in {time.time() - stage_start_time:.2f}s.")
	# load sentence-splitted Wikipedia dataset
	stage_start_time = time.time()
	dataset_wiki = load_dataset('csv', data_files={'train': sentencized_wiki_path})['train']
	logging.info(
		f"Loaded sentence-splitted 'Wikipedia' dataset from '{sentencized_wiki_path}' "
		f"in {time.time() - stage_start_time:.2f}s:\n"
		f"{dataset_wiki}"
	)

	# load BookCorpus
	stage_start_time = time.time()
	dataset_book = load_dataset('bookcorpus')['train']
	logging.info(f"Loaded 'BookCorpus' dataset in {time.time() - stage_start_time:.2f}s:\n{dataset_book}")

	# concatenate datasets
	dataset = concatenate_datasets([dataset_wiki, dataset_book])
	logging.info(f"Concatenated datasets into:\n{dataset}")

	# load sentence pairs and labels
	nsp_pairs_path = os.path.join(args.out_path, 'sentence-pairs.npy')
	nsp_labels_path = os.path.join(args.out_path, 'sentence-pair-labels.npy')
	if os.path.exists(nsp_pairs_path) and os.path.exists(nsp_labels_path):
		stage_start_time = time.time()
		nsp_pairs = np.load(nsp_pairs_path)
		nsp_labels = np.load(nsp_labels_path)
		logging.info(
			f"Loaded existing sentence pairs and data order from '{nsp_pairs_path}' and '{nsp_labels_path}' "
			f"in {time.time() - stage_start_time:.2f}s."
		)
	# generate sentence pairs ((current_sentence, next_sentence), num_instances/2)
	else:
		stage_start_time = time.time()
		nsp_pairs, nsp_labels = generate_sentence_pairs(dataset)
		logging.info(
			f"Sampled {nsp_pairs.shape[0] // 2} correct/incorrect next sentence pairs "
			f"in {time.time() - stage_start_time:.2f}s."
		)

		# generate random data instance order
		stage_start_time = time.time()
		pair_idcs = np.arange(nsp_pairs.shape[0])
		np.random.shuffle(pair_idcs)
		logging.info(f"Created random data order in {time.time() - stage_start_time:.2f}s.")

		# export NSP metadata
		nsp_pairs_path = os.path.join(args.out_path, 'sentence-pairs.npy')
		nsp_labels_path = os.path.join(args.out_path, 'sentence-pair-labels.npy')
		np.save(nsp_pairs_path, nsp_pairs[pair_idcs, :])
		np.save(nsp_labels_path, nsp_labels[pair_idcs])
		logging.info(f"Saved NSP metadata '{nsp_pairs_path}' and '{nsp_labels_path}'.")

	# load tokenizer
	tokenizer = AutoTokenizer.from_pretrained(f'google/multiberts-seed_{args.seed}-step_0k')
	logging.info(f"Loaded tokenizer:\n{tokenizer}")

	# load MLM data collator
	collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=.15)

	# open output file pointer
	output_path = os.path.join(args.out_path, 'books-wiki.json')
	file_mode = 'a' if args.continued else 'w'
	output_file = open(output_path, file_mode, encoding='utf8')

	# main processing loop
	stage_start_time = time.time()
	errors = []
	num_tokens, num_masked = 0, 0
	cursor = 0
	# set cursor to end of existing file
	if args.continued and (os.path.exists(output_path)):
		cursor = sum(1 for _ in open(output_path, 'r'))
	# iterate over all pairs in batches
	while cursor < nsp_pairs.shape[0]:
		print(f"\r[{cursor}/{nsp_pairs.shape[0]}] Applying tokenization and masking...", end='', flush=True)
		# set up batch range
		start_idx = cursor
		end_idx = min(start_idx + args.batch_size, nsp_pairs.shape[0])
		cursor = end_idx

		# retrieve current instances
		batch_idcs = np.arange(start_idx, end_idx)
		pairs = nsp_pairs[batch_idcs, :]
		s1_idcs, s2_idcs = pairs[:, 0], pairs[:, 1]
		sentences1 = [str(instance['text']) for instance in dataset.select(s1_idcs)]
		sentences2 = [str(instance['text']) for instance in dataset.select(s2_idcs)]

		# tokenize sentences and apply masking
		tok_inputs = [list(pair) for pair in zip(sentences1, sentences2)]
		try:
			tok_outputs = tokenizer(tok_inputs, padding=True)  # has to be padded, but not tensor due to collator stacking
			clt_outputs = collator(tok_outputs['input_ids'])
		except TypeError as error:
			logging.error(f"[Error] Unable to tokenize and prepare data instances [{start_idx}:{end_idx}]:")
			for sidx, (s1, s2) in enumerate(tok_inputs):
				logging.error(f"  {sidx:>4}: '{s1}', '{s2}'")
			errors.append((batch_idcs, error))
			logging.error(error)
			logging.info("Skipped batch due to error.")
			continue

		# gather output tensors
		cur_token_ids = clt_outputs['input_ids']
		cur_mlm_labels = clt_outputs['labels']
		cur_nsp_labels = nsp_labels[batch_idcs]
		cur_attentions = torch.tensor(tok_outputs['attention_mask'], dtype=torch.bool)
		assert cur_token_ids.shape[0] == cur_mlm_labels.shape[0] == cur_nsp_labels.shape[0], \
			f"[Error] Number of token IDs ({cur_token_ids.shape[0]}), " \
			f"MLM labels {cur_mlm_labels.shape[0]} and NSP labels {cur_nsp_labels.shape[0]} do not match."

		# export outputs to JSON objects
		json_output = []
		for idx in range(batch_idcs.shape[0]):
			# truncate tensors based on attention masks
			ids = cur_token_ids[idx, cur_attentions[idx]]
			mlm = cur_mlm_labels[idx, cur_attentions[idx]]
			nsp = cur_nsp_labels[idx]

			# ensure max 80 masked tokens
			masked_tokens_mask = (mlm != -100)
			masked_token_idcs = torch.where(masked_tokens_mask)[0]
			num_masked_tokens = torch.sum(masked_tokens_mask)
			if num_masked_tokens > 80:
				unmask_idcs = random.sample(masked_token_idcs.tolist(), k=(num_masked_tokens - 80))
				ids[unmask_idcs] = torch.tensor(tok_outputs['input_ids'], dtype=torch.long)[idx, unmask_idcs]
				mlm[unmask_idcs] = -100

			# convert to instance JSON object
			json_output.append(json.dumps({
				'token_ids': ids.tolist(),
				'mlm_labels': mlm.tolist(),
				'nsp_label': int(nsp)
			}))
			# store statistics
			num_tokens += ids.shape[0]
			num_masked += torch.sum(mlm != -100)
		output_file.write('\n'.join(json_output))
		if cursor < nsp_pairs.shape[0]:
			output_file.write('\n')
	logging.info(
		f"\rTokenized {nsp_pairs.shape[0]} sentence pairs into {num_tokens} tokens "
		f"and applied masking to {num_masked} tokens ({(num_masked * 100)/num_tokens:.2f}%) "
		f"in {time.time() - stage_start_time:.2f}s {'(continued)' if args.continued else ''} "
		f"and {len(errors)} error(s)."
	)

	# export errors for later reference
	if len(errors) > 0:
		errors_path = os.path.join(args.out_path, 'errors.pkl')
		with open(errors_path, 'wb') as fp:
			pickle.dump(errors_path, fp)
		logging.info(f"Saved error information to '{errors_path}'.")

	output_file.close()
	logging.info(f"Saved JSON dataset file to '{output_path}'.")
	logging.info(
		f"Completed data pre-processing at {datetime.datetime.now()} "
		f"({time.time() - total_start_time:.2f}s total)."
	)


if __name__ == '__main__':
	main()
