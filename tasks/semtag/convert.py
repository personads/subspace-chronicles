#!/usr/bin/python3

import argparse, csv, os, re

import urllib.request

# PMB formatted as CONLL (van Noord et al., 2018)
REMOTE_ROOT = 'https://raw.githubusercontent.com/RikVN/DRS_parsing/master/parsing/layer_data/4.0.0/en/gold/'


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Parallel Meaning Bank Semantic Tags - Dataset Conversion')
	arg_parser.add_argument('output_path', help='output directory for CSV corpus')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	splits = ['train', 'dev', 'test', 'eval']

	for split in splits:
		remote_url = REMOTE_ROOT + split + '.conll'
		print(f"Downloading data from '{remote_url}'...")
		response = urllib.request.urlopen(remote_url)
		data = response.read()
		conll = data.decode('utf-8')

		texts = []
		labels = []

		# process CONLL data line-by line
		cur_tokens, cur_labels = [], []
		lines = conll.split('\n')
		for idx, line in enumerate(lines):
			print(f"\r[{idx + 1}/{len(lines)}] Processing CoNLL...", end='', flush=True)
			# skip comments
			if line.startswith('#'):
				continue
			# check for end of sentence
			if line == '':
				if len(cur_tokens) > 0:
					texts.append(' '.join(cur_tokens))
					labels.append(' '.join(cur_labels))
					cur_tokens, cur_labels = [], []
				continue
			# get tab-separated values
			parts = line.split('\t')
			label = parts[3]
			# append token (potentially multi-token separated by tilde)
			for subtoken in parts[0].split('~'):
				cur_tokens.append(subtoken)
				cur_labels.append(label)
		# append last entry
		if len(cur_tokens) > 0:
			texts.append(' '.join(cur_tokens))
			labels.append(' '.join(cur_labels))
		print('\r')

		assert len(texts) == len(labels),\
			f"[Error] Unequal number of inputs (N={len(texts)}) and labels (N={len(labels)}."

		# write to CSV
		split_path = os.path.join(args.output_path, f'{split}.csv')
		with open(split_path, 'w', encoding='utf8', newline='') as output_file:
			csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
			csv_writer.writerow(['text', 'label'])
			csv_writer.writerows(zip(texts, labels))
		print(f"Saved {len(labels)} instances to '{split_path}'.")


if __name__ == '__main__':
	main()
