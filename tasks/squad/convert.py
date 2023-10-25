import argparse, csv, os

import transformers

from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Stanford Question Answering Dataset - Dataset Conversion')
	arg_parser.add_argument('tokenizer', help='name of HuggingFace tokenizer')
	arg_parser.add_argument('output_path', help='prefix for CSV output corpora')
	arg_parser.add_argument('--version', type=int, default=1, help='SQuAD version number (default: 1)')
	return arg_parser.parse_args()


def save(path, inputs, labels):
	assert len(inputs) == len(labels), f"[Error] Unequal number of inputs and labels ({len(inputs)} ≠ {len(labels)})."

	with open(path, 'w', encoding='utf8', newline='') as output_file:
		csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
		csv_writer.writerow(['text', 'label'])
		csv_writer.writerows(zip(inputs, labels))
	print(f"Saved {len(inputs)} instances to '{path}'.")


def main():
	args = parse_arguments()

	tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
	print(f"Loaded '{args.tokenizer}' ({tokenizer.__class__.__name__}).")

	print(f"Loading SQuAD v{args.version} dataset from HuggingFace...")
	split_map = {'train': 'train', 'validation': 'dev'}

	squad = load_dataset('squad') if args.version == 1 else load_dataset(f'squad_v{args.version}')
	for split_name, split in squad.items():
		# initialize outputs
		questions = []
		contexts = []
		labels = []

		# iterate over instances
		for idx, instance in enumerate(split):
			print(f"\r[{idx+1}/{len(split)}] Converting {split_name}-split...", end='', flush=True)

			# perform simple tokenization
			tok_question = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(instance['question'])
			tok_context = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(instance['context'])

			# gather answer spans
			answer_spans = []
			for start, answer in zip(instance['answers']['answer_start'], instance['answers']['text']):
				answer_spans.append((start, start + len(answer)))

			# generate label sequence
			cur_labels = ['O' for _ in range(len(tok_question))]
			for token, (tok_start, tok_end) in tok_context:
				# check if current token span is within an answer span
				in_answer = False
				for ans_start, ans_end in answer_spans:
					# token is in answer if its start and end are within the answer span
					if (tok_start >= ans_start) and (tok_end <= ans_end):
						in_answer = True
						break
				if in_answer:
					cur_labels.append('I')
				else:
					cur_labels.append('O')

			assert len(tok_question) + len(tok_context) == len(cur_labels), \
				f"[Error] Number of sequence labels does not match length of inputs " \
				f"({len(tok_question)} + {len(tok_context)} ≠ {len(cur_labels)})."

			questions.append(' '.join([t for t, s, in tok_question]))
			contexts.append(' '.join([t for t, s, in tok_context]))
			labels.append(' '.join(cur_labels))

		print("\r")

		assert len(questions) == len(contexts) == len(labels),\
			f"[Error] Unequal number of questions (N={len(questions)}), contexts (N={len(contexts)}) " \
			f"and labels (N={len(labels)}."

		# write to CSV
		split_path = os.path.join(args.output_path, f'{split_map[split_name]}.csv')
		with open(split_path, 'w', encoding='utf8', newline='') as output_file:
			csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
			csv_writer.writerow(['text0', 'text1', 'label'])
			csv_writer.writerows(zip(questions, contexts, labels))
		print(f"Saved {len(labels)} instances to '{split_path}'.")


if __name__ == '__main__':
	main()
