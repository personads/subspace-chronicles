import argparse, csv, os

from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='TweetEval - Dataset Conversion')
	arg_parser.add_argument('--task', choices=['sentiment'], help='select task to export')
	arg_parser.add_argument('--out-path', help='prefix for CSV output corpora')
	arg_parser.add_argument('--exclude-neutral', action='store_true', default=False, help='set flag to exclude neutral labels')
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

	print("Loading TweetEval dataset from HuggingFace...")
	split_map = {'train': 'train', 'validation': 'dev', 'test': 'test'}

	sst = load_dataset('tweet_eval', args.task)
	for split_name, split in sst.items():
		# initialize outputs
		inputs = []
		labels = []

		# load label information
		label_names = split.features['label'].names

		# iterate over instances
		for idx, instance in enumerate(split):
			print(f"\r[{idx}/{len(split)}] Converting {split_name}-split...", end='', flush=True)
			if (args.task == 'sentiment') and args.exclude_neutral and (instance['label'] == 1):
				continue
			inputs.append(instance['text'].strip())  # remove trailing space
			labels.append(label_names[instance['label']])
		print("\r")

		path = os.path.join(args.out_path, f'{split_map[split_name]}.csv')
		save(path, inputs, labels)


if __name__ == '__main__':
	main()
