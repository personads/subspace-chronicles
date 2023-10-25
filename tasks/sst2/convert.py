import argparse, csv, os

from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Stanford Sentiment Treebank Binary - Dataset Conversion')
	arg_parser.add_argument('output_path', help='prefix for CSV output corpora')
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

	print("Loading SST-2 dataset from HuggingFace...")
	split_map = {'train': 'train', 'validation': 'dev', 'test': 'test'}

	sst = load_dataset('sst2')
	for split_name, split in sst.items():
		# initialize outputs
		inputs = []
		labels = []

		# load label information
		label_names = split.features['label'].names

		# iterate over instances
		for idx, instance in enumerate(split):
			print(f"\r[{idx}/{len(split)}] Converting {split_name}-split...", end='', flush=True)
			inputs.append(instance['sentence'].strip())  # remove trailing space
			labels.append(label_names[instance['label']])
		print("\r")

		path = os.path.join(args.output_path, f'{split_map[split_name]}.csv')
		save(path, inputs, labels)


if __name__ == '__main__':
	main()
