import argparse, csv, os

from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Stanford Natural Language Inference - Dataset Conversion')
	arg_parser.add_argument('output_path', help='prefix for CSV output corpora')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	print("Loading SNLI dataset from HuggingFace...")
	split_map = {'train': 'train', 'validation': 'dev', 'test': 'test'}

	snli = load_dataset('snli')
	for split_name, split in snli.items():
		# initialize outputs
		premises = []
		hypotheses = []
		labels = []

		# load label information
		label_names = split.features['label'].names

		# iterate over instances
		for idx, instance in enumerate(split):
			print(f"\r[{idx+1}/{len(split)}] Converting {split_name}-split...", end='', flush=True)
			premises.append(instance['premise'])
			hypotheses.append(instance['hypothesis'])
			labels.append(label_names[instance['label']])
		print("\r")

		assert len(premises) == len(hypotheses) == len(labels),\
			f"[Error] Unequal number of premises (N={len(premises)}), hypotheses (N={len(hypotheses)}) " \
			f"and labels (N={len(labels)}."

		# write to CSV
		split_path = os.path.join(args.output_path, f'{split_map[split_name]}.csv')
		with open(split_path, 'w', encoding='utf8', newline='') as output_file:
			csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
			csv_writer.writerow(['text0', 'text1', 'label'])
			csv_writer.writerows(zip(premises, hypotheses, labels))
		print(f"Saved {len(labels)} instances to '{split_path}'.")


if __name__ == '__main__':
	main()
