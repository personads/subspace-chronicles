import argparse, csv, os

from collections import defaultdict

from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='OntoNotes - Dataset Conversion')
	arg_parser.add_argument('output_path', help='prefix for CSV output corpora')
	arg_parser.add_argument('--split-domains', action='store_true', default=False, help='set flag to split by domain (nw, bn, mg) + (bc, tc, wd)')
	return arg_parser.parse_args()


def linearize_coreferences(sentence):
	spans = sentence['coref_spans']
	# gather coreferences by cluster ID
	clusters = defaultdict(list)
	for cluster_idx, start_idx, end_idx in spans:
		clusters[cluster_idx].append((start_idx, end_idx))

	# init coreference token labels as O
	labels = ['O' for _ in sentence['words']]
	num_corefs = 0
	# set coreferences to I
	for cluster_idx, positions in clusters.items():
		# check if coreference occurs within sentence
		if len(positions) < 2:
			continue
		for start_idx, end_idx in positions:
			labels[start_idx:end_idx + 1] = ['I' for _ in range(end_idx - start_idx + 1)]
		num_corefs += 1

	if num_corefs == 0:
		return []

	return labels


def save(path, inputs, labels):
	assert len(inputs) == len(labels), f"[Error] Unequal number of inputs and labels ({len(inputs)} ≠ {len(labels)})."

	# filter out inputs w/o corresponding labels
	dataset_rows = []
	for text, label in zip(inputs, labels):
		if len(label) < 1:
			continue
		dataset_rows.append((text, ' '.join(label)))
	print(f"Filtered {len(inputs)} to {len(dataset_rows)} with labels.")

	with open(path, 'w', encoding='utf8', newline='') as output_file:
		csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
		csv_writer.writerow(['text', 'label'])
		csv_writer.writerows(dataset_rows)
	print(f"Saved {len(dataset_rows)} instances to '{path}'.")


def main():
	args = parse_arguments()

	print("Loading OntoNotes 5.0 dataset from HuggingFace...")
	split_map = {'train': 'train', 'validation': 'dev', 'test': 'test'}
	domain_map = ['bn-mz-nw-pt', 'bc-tc-wb']
	domain_split = {'nw': 0, 'bn': 0, 'mz': 0, 'pt': 0, 'bc': 1, 'tc': 1, 'wb': 1}

	ontonotes = load_dataset('conll2012_ontonotesv5', 'english_v12')
	for split_name, split in ontonotes.items():
		# initialize outputs (one list per domain)
		inputs = [[], []]
		pos = [[], []]
		entities = [[], []]
		coreferences = [[], []]

		# load label information
		pos_labels = split.info.features['sentences'][0]['pos_tags'].feature.names
		ner_labels = split.info.features['sentences'][0]['named_entities'].feature.names

		# iterate over documents
		for doc_idx, document in enumerate(split):
			print(f"\r[{doc_idx}/{len(split)}] Converting {split_name}-split...", end='', flush=True)
			domain_idx = domain_split[document['document_id'].split('/')[0]]
			# iterate over sentences
			for sen_idx, sentence in enumerate(document['sentences']):
				# append raw, tokenized text
				inputs[domain_idx].append(' '.join(sentence['words']))

				# append PoS-tags as string labels
				pos[domain_idx].append([pos_labels[pos_idx] for pos_idx in sentence['pos_tags']])

				# append named entities
				entities[domain_idx].append([ner_labels[ent_idx] for ent_idx in sentence['named_entities']])

				# append coreference
				coreferences[domain_idx].append(linearize_coreferences(sentence))
		print("\r")

		# merge data when not splitting across domains
		if not args.split_domains:
			inputs = inputs[0] + inputs[1]
			pos = pos[0] + pos[1]
			entities = entities[0] + entities[1]
			coreferences = coreferences[0] + coreferences[1]
			
			os.makedirs(os.path.join(args.output_path, 'pos'), exist_ok=True)
			path = os.path.join(args.output_path, 'pos', f'{split_map[split_name]}.csv')
			save(path, inputs, pos)

			# save named entity data
			os.makedirs(os.path.join(args.output_path, 'ner'), exist_ok=True)
			path = os.path.join(args.output_path, 'ner', f'{split_map[split_name]}.csv')
			save(path, inputs, entities)

			# save coreference data
			os.makedirs(os.path.join(args.output_path, 'coref'), exist_ok=True)
			path = os.path.join(args.output_path, 'coref', f'{split_map[split_name]}.csv')
			save(path, inputs, coreferences)
		else:
			# iterate over domain splits
			for domain_idx, domain_name in enumerate(domain_map):
				# save PoS data
				os.makedirs(os.path.join(args.output_path, domain_name, 'pos'), exist_ok=True)
				path = os.path.join(args.output_path, domain_name, 'pos', f'{split_map[split_name]}.csv')
				save(path, inputs[domain_idx], pos[domain_idx])

				# save named entity data
				os.makedirs(os.path.join(args.output_path, domain_name, 'ner'), exist_ok=True)
				path = os.path.join(args.output_path, domain_name, 'ner', f'{split_map[split_name]}.csv')
				save(path, inputs[domain_idx], entities[domain_idx])

				# save coreference data
				os.makedirs(os.path.join(args.output_path, domain_name, 'coref'), exist_ok=True)
				path = os.path.join(args.output_path, domain_name, 'coref', f'{split_map[split_name]}.csv')
				save(path, inputs[domain_idx], coreferences[domain_idx])


if __name__ == '__main__':
	main()
