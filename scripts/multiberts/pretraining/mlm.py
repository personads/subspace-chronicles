#!/usr/bin/python3

import argparse, datetime, json, logging, os, sys

import torch

from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForPreTraining, get_polynomial_decay_schedule_with_warmup

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.setup import setup_experiment


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='MultiBERTs MLM Training')

	# data setup
	arg_parser.add_argument(
		'--data-path', required=True,
		help='path to pre-processed JSON data')

	# model setup
	arg_parser.add_argument(
		'--seed', type=int, required=True, choices=list(range(5)),
		help='random seed for model selection and random components')

	# experiment setup
	arg_parser.add_argument(
		'--exp-path', required=True,
		help='path to experiment directory')
	arg_parser.add_argument(
		'--batch-size', type=int, default=256,
		help='batch size (default: 256)')
	arg_parser.add_argument(
		'--max-steps', type=int, default=int(4e4),
		help='maximum number of update steps (default: 40k)')
	arg_parser.add_argument(
		'--checkpoints', nargs='+',
		default=[10] + list(range(100, 1000, 100)) + list(range(1000, 20000, 1000)) + list(range(20000, 60000, 20000)),
		help='steps at which to export model checkpoints and statistics'
	)

	return arg_parser.parse_args()


def prepare_model_inputs(batch, max_model_length, pad_id, sep_id):
	# load appropriate non-tensor data from dataset batch
	token_ids = batch['token_ids']
	mlm_labels = batch['mlm_labels']
	nsp_labels = batch['nsp_label']

	# set up initial output tensors
	num_seqs = len(token_ids)
	max_len = min(max([len(seq_tok_ids) for seq_tok_ids in token_ids]), max_model_length)
	model_inputs = {
		'input_ids': torch.full((num_seqs, max_len), pad_id, dtype=torch.long),
		'attention_mask': torch.zeros((num_seqs, max_len), dtype=torch.double),
		'token_type_ids': torch.zeros((num_seqs, max_len), dtype=torch.long)
	}
	targets = {
		'labels': torch.full((num_seqs, max_len), -100, dtype=torch.long),
		'next_sentence_label': torch.tensor(nsp_labels, dtype=torch.long)
	}

	# fill tensors
	for seq_idx in range(num_seqs):
		seq_token_ids = torch.tensor(token_ids[seq_idx])
		seq_mlm_labels = torch.tensor(mlm_labels[seq_idx])
		seq_len = seq_token_ids.shape[0]
		# identify segment boundaries that have not been randomly replaced
		seg1_end_idx, seg2_end_idx = torch.where(torch.logical_and(seq_token_ids == sep_id, seq_mlm_labels == -100))[0]
		# truncate sequences exceeding maximum model length
		if seq_len > max_model_length:
			seq_token_ids = seq_token_ids[:max_model_length-1]
			seq_token_ids[-1] = sep_id  # re-add [SEP] at end
			seq_mlm_labels = seq_mlm_labels[:max_model_length-1]
			seq_mlm_labels[-1] = -100
			# update sequence length and boundary positions
			seq_len = seq_token_ids.shape[0]
			seg2_end_idx = seq_len - 1
			# check for truncation excluding second segment
			if seg1_end_idx >= (seq_len - 1):
				seg1_end_idx = seq_len - 1
				targets['next_sentence_label'][seq_idx] = 0

		# fill token IDs
		model_inputs['input_ids'][seq_idx, :seq_len] = seq_token_ids
		# fill attention mask
		model_inputs['attention_mask'][seq_idx, :seq_len] = 1
		# fill segment IDs
		model_inputs['token_type_ids'][seq_idx, seg1_end_idx+1:seq_len] = 1

		# fill MLM labels
		targets['labels'][seq_idx, :seq_len] = seq_mlm_labels

	# move tensors to appropriate device
	if torch.cuda.is_available():
		model_inputs = {k: v.to(torch.device('cuda')) for k, v in model_inputs.items()}
		targets = {k: v.to(torch.device('cuda')) for k, v in targets.items()}

	return model_inputs, targets


def compute_loss(criterion, model_outputs, targets):
	# adapted from
	# https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/models/bert/modeling_bert.py#L1140
	# to return losses separately
	mlm_loss = criterion(
		model_outputs.prediction_logits.view(-1, model_outputs.prediction_logits.shape[-1]),
		targets['labels'].view(-1)
	)
	nsp_loss = criterion(
		model_outputs.seq_relationship_logits.view(-1, 2),
		targets['next_sentence_label'].view(-1)
	)
	total_loss = mlm_loss + nsp_loss
	return total_loss, mlm_loss, nsp_loss


def main():
	args = parse_arguments()

	# set up experiment directory and logging
	setup_experiment(args.exp_path)

	logging.info("MultiBERTs Masked Language Modeling Training")
	logging.info(f"Started at {datetime.datetime.now()}.")
	logging.info(f"Checkpointing schedule at {len(args.checkpoints)} steps:\n{args.checkpoints}")
	checkpoints = set(args.checkpoints)

	# set random seed (for dropout etc.)
	torch.random.manual_seed(args.seed)
	logging.info(f"Set random seed to {args.seed}.")

	# load pre-processed dataset
	dataset = load_dataset('json', data_files={'train': args.data_path})['train']
	logging.info(f"Loaded pre-processed dataset from '{args.data_path}':\n{dataset}")

	# set up initial model
	tokenizer = AutoTokenizer.from_pretrained(f'google/multiberts-seed_{args.seed}-step_0k')
	logging.info(f"Loaded tokenizer:\n{tokenizer}")
	lm_name = f'google/multiberts-seed_{args.seed}-step_0k'
	model = AutoModelForPreTraining.from_pretrained(lm_name)
	logging.info(f"Loaded initial LM checkpoint:\n{model}")
	if torch.cuda.is_available():
		model = model.to(torch.device('cuda'))
		logging.info(f"Moved LM to {model.device}.")

	# set up optimizer
	optimizer = torch.optim.AdamW(
		params=model.parameters(),
		lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-2
	)
	logging.info(f"Set up {optimizer.__class__.__name__} optimizer:\n{optimizer}.")
	# set up scheduler
	scheduler = get_polynomial_decay_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=int(1e4), num_training_steps=int(2e6)
	)
	logging.info(
		f"Set up scheduler with polynomial decay across {int(2e6)} steps and "
		f"warmup over {int(1e4)} steps."
	)
	# set up gradient accumulation
	assert args.batch_size <= 256, f"[Error] Batch size {args.batch_size} > 256."
	assert 256 % args.batch_size == 0, f"[Error] Batch size {args.batch_size} must be divisor of 256."
	accumulation_steps = 256 // args.batch_size

	# set up loss
	criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
	logging.info(f"Set up criterion: {criterion.__class__.__name__}.")

	# main training loop
	statistics = defaultdict(list)
	cursor = 0
	for step in range(args.max_steps):
		statistics['loss'].append(0)
		statistics['mlm_loss'].append(0)
		statistics['nsp_loss'].append(0)
		# iterate over batches until gradients have accumulated
		acc_step = 0
		while (acc_step == 0) or (acc_step % accumulation_steps != 0):
			# set up batch
			start_idx = cursor
			end_idx = min(start_idx + args.batch_size, dataset.num_rows)
			cursor = end_idx if end_idx < dataset.num_rows else 0
			batch = dataset[start_idx:end_idx]

			# generate model inputs
			model_inputs, targets = prepare_model_inputs(
				batch,
				max_model_length=model.config.max_position_embeddings,
				pad_id=tokenizer.pad_token_id, sep_id=tokenizer.sep_token_id
			)
			model_outputs = model(**model_inputs)

			# compute loss
			loss, mlm_loss, nsp_loss = compute_loss(criterion, model_outputs, targets)

			# perform backward pass
			loss /= accumulation_steps
			loss.backward()

			# update statistics
			acc_step += 1
			statistics['loss'][-1] += float(loss.detach())
			statistics['mlm_loss'][-1] += float(mlm_loss.detach() / accumulation_steps)
			statistics['nsp_loss'][-1] += float(nsp_loss.detach() / accumulation_steps)

			# print batch statistics
			step_info = ''
			if step > 0:
				step_info = f"Loss: {statistics['loss'][step-1]:.4f} = " \
					f"{statistics['mlm_loss'][step-1]:.4f} (MLM) + {statistics['nsp_loss'][step-1]:.4f} (NSP) / "
			print(
				f"\x1b[1K\r[Step {step+1}-{acc_step:<2} / {args.max_steps}] {step_info}"
				f"AccLoss: {loss:.4f}",
				end='', flush=True
			)

		# perform optimizer step with accumulated gradients
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()

		# prepare checkpoint export
		if (step+1 in checkpoints) or (step+1 == args.max_steps):
			print()
			checkpoint_path = os.path.join(args.exp_path, f'step-{step+1}.tar')
			torch.save({
				'step': step,
				'language_model_name': lm_name,
				'language_model': {k.replace('bert.', ''): v for k, v in model.state_dict().items()},
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'loss': statistics['loss'][step],
				'data_cursor': cursor
			}, checkpoint_path)
			logging.info(f"Exported model checkpoint to '{checkpoint_path}'.")
			statistics_path = os.path.join(args.exp_path, f'statistics.json')
			with open(statistics_path, 'w') as fp:
				json.dump(statistics, fp, indent=4, sort_keys=True)
			logging.info(f"Exported training statistics to '{statistics_path}'.")

	logging.info(f"Completed training for {args.max_steps} steps at {datetime.datetime.now()}.")


if __name__ == '__main__':
	main()
