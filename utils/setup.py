import logging, os, re, sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.classifiers import *
from models.losses import *


def setup_model_arguments(arg_parser):
	# language model setup
	arg_parser.add_argument(
		'--model-name', required=True,
		help='embedding model identifier')
	arg_parser.add_argument(
		'--model-revision', default='main',
		help='embedding model revision identifier (default: main)')
	arg_parser.add_argument(
		'--encoder-path',
		help='path to local pre-trained encoder checkpoint (default: None | "exp-path/best.pt" in prediction mode)')
	arg_parser.add_argument(
		'--embedding-caching', action='store_true', default=False,
		help='flag to activate RAM embedding caching (default: False)')
	arg_parser.add_argument(
		'--embedding-tuning', action='store_true', default=False,
		help='set flag to tune the full model including embeddings (default: False)')
	arg_parser.add_argument(
		'--embedding-pooling', choices=['first', 'mean'],
		help='embedding pooling strategy (default: None)')

	# layer selector setup
	arg_parser.add_argument(
		'--layers', required=True,
		help='layer selection strategy')

	# classifier setup
	arg_parser.add_argument(
		'--classifier', required=True,
		help='classifier identifier')

	return arg_parser


def setup_experiment(out_path, prediction=False):
	if not os.path.exists(out_path):
		if prediction:
			print(f"Experiment path '{out_path}' does not exist. Cannot run prediction. Exiting.")
			exit(1)

		# if output dir does not exist, create it (new experiment)
		print(f"Path '{out_path}' does not exist. Creating...")
		os.mkdir(out_path)
	# if output dir exist, check if predicting
	else:
		# if not predicting, verify overwrite
		if not prediction:
			response = None

			while response not in ['y', 'n']:
				response = input(f"Path '{out_path}' already exists. Overwrite? [y/n] ")
			if response == 'n':
				exit(1)

	# setup logging
	setup_logging(os.path.join(out_path, 'classify.log'))


def setup_logging(path):
	log_format = '%(message)s'
	log_level = logging.INFO
	logging.basicConfig(filename=path, filemode='a', format=log_format, level=log_level)

	logger = logging.getLogger()
	logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_layers(layers_id):
	# set up layer selection definitions
	layers_match = re.match(r'(?P<name>[a-z]+?)(?P<args>\((\d+,? *)*\))', layers_id, flags=re.IGNORECASE)
	args_pattern = re.compile(r'(\d+),?')

	# check if layer syntax is correct
	if layers_match:
		layers_args = [int(a) for a in args_pattern.findall(layers_match['args'])]
		# parse all
		if layers_match['name'] == 'all':
			if len(layers_args) != 1:
				logging.error(f"[Error] all layers selector requires one argument 'num_layers'. Exiting.")
				exit(1)
			return torch.ones(layers_args[0], dtype=torch.bool)
		# parse mask
		elif layers_match['name'] == 'mask':
			if len(layers_args) < 1:
				logging.error(f"[Error] mask layer selector requires list of mask values. Exiting.")
				exit(1)
			return (torch.tensor(layers_args) > 0)
		# parse autolayer
		elif layers_match['name'] == 'auto':
			if len(layers_args) != 1:
				logging.error(f"[Error] auto layers selector requires one argument 'num_layers'. Exiting.")
				exit(1)
			return torch.ones(layers_args[0])
		else:
			logging.error(f"[Error] Unknown layer selector '{layers_match['name']}'. Exiting.")
			exit(1)
	else:
		logging.error(f"[Error] Could not parse layer selector '{layers_id}'. Exiting.")
		exit(1)


def setup_pooling(identifier):
	if identifier == 'mean':
		return get_mean_embedding
	elif identifier == 'first':
		return get_first_embedding
	else:
		raise ValueError(f"[Error] Unknown pooling specification '{identifier}'.")


def setup_encoder(
		model_name, model_revision, layers_id,
		encoder_path=None, model_path=None, emb_caching=None, emb_pooling=None, emb_tuning=False, lyr_pooling=False):
	# set up layer selector
	lyr_selector = setup_layers(layers_id)
	lyr_tuning = layers_id.startswith('auto')
	logging.info(f"Loaded layer selector '{layers_id}'.")

	# set up embedding pooling
	pooling_strategy = None if emb_pooling is None else setup_pooling(emb_pooling)
	if pooling_strategy is None:
		logging.info(f"Subword embedding pooling inactive.")
	else:
		logging.info(f"Loaded embedding pooling function '{emb_pooling}'.")

	# load spectral encoder
	if model_path is None:
		encoder = Encoder(
			model_name=model_name,
			lyr_selector=lyr_selector, lyr_tuning=lyr_tuning, lyr_pooling=lyr_pooling,
			emb_tuning=emb_tuning, emb_pooling=pooling_strategy, model_revision=model_revision,
			cache=({} if emb_caching else None))
	else:
		encoder = Encoder.load(
			model_path,
			lyr_selector=lyr_selector, lyr_tuning=lyr_tuning, lyr_pooling=lyr_pooling,
			emb_tuning=emb_tuning, emb_pooling=pooling_strategy,
			cache=({} if emb_caching else None)
		)
		logging.info(f"Loaded pre-trained {encoder.__class__.__name__} from '{model_path}'.")

	# load locally trained LM
	if encoder_path is not None:
		encoder.load_language_model(encoder_path)
		logging.info(f"Loaded locally trained LM from '{encoder_path}'.")

	return encoder


def setup_classifier(classifier_id, encoder, train_data, valid_data, classes, model_path=None):
	# set up layer selection definitions
	classifier_match = re.match(r'(?P<name>[a-z/]+)(?P<args>\((\d+,? *)*\))?', classifier_id, flags=re.IGNORECASE)
	args_pattern = re.compile(r'(\d+),?')

	# check if classifier syntax is correct
	if classifier_match:
		classifier_args = [int(a) for a in args_pattern.findall(classifier_match['args'])] if classifier_match['args'] else []
		# parse linear classifier
		if classifier_match['name'] == 'linear':
			if len(classifier_args) != 0:
				logging.error(f"[Error] LinearClassifier takes no arguments. Exiting.")
				exit(1)
			classifier = LinearClassifier(emb_model=encoder, classes=classes)
			train_loss = LabelLoss(dataset=train_data, classes=classes) if train_data is not None else None
			valid_loss = LabelLoss(dataset=valid_data, classes=classes)
		# parse linear MDL classifier
		elif classifier_match['name'] == 'mdl/linear':
			if len(classifier_args) != 0:
				logging.error(f"[Error] MdlLinearClassifier takes no arguments. Exiting.")
				exit(1)
			classifier = MdlLinearClassifier(emb_model=encoder, classes=classes)
			train_loss = MdlLoss(dataset=train_data, classes=classes) if train_data is not None else None
			valid_loss = MdlLoss(dataset=valid_data, classes=classes)
		else:
			logging.error(f"[Error] Unknown classifier '{classifier_match['name']}'. Exiting.")
			exit(1)
	else:
		logging.error(f"[Error] Could not parse classifier '{classifier_id}'. Exiting.")
		exit(1)

	# load pre-trained model if available
	if model_path is not None:
		# load classifier parameters
		classifier = classifier.load(
			model_path, emb_model=encoder
		)
		# align classifier labels with losses (in case dataset has changed)
		if train_loss is not None:
			train_loss.__init__(dataset=train_data, classes=classifier._classes)
		valid_loss.__init__(dataset=valid_data, classes=classifier._classes)
		logging.info(f"Loaded pre-trained {classifier.__class__.__name__} from '{model_path}'.")
		if set(classifier._classes) < set(classes):
			logging.warning(
				f"[Warning] Data contains labels unseen by the classifier: "
				f"{', '.join(sorted(set(classes) - set(classifier._classes)))}."
				f"These will be ignored during loss calculation."
			)

	return classifier, train_loss, valid_loss
