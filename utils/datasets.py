import csv
import sys

import numpy as np
import transformers.models.bert


class LabelledDataset:
    def __init__(self, inputs, labels):
        self._inputs = inputs  # List(List(Str)): [['t0', 't1', ...], ['t0', 't1', ...]] or List(Str): ['t0 t1 ... tN']
        self._labels = labels  # List(List(Str)): [['l0', 'l1', ...], ['l0', 'l1', ...]] or List(Str): ['l0', 'l1', ...]
        self._repeated_labels = None
        self._label_level = None

    def __len__(self):
        return len(list(self.get_flattened_labels()))

    def __repr__(self):
        return f'<LabelledDataset: {len(self._inputs)} inputs, {len(self)} labels ({self.get_label_level()}-level)>'

    def __getitem__(self, key):
        return self._inputs[key], self._labels[key]

    def get_inputs(self):
        return self._inputs

    def get_input_count(self):
        return len(self._inputs)

    def get_flattened_labels(self):
        for cur_labels in self._labels:
            if type(cur_labels) is list:
                for cur_label in cur_labels:
                    yield cur_label
            else:
                yield cur_labels

    def get_label_types(self):
        label_types = set()
        for label in self.get_flattened_labels():
            label_types.add(label)
        return sorted(label_types)

    def get_label_level(self):
        level = self._label_level
        if level is not None:
            return self._label_level

        for idx in range(self.get_input_count()):
            if type(self._labels[idx]) is list:
                if level is None:
                    level = 'token'
                elif level == 'sequence':
                    level = 'multiple'
                    break
            else:
                if level is None:
                    level = 'sequence'
                elif level == 'token':
                    level = 'multiple'
                    break
        self._label_level = level

        return level

    def get_batches(self, batch_size):
        cursor = 0
        while cursor < len(self._inputs):
            # set up batch range
            start_idx = cursor
            end_idx = min(start_idx + batch_size, len(self._inputs))
            cursor = end_idx
            num_remaining = len(self._inputs) - cursor

            # slice data
            inputs = self._inputs[start_idx:end_idx]

            # retrieve labels
            if self._repeated_labels is None:
                labels = self._labels[start_idx:end_idx]
            else:
                labels = self._repeated_labels[start_idx:end_idx]
            # flatten sequential labels if necessary
            if (self.get_label_level() == 'token') or (self._repeated_labels is not None):
                labels = [l for seq in labels for l in seq]

            # yield batch
            yield inputs, labels, num_remaining

    def get_shuffled_batches(self, batch_size):
        # start with list of all input indices
        remaining_idcs = list(range(len(self._inputs)))
        np.random.shuffle(remaining_idcs)

        # generate batches while indices remain
        while len(remaining_idcs) > 0:
            # pop-off relevant number of instances from pre-shuffled set of remaining indices
            batch_idcs = [remaining_idcs.pop() for _ in range(min(batch_size, len(remaining_idcs)))]

            # gather batch data
            inputs = [self._inputs[idx] for idx in batch_idcs]
            # flatten sequential labels if necessary
            if (self.get_label_level() == 'token') and (self._repeated_labels is None):
                labels = [l for idx in batch_idcs for l in self._labels[idx]]
            elif self._repeated_labels is not None:
                labels = [l for idx in batch_idcs for l in self._repeated_labels[idx]]
            # one label per input does not require flattening
            else:
                labels = [self._labels[idx] for idx in batch_idcs]
            # yield batch + number of remaining instances
            yield inputs, labels, len(remaining_idcs)

    def repeat_labels(self, encoder, batch_size, verbose=False):
        repeated_labels = []

        for inputs, labels, num_remaining in self.get_batches(batch_size):
            if verbose:
                print(
                    f"\x1b[1K\r[{(1 - (num_remaining / len(self._inputs))) * 100:.2f}%] "
                    f"Aligning {self.get_label_level()} labels to tokenization...",
                    end="", flush=True
                )
            # assume pre-tokenized space-separated input for token-level tasks
            if self.get_label_level() == 'token':
                tokenizer_batch = []
                for sequence in inputs:
                    # case: space-separated token sequence
                    if type(sequence) is str:
                        tokenizer_batch.append(sequence.split(' '))
                    # case: multiple space-separated token sequences
                    elif type(sequence) is tuple:
                        tokenizer_batch.append(tuple([s.split(' ') for s in sequence]))
                    else:
                        raise TypeError(f"Unable to repeat labels across sequence of type {type(sequence)}.")
                # pass pre-tokenized batch to model tokenizer
                tokenization = encoder.tokenize(tokenizer_batch, tokenized=True)
            # use encoder-specific tokenization for sentence-level tasks
            else:
                tokenization = encoder.tokenize(inputs)

            # iterate over inputs
            token_label_cursor = -1  # cursor over flattened batch labels
            for idx in range(len(inputs)):
                repeated_labels.append([])
                # convert token IDs to pieces
                pieces = encoder._tok.convert_ids_to_tokens(tokenization['input_ids'][idx])
                # iterate over subword tokens
                for tidx in range(tokenization['attention_mask'][idx].sum()):
                    # skip special tokens
                    if (not encoder._specials) and (tokenization['special_tokens_mask'][idx, tidx]):
                        continue
                    # case: token-level tasks repeat token labels across all subwords of the original token
                    if self.get_label_level() == 'token':
                        # check for start of new token
                        if tokenization['offset_mapping'][idx, tidx, 0] == 0:
                            # check for edge cases in which offsets do not align with original tokenization
                            # 1) sanity check BERT-style tokenizers for sub-words spanning one character
                            # example: one Hangul character decomposing into multiple parts all with offsets [0, 1]
                            if (
                                    isinstance(encoder._tok, transformers.models.bert.BertTokenizerFast)
                                    and
                                    pieces[tidx].startswith('##')
                            ):
                                token_label_cursor += 0
                            # 2) check for incorrect offset mapping in SentencePiece tokenizers (e.g. XLM-R)
                            # example: ',' -> '▁', ',' with [0, 1], [0, 1] which increment the label cursor prematurely
                            # https://github.com/huggingface/transformers/issues/9637
                            elif (tidx > 0) and (pieces[tidx - 1] == '▁'):
                                token_label_cursor += 0
                            # 3) TODO
                            elif (
                                    isinstance(encoder._tok, transformers.models.gpt2.GPT2TokenizerFast)
                                    and
                                    (not pieces[tidx].startswith('Ġ'))
                            ):
                                token_label_cursor += 0
                            # increment label cursor on new standard token
                            else:
                                token_label_cursor += 1
                        repeated_labels[-1].append(labels[token_label_cursor])
                    # case: sequence-level tasks repeat single sequence label across all sequence subwords
                    else:
                        repeated_labels[-1].append(labels[idx])

        # set internal labels to repeated
        self._repeated_labels = repeated_labels
        return repeated_labels

    def save(self, path):
        with open(path, 'w', encoding='utf8', newline='') as output_file:
            csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
            # construct header for single-/multi-sequence inputs + labels
            header = [f'text{i}' for i in range(len(self._inputs[0]))] if type(self._inputs[0]) is tuple else ['text']
            header.append('label')
            csv_writer.writerow(header)
            # iterate over instances
            for idx, text in enumerate(self._inputs):
                row = []
                # add one row each for multi-sequence inputs
                if type(text) is tuple:
                    row += list(text)
                else:
                    row.append(text)
                # convert token-level labels back to a single string
                label = self._labels[idx]
                if type(label) is list:
                    label = ' '.join([str(l) for l in label])
                row.append(label)
                # write row
                csv_writer.writerow(row)

    @staticmethod
    def from_path(path):
        inputs, labels = [], []
        label_level = 'sequence'
        with open(path, 'r', encoding='utf8', newline='') as fp:
            csv.field_size_limit(sys.maxsize)
            csv_reader = csv.DictReader(fp)
            for row in csv_reader:
                # convert all previous labels to token-level when encountering the first token-level label set
                if (' ' in row['label']) and (label_level != 'token'):
                    labels = [[l] for l in labels]
                    label_level = 'token'
                # covert current label(s) into appropriate form
                if label_level == 'token':
                    label = row['label'].split(' ')
                else:
                    label = row['label']
                # check if text consists of multiple sequences
                if len(csv_reader.fieldnames) > 2:
                    text = tuple([row[f'text{i}'] for i in range(len(csv_reader.fieldnames) - 1)])
                # otherwise, simply retrieve the text field
                else:
                    text = row['text']
                # append inputs and labels to overall dataset
                inputs.append(text)
                labels.append(label)

        return LabelledDataset(inputs, labels)