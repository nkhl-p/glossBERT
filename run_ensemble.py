# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict
import csv
import logging
import os
import random
import sys
import pandas as pd
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from modeling import BertForSequenceClassification, BertConfig, BertForTokenClassification
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear

from run_classifier_WSD_baseline import WSDProcessor
from run_classifier_WSD_sent import WSD_sent_Processor


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# class DataProcessor(object):
#     """Base class for data converters for sequence classification data sets."""
#
#     def get_train_examples(self, data_dir):
#         """Gets a collection of `InputExample`s for the train set."""
#         raise NotImplementedError()
#
#     def get_dev_examples(self, data_dir):
#         """Gets a collection of `InputExample`s for the dev set."""
#         raise NotImplementedError()
#
#     def get_test_examples(self, data_dir):
#         """Gets a collection of `InputExample`s for the test set."""
#         raise NotImplementedError()
#
#     def get_labels(self):
#         """Gets the list of labels for this data set."""
#         raise NotImplementedError()
#
#     @classmethod
#     def _read_tsv(cls, input_file, quotechar=None):
#         """Reads a tab separated value file."""
#         with open(input_file, "r") as f:
#             reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#             lines = []
#             for line in reader:
#                 lines.append(line)
#             return lines
#
# class WSD_sent_Processor(DataProcessor):
#     """Processor for the WSD data set."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         train_data = pd.read_csv(data_dir, sep="\t", na_filter=False).values
#         return self._create_examples(train_data, "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         dev_data = pd.read_csv(data_dir, sep="\t", na_filter=False).values
#         return self._create_examples(dev_data, "dev")
#
#     def get_labels(self):
#         """See base class."""
#         return ["0","1"]
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             text_a = str(line[2])
#             text_b = str(line[3])
#             label = str(line[1])
#             if i%1000==0:
#                 print(i)
#                 print("guid=",guid)
#                 print("text_a=",text_a)
#                 print("text_b=",text_b)
#                 print("label=",label)
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["WSD"],
                        help="The name of the task to train.")
    parser.add_argument("--train_data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--eval_data_dir_baseline",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_data_dir_wsd_sent_cls",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_data_dir_wsd_token",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_data_dir_wsd_sent_ws",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--label_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The label data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--bert_model_baseline", default=None, type=str, required=True,
                        help='''a path or url to a pretrained model archive containing:
                        'bert_config.json' a configuration file for the model
                        'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')
    parser.add_argument("--bert_model_wsd_sent_cls", default=None, type=str, required=True,
                        help='''a path or url to a pretrained model archive containing:
                        'bert_config.json' a configuration file for the model
                        'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')
    parser.add_argument("--bert_model_wsd_token", default=None, type=str, required=True,
                    help='''a path or url to a pretrained model archive containing:
                    'bert_config.json' a configuration file for the model
                    'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')
    parser.add_argument("--bert_model_wsd_sent_ws", default=None, type=str, required=True,
                        help='''a path or url to a pretrained model archive containing:
                        'bert_config.json' a configuration file for the model
                        'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')


    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")
    if args.do_train:
        assert args.train_data_dir != None, "train_data_dir can not be None"
    if args.do_eval:
        assert args.eval_data_dir != None, "eval_data_dir can not be None"

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataloaders
    # processors = {
    #     "WSD":WSD_sent_Processor
    # }

    output_modes = {
        "WSD": "classification"
    }

    processor_WSD_sent_Processor = WSD_sent_Processor()
    processor_WSDProcessor = WSDProcessor()
    output_mode = output_modes[args.task_name]
    label_list = processor_WSD_sent_Processor.get_labels()
    label_list_token = processor_WSDProcessor.get_labels(args.label_data_dir)
    num_labels = len(label_list)

    tokenizer_baseline = BertTokenizer.from_pretrained(args.bert_model_baseline, do_lower_case=args.do_lower_case)
    tokenizer_wsd_token = BertTokenizer.from_pretrained(args.bert_model_wsd_token, do_lower_case=args.do_lower_case)
    tokenizer_wsd_sent_ws = BertTokenizer.from_pretrained(args.bert_model_wsd_sent_ws, do_lower_case=args.do_lower_case)
    tokenizer_wsd_sent_cls = BertTokenizer.from_pretrained(args.bert_model_wsd_sent_cls, do_lower_case=args.do_lower_case)


    # training set
    train_examples = None
    num_train_optimization_steps = None
    # if args.do_train:
    #     train_examples = processor.get_train_examples(args.train_data_dir)
    #     num_train_optimization_steps = int(
    #         len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    #     if args.local_rank != -1:
    #         num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()


    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model_baseline = BertForTokenClassification.from_pretrained(args.bert_model_baseline,
              cache_dir=cache_dir,
              num_labels=len(label_list_token))
    model_wsd_token = BertForSequenceClassification.from_pretrained(args.bert_model_wsd_token,
              cache_dir=cache_dir,
              num_labels=num_labels)
    model_wsd_sent_ws = BertForSequenceClassification.from_pretrained(args.bert_model_wsd_sent_ws,
              cache_dir=cache_dir,
              num_labels=num_labels)
    model_wsd_sent_cls = BertForSequenceClassification.from_pretrained(args.bert_model_wsd_sent_cls,
              cache_dir=cache_dir,
              num_labels=num_labels)


    if args.fp16:
        model_baseline.half()
        model_wsd_token.half()
        model_wsd_sent_ws.half()
        model_wsd_sent_cls.half()
    model_baseline.to(device)
    model_wsd_token.to(device)
    model_wsd_sent_ws.to(device)
    model_wsd_sent_cls.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model_baseline = DDP(model_baseline)
        model_wsd_token = DDP(model_wsd_token)
        model_wsd_sent_ws = DDP(model_wsd_sent_ws)
        model_wsd_sent_cls = DDP(model_wsd_sent_cls)

    elif n_gpu > 1:
        model_baseline = torch.nn.DataParallel(model_baseline)
        model_wsd_token = torch.nn.DataParallel(model_wsd_token)
        model_wsd_sent_ws = torch.nn.DataParallel(model_wsd_sent_ws)
        model_wsd_sent_cls = torch.nn.DataParallel(model_wsd_sent_cls)


    # Prepare optimizer
    # param_optimizer = list(model_baseline.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    # if args.fp16:
    #     try:
    #         from apex.optimizers import FP16_Optimizer
    #         from apex.optimizers import FusedAdam
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #
    #     optimizer = FusedAdam(optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           bias_correction=False,
    #                           max_grad_norm=1.0)
    #     if args.loss_scale == 0:
    #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    #     else:
    #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    #
    # else:
    #     optimizer = BertAdam(optimizer_grouped_parameters,
    #                          lr=args.learning_rate,
    #                          warmup=args.warmup_proportion,
    #                          t_total=num_train_optimization_steps)



    # load data
    # if args.do_train:
    #     train_features = convert_examples_to_features(
    #         train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    #     logger.info("***** Running training *****")
    #     logger.info("  Num examples = %d", len(train_examples))
    #     logger.info("  Batch size = %d", args.train_batch_size)
    #     logger.info("  Num steps = %d", num_train_optimization_steps)
    #     all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    #
    #     if output_mode == "classification":
    #         all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    #     elif output_mode == "regression":
    #         all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
    #
    #     train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    #     if args.local_rank == -1:
    #         train_sampler = RandomSampler(train_data)
    #     else:
    #         train_sampler = DistributedSampler(train_data)
    #     train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


    # if args.do_eval:
    #     eval_examples = processor.get_dev_examples(args.eval_data_dir)
    #     eval_features = convert_examples_to_features(
    #         eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    #     logger.info("***** Running evaluation *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #
    #     if output_mode == "classification":
    #         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    #     elif output_mode == "regression":
    #         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
    #
    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    #     eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)




    # train
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    # if args.do_train:
    #     model.train()
    #     epoch = 0
    #     for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    #         epoch += 1
    #         tr_loss = 0
    #         nb_tr_examples, nb_tr_steps = 0, 0
    #         for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    #             batch = tuple(t.to(device) for t in batch)
    #             input_ids, input_mask, segment_ids, label_ids = batch
    #
    #             logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
    #
    #             if output_mode == "classification":
    #                 loss_fct = CrossEntropyLoss()
    #                 loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    #             elif output_mode == "regression":
    #                 loss_fct = MSELoss()
    #                 loss = loss_fct(logits.view(-1), label_ids.view(-1))
    #
    #             if n_gpu > 1:
    #                 loss = loss.mean() # mean() to average on multi-gpu.
    #             if args.gradient_accumulation_steps > 1:
    #                 loss = loss / args.gradient_accumulation_steps
    #
    #             if args.fp16:
    #                 optimizer.backward(loss)
    #             else:
    #                 loss.backward()
    #
    #             tr_loss += loss.item()
    #             nb_tr_examples += input_ids.size(0)
    #             nb_tr_steps += 1
    #             if (step + 1) % args.gradient_accumulation_steps == 0:
    #                 if args.fp16:
    #                     # modify learning rate with special warm up BERT uses
    #                     # if args.fp16 is False, BertAdam is used that handles this automatically
    #                     lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
    #                     for param_group in optimizer.param_groups:
    #                         param_group['lr'] = lr_this_step
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #                 global_step += 1
    #
    #
    #         # Save a trained model, configuration and tokenizer
    #         model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    #
    #         # If we save using the predefined names, we can load using `from_pretrained`
    #         model_output_dir = os.path.join(args.output_dir, str(epoch))
    #         if not os.path.exists(model_output_dir):
    #             os.makedirs(model_output_dir)
    #         output_model_file = os.path.join(model_output_dir, WEIGHTS_NAME)
    #         output_config_file = os.path.join(model_output_dir, CONFIG_NAME)
    #
    #         torch.save(model_to_save.state_dict(), output_model_file)
    #         model_to_save.config.to_json_file(output_config_file)
    #         tokenizer.save_vocabulary(model_output_dir)
    #
    #
    #
    #         if args.do_eval:
    #             model.eval()
    #             eval_loss, eval_accuracy = 0, 0
    #             nb_eval_steps, nb_eval_examples = 0, 0
    #
    #             with open(os.path.join(args.output_dir, "results_"+str(epoch)+".txt"),"w") as f:
    #                 for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    #                     input_ids = input_ids.to(device)
    #                     input_mask = input_mask.to(device)
    #                     segment_ids = segment_ids.to(device)
    #                     label_ids = label_ids.to(device)
    #
    #                     with torch.no_grad():
    #                         logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
    #
    #                     logits_ = F.softmax(logits, dim=-1)
    #                     logits_ = logits_.detach().cpu().numpy()
    #                     label_ids_ = label_ids.to('cpu').numpy()
    #                     outputs = np.argmax(logits_, axis=1)
    #                     for output_i in range(len(outputs)):
    #                         f.write(str(outputs[output_i]))
    #                         for ou in logits_[output_i]:
    #                             f.write(" " + str(ou))
    #                         f.write("\n")
    #                     tmp_eval_accuracy = np.sum(outputs == label_ids_)
    #
    #                     # create eval loss and other metric required by the task
    #                     if output_mode == "classification":
    #                         loss_fct = CrossEntropyLoss()
    #                         tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    #                     elif output_mode == "regression":
    #                         loss_fct = MSELoss()
    #                         tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
    #
    #                     eval_loss += tmp_eval_loss.mean().item()
    #                     eval_accuracy += tmp_eval_accuracy
    #                     nb_eval_examples += input_ids.size(0)
    #                     nb_eval_steps += 1
    #
    #             eval_loss = eval_loss / nb_eval_steps
    #             eval_accuracy = eval_accuracy / nb_eval_examples
    #             loss = tr_loss/nb_tr_steps if args.do_train else None
    #
    #             result = OrderedDict()
    #             result['eval_loss'] = eval_loss
    #             result['eval_accuracy'] = eval_accuracy
    #             result['global_step'] = global_step
    #             result['loss'] = loss
    #
    #             output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    #             with open(output_eval_file, "a+") as writer:
    #                 writer.write("epoch=%s\n"%str(epoch))
    #                 logger.info("***** Eval results *****")
    #                 for key in result.keys():
    #                     logger.info("  %s = %s", key, str(result[key]))
    #                     writer.write("%s = %s\n" % (key, str(result[key])))



    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):


    #         parser.add_argument("--bert_model_baseline", default=None, type=str, required=True,
    #                     help='''a path or url to a pretrained model archive containing:
    #                     'bert_config.json' a configuration file for the model
    #                     'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')
    # parser.add_argument("--bert_model_wsd_sent_cls", default=None, type=str, required=True,
    #                     help='''a path or url to a pretrained model archive containing:
    #                     'bert_config.json' a configuration file for the model
    #                     'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')
    # parser.add_argument("--bert_model_wsd_token", default=None, type=str, required=True,
    #                 help='''a path or url to a pretrained model archive containing:
    #                 'bert_config.json' a configuration file for the model
    #                 'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')
    # parser.add_argument("--bert_model_wsd_sent_ws", default=None, type=str, required=True,
    #                     help='''a path or url to a pretrained model archive containing:
    #                     'bert_config.json' a configuration file for the model
    #                     'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')
        eval_examples_baseline = processor_WSDProcessor.get_dev_examples(args.eval_data_dir_baseline, args.label_data_dir)
        eval_examples_wsd_token = processor_WSD_sent_Processor.get_dev_examples(args.eval_data_dir_wsd_token)
        eval_examples_wsd_sent_ws = processor_WSD_sent_Processor.get_dev_examples(args.eval_data_dir_wsd_sent_ws)
        eval_examples_wsd_sent_cls = processor_WSD_sent_Processor.get_dev_examples(args.eval_data_dir_wsd_sent_cls)

        eval_features_baseline = convert_examples_to_features(
            eval_examples_baseline, label_list_token, args.max_seq_length, tokenizer_baseline, output_mode)
        eval_features_wsd_token = convert_examples_to_features(
            eval_examples_wsd_token, label_list, args.max_seq_length, tokenizer_wsd_token, output_mode)
        eval_features_wsd_sent_ws = convert_examples_to_features(
            eval_examples_wsd_sent_ws, label_list, args.max_seq_length, tokenizer_wsd_sent_ws, output_mode)
        eval_features_wsd_sent_cls = convert_examples_to_features(
            eval_examples_wsd_sent_cls, label_list, args.max_seq_length, tokenizer_wsd_sent_cls, output_mode)


        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples_baseline))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids_baseline = torch.tensor([f.input_ids for f in eval_features_baseline], dtype=torch.long)
        all_input_mask_baseline = torch.tensor([f.input_mask for f in eval_features_baseline], dtype=torch.long)
        all_segment_ids_baseline = torch.tensor([f.segment_ids for f in eval_features_baseline], dtype=torch.long)

        all_input_ids_wsd_token = torch.tensor([f.input_ids for f in eval_features_wsd_token], dtype=torch.long)
        all_input_mask_wsd_token = torch.tensor([f.input_mask for f in eval_features_wsd_token], dtype=torch.long)
        all_segment_ids_wsd_token = torch.tensor([f.segment_ids for f in eval_features_wsd_token], dtype=torch.long)

        all_input_ids_wsd_sent_ws = torch.tensor([f.input_ids for f in eval_features_wsd_sent_ws], dtype=torch.long)
        all_input_mask_wsd_sent_ws = torch.tensor([f.input_mask for f in eval_features_wsd_sent_ws], dtype=torch.long)
        all_segment_ids_wsd_sent_ws = torch.tensor([f.segment_ids for f in eval_features_wsd_sent_ws], dtype=torch.long)

        all_input_ids_wsd_sent_cls = torch.tensor([f.input_ids for f in eval_features_wsd_sent_cls], dtype=torch.long)
        all_input_mask_wsd_sent_cls = torch.tensor([f.input_mask for f in eval_features_wsd_sent_cls], dtype=torch.long)
        all_segment_ids_wsd_sent_cls = torch.tensor([f.segment_ids for f in eval_features_wsd_sent_cls], dtype=torch.long)


        if output_mode == "classification":
            all_label_ids_baseline = torch.tensor([f.label_id for f in eval_features_baseline], dtype=torch.long)
            all_label_ids_wsd_token = torch.tensor([f.label_id for f in eval_features_wsd_token], dtype=torch.long)
            all_label_ids_wsd_sent_ws = torch.tensor([f.label_id for f in eval_features_wsd_sent_ws], dtype=torch.long)
            all_label_ids_wsd_sent_cls = torch.tensor([f.label_id for f in eval_features_wsd_sent_cls], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids_baseline = torch.tensor([f.label_id for f in eval_features_baseline], dtype=torch.float)
            all_label_ids_wsd_token = torch.tensor([f.label_id for f in eval_features_wsd_token], dtype=torch.float)
            all_label_ids_wsd_sent_ws = torch.tensor([f.label_id for f in eval_features_wsd_sent_ws], dtype=torch.float)
            all_label_ids_wsd_sent_cls = torch.tensor([f.label_id for f in eval_features_wsd_sent_cls], dtype=torch.float)

        # eval_data_baseline = TensorDataset(all_input_ids_baseline, all_input_mask_baseline, all_segment_ids_baseline, all_label_ids_baseline)
        # eval_dataloader_baseline = DataLoader(eval_data_baseline, batch_size=args.eval_batch_size, shuffle=False)
        # eval_data_wsd_token = TensorDataset(all_input_ids_wsd_token, all_input_mask_wsd_token, all_segment_ids_wsd_token, all_label_ids_wsd_token)
        # eval_dataloader_wsd_token = DataLoader(eval_data_wsd_token, batch_size=args.eval_batch_size, shuffle=False)
        # eval_data_wsd_sent_ws = TensorDataset(all_input_ids_wsd_sent_ws, all_input_mask_wsd_sent_ws, all_segment_ids_wsd_sent_ws, all_label_ids_wsd_sent_ws)
        # eval_dataloader_wsd_sent_ws = DataLoader(eval_data_wsd_sent_ws, batch_size=args.eval_batch_size, shuffle=False)
        # eval_data_wsd_sent_cls = TensorDataset(all_input_ids_wsd_sent_cls, all_input_mask_wsd_sent_cls, all_segment_ids_wsd_sent_cls, all_label_ids_wsd_sent_cls)
        # eval_dataloader_wsd_sent_cls = DataLoader(eval_data_wsd_sent_cls, batch_size=args.eval_batch_size, shuffle=False)
        eval_data = TensorDataset(all_input_ids_wsd_token, all_input_ids_wsd_sent_ws, all_input_ids_wsd_sent_cls, all_input_mask_wsd_token, all_input_mask_wsd_sent_ws, all_input_mask_wsd_sent_cls, all_segment_ids_wsd_token, all_segment_ids_wsd_sent_ws, all_segment_ids_wsd_sent_cls, all_label_ids_wsd_token, all_label_ids_wsd_sent_ws, all_label_ids_wsd_sent_cls)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)
        # model_baseline.eval()
        model_wsd_token.eval()
        model_wsd_sent_ws.eval()
        model_wsd_sent_cls.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        with open(os.path.join(args.output_dir, "results.txt"),"w") as f:
            for input_ids2, input_ids3, input_ids4, input_mask2, input_mask3, input_mask4, segment_ids2, segment_ids3, segment_ids4, label_ids2, label_ids3, label_ids4 in tqdm(eval_dataloader, desc="Evaluating"):
                # input_ids1 = input_ids1.to(device)
                input_ids2 = input_ids2.to(device)
                input_ids3 = input_ids3.to(device)
                input_ids4 = input_ids4.to(device)

                # input_mask1 = input_mask1.to(device)
                input_mask2 = input_mask2.to(device)
                input_mask3 = input_mask3.to(device)
                input_mask4 = input_mask4.to(device)

                # segment_ids1 = segment_ids1.to(device)
                segment_ids2 = segment_ids2.to(device)
                segment_ids3 = segment_ids3.to(device)
                segment_ids4 = segment_ids4.to(device)

                # label_ids1 = label_ids1.to(device)
                label_ids2 = label_ids2.to(device)
                label_ids3 = label_ids3.to(device)
                label_ids4 = label_ids4.to(device)

                with torch.no_grad():
                    # logits1 = model_baseline(input_ids=input_ids1, token_type_ids=segment_ids1, attention_mask=input_mask1, labels=None)
                    logits2 = model_wsd_token(input_ids=input_ids2, token_type_ids=segment_ids2, attention_mask=input_mask2, labels=None)
                    logits3 = model_wsd_sent_ws(input_ids=input_ids3, token_type_ids=segment_ids3, attention_mask=input_mask3, labels=None)
                    logits4 = model_wsd_sent_cls(input_ids=input_ids4, token_type_ids=segment_ids4, attention_mask=input_mask4, labels=None)

                logits = logits2.add(logits3).add(logits4)
                logits_ = F.softmax(logits, dim=-1)
                logits_ = logits_.detach().cpu().numpy()
                label_ids_ = label_ids2.to('cpu').numpy()
                outputs = np.argmax(logits_, axis=1)
                for output_i in range(len(outputs)):
                    f.write(str(outputs[output_i]))
                    for ou in logits_[output_i]:
                        f.write(" " + str(ou))
                    f.write("\n")
                tmp_eval_accuracy = np.sum(outputs == label_ids_)

                # create eval loss and other metric required by the task
                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids2.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids2.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                nb_eval_examples += input_ids2.size(0)
                nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result = OrderedDict()
        result['eval_loss'] = eval_loss
        result['eval_accuracy'] = eval_accuracy
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results *****")
            for key in result.keys():
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
