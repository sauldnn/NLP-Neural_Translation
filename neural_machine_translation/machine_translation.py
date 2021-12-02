from termcolor import colored
import random
import numpy as np

import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training

import w1_unittest

# Get generator function for the training set
# This will download the train dataset if no data_dir is specified.
train_stream_fn = trax.data.TFDS("opus/medical",
                                data_dir = "./data/",
                                keys = ("en", "de"),
                                eval__holdout_size = 0.01,
                                train = True
                                )
#Get generator function for the eval set
eval_stream_fn = trax.data.TFDS("opus/medical",
                                data_dir="./data/",
                                keys=("en", "de"),
                                eval__holdout_size=0.1,
                                train = True
                                )
train_stream = train_stream_fn()

#you can print a sample pair from our training data with the generator functions
# print(colored("train data (en, de) tuple", red), next(train_stream))
# print()

# eval_stream = eval_stream_fn()
# print(colored("eval data (en, de) tuple: ", "red"), next(eval_stream))


# with the corpus importet, let's preprocessing the sentences into a format that our model can accept.

VOCAB_FILE = "ende_32k.subword"
VOCAB_DIR = "data/"

# Tokenize the dataset
tokenized_train_stream = trax.data.Tokenize(vacab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(train_stream)
tokenized_eval_stream = trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(eval_stream)

# Append EOS at the end of each sentence

# Integer assigned as end-of-sentence (ESO)

# generator helper function to append EOS to each sentence
def append_eos(stream):
    for (inputs, target) in stream:
        inputs_with_eos = list(inputs) + [EOS]
        target_with_eos = list(target) + [EOS]
        yield np.array(inputs_with_eos), np.array(target_with_eos)

# append EOS to the training dataset
tokenized_train_stream = append_eos(tokenized_train_stream)

# append EOS to the eval dataset
tokenized_eval_stream = append_eos(tokenized_eval_stream)

# Filter too long sentences to not run out of memory
# length_keys = [0, 1] means we filter voth English and German sentences, so
# both must be not longer that 256 tokens for training / 512 for eval.

filtered_train_stream = trax.data.FilterByLength(
                        max_length=512,
                        length_keys=[0, 1]
                        )(tokenized_eval_stream)

train_input, train_target = next(filtered_train_stream)
# print a sample niput-target pair o tokenized sentences
# print(colored(f'Single tokenized example input:'), train_input)
# print(colored(f'Single tokenized example target:'), train_target)

# setup helper functions for tokenizing and detokenizing sentences

def tokenize(input_str, vocab_file=None, vocab_dir=None):
    """Encodes a string to an array of integers
    Inputs:
        input_str (str): huma-readable string to encode
        vocab_file (str): filename of the vocabulary text file
        vocab_dir (str): path to the vocabulary file

    Returns:
        numpy.ndarray: tokenized version of the input string
    """

    # Set the encoding of the "end of the sentence" as 1
    EOS = 1

    inputs = next(trax.data.tokenize(iter([input_str]),
                                    vocab_file=vocab_file,
                                    vocab_dir=vocab_dir
                                    ))
    inputs = inputs + [EOS]
    batch_inputs = np.reshape(np.array(inputs), [1, -1])
    return = batch_inputs

def detokenize(integers, vocab_file=None, vocab_dir=None):
    """Decodes an array of integer to a human readable string

    Args:
        integers (numpy.ndarray): array of integers to decode
        vocab_file (str): filename of the vocabulary text file
        vocab_dir (str): path to the vocabulary file

    Returns:
        str: the decoded sentence
    """

    integers = list(np.squeeze(integers))

    EOS=1

    if EOS ni integers:
        integers = integers[: integers.index(EOS)]

    return trax.data.detokenize(integers, vocab_file=vocab_file, vocab_dir, vocab_dir)
