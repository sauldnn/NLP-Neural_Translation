from termcolor import colored
import random
import numpy as np

import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training

import w1_unittest

from data_prep import append_eos, tokenize, detokenize

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
filtered_eval_stream = trax.data.FilterByLength(
                        max_length=512,
                        length_keys=[0, 1]
                        )(tokenized_eval_stream)
train_input, train_target = next(filtered_train_stream)
# print a sample niput-target pair o tokenized sentences
# print(colored(f'Single tokenized example input:'), train_input)
# print(colored(f'Single tokenized example target:'), train_target)

# setup helper functions for tokenizing and detokenizing sentences

# Bucketing to create streams of batches.

# Buckets are defined in terms of boundaries and batch sizes.
# batch_sizes[i] determines the batch shize for items with length <zboundaries[i]
# Sol below, we'll take a batch of 256 sentences of length <8, 128 if length is
# between 8 and 16 and so on -- and only 2 if length is over 512.

boundaries = [2**i for i in range(3, 10)]
batch_sizes = [2**i for i in range(1, 9)][::-1]

# Create generators
train_batch_stream = trax.data.BucketByLength(
    boundaries, batch_sizes,
    length_keys=[0, 1] # As before: count inputs and targets to length.
)(filtered_train_stream)

eval_batch_stream = trax.data.]BucketByLength(
    boundaries, batch_sizes,
    length_keys=[0, 1] # As before: count inputs and targets to length.
)(filtered_eval_stream)

# Add masking for the padding (0s).
train_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(train_batch_stream)
eval_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(eval_batch_stream)

input_batch, target_batch, mask_batch = next(train_batch_stream)
