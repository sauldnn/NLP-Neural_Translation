# UNQ_C5
# GRADED PART
def train_task_function(train_batch_stream):
    """Returns a trax.training.TrainTask object.

    Args:
    train_batch_stream generator: labeled data generator

    Returns:
    A trax.training.TrainTask object.
    """
    return training.TrainTask(

        ### START CODE HERE

        # use the train batch stream as labeled data
        labeled_data= train_batch_stream,

        # use the cross entropy loss
        loss_layer=tl.CrossEntropyLoss(),

        # use the Adam optimizer with learning rate of 0.01
        optimizer= trax.optimizers.Adam(0.01),

        # use the `trax.lr.warmup_and_rsqrt_decay` as the learning rate schedule
        # have 1000 warmup steps with a max value of 0.01
        lr_schedule= trax.lr.warmup_and_rsqrt_decay(1000, 0.01),

        # have a checkpoint every 10 steps
        n_steps_per_checkpoint= 10,

        ### END CODE HERE
    )

def logsoftmax_sample(log_probs, temperature=1.0):  # pylint: disable=invalid-name
  """Returns a sample from a log-softmax output, with temperature.

  Args:
    log_probs: Logarithms of probabilities (often coming from LogSofmax)
    temperature: For scaling before sampling (1.0 = default, 0.0 = pick argmax)
  """
  # This is equivalent to sampling from a softmax with temperature.
  u = np.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
  g = -np.log(-np.log(u))
  return np.argmax(log_probs + g * temperature, axis=-1)

 def next_symbol(NMTAttn, input_tokens, cur_output_tokens, temperature):
    """Returns the index of the next token.

    Args:
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        input_tokens (np.ndarray 1 x n_tokens): tokenized representation of the input sentence
        cur_output_tokens (list): tokenized representation of previously translated words
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)

    Returns:
        int: index of the next token in the translated sentence
        float: log probability of the next symbol
    """

    ### START CODE HERE ###

    # set the length of the current output tokens
    token_length = len(cur_output_tokens)

    # calculate next power of 2 for padding length
    padded_length = np.power(2, int(np.ceil(np.log2(token_length + 1))))

    # pad cur_output_tokens up to the padded_length
    padded = cur_output_tokens + [0] * (padded_length - token_length)

    padded_with_batch = np.expand_dims(padded, axis=0)

    # get the model prediction. remember to use the `NMTAttn` argument defined above.
    # hint: the model accepts a tuple as input (e.g. `my_model((input1, input2))`)
    output, _ = NMTAttn((input_tokens, padded_with_batch))

    # get log probabilities from the last token output
    log_probs = output[0, token_length, :]

    # get the next symbol by getting a logsoftmax sample (*hint: cast to an int)
    symbol = int(tl.logsoftmax_sample(log_probs, temperature))

    ### END CODE HERE ###
    return symbol, float(log_probs[symbol])

def next_symbol(NMTAttn, input_tokens, cur_output_tokens, temperature):
    """Returns the index of the next token.

    Args:
        NMTAttn (tl.Serial): An LSTM sequence-to-sequence model with attention.
        input_tokens (np.ndarray 1 x n_tokens): tokenized representation of the input sentence
        cur_output_tokens (list): tokenized representation of previously translated words
        temperature (float): parameter for sampling ranging from 0.0 to 1.0.
            0.0: same as argmax, always pick the most probable token
            1.0: sampling from the distribution (can sometimes say random things)

    Returns:
        int: index of the next token in the translated sentence
        float: log probability of the next symbol
    """

    ### START CODE HERE ###

    # set the length of the current output tokens
    token_length = len(cur_output_tokens)

    # calculate next power of 2 for padding length
    padded_length = np.power(2, int(np.ceil(np.log2(token_length + 1))))

    # pad cur_output_tokens up to the padded_length
    padded = cur_output_tokens + [0] * (padded_length - token_length)

    padded_with_batch = np.expand_dims(padded, axis=0)

    # get the model prediction. remember to use the `NMTAttn` argument defined above.
    # hint: the model accepts a tuple as input (e.g. `my_model((input1, input2))`)
    output, _ = NMTAttn((input_tokens, padded_with_batch))

    # get log probabilities from the last token output
    log_probs = output[0, token_length, :]

    # get the next symbol by getting a logsoftmax sample (*hint: cast to an int)
    symbol = int(tl.logsoftmax_sample(log_probs, temperature))

    ### END CODE HERE ###
    return symbol, float(log_probs[symbol])
