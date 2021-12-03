def append_eos(stream):
    for (inputs, target) in stream:
        inputs_with_eos = list(inputs) + [EOS]
        target_with_eos = list(target) + [EOS]
        yield np.array(inputs_with_eos), np.array(target_with_eos)


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
