def input_encoder_fn(input_vocab_size, d_model, n_encoder_layers):
    """ Input encoder runs on the input sentence and creates
    activation hat will be the keys and values for attention.

    Args:
        input_vocab_size (int): voacab siz of the input
        d_model (int): depth of embeddin (n_units in the LSTM cell)
        n_encoder_layers (int): number of LSTM layers in the encoder

    Returns:
        tl.Serial: The input encoder
    """
    # create a serial network
    input_encoder = tl.Serial(
        # create an embeding layer to convert tokens to vectors
        tl.Embedding(vocab_size=input_vocab_size, d_feature=d_model),
        [tl.LSTM(n_units=model) for _ in range(n_encoder_layers)]
    )
    return input_encoder

def pre_attention_decoder_fn(mode, target_vocab_size, d_moedl):
    """ Pre attention decoder runs on the target and creates
    activations that are used as queries in attention.

    Args:
        mode (str): 'train' or 'eval'
        target_vocab_size (int): depth of embedding (n_units in the LSTM cell)
    Returns:
        tl.Serial The pre-attention decoder
    """

    pre_attention_decoder = tl.Seria(
        tl.ShiftRight(mode),
        tl.Embedding(vocab_size=target_vocab_size, d_feature=d_model),
        tl.LSTM(n_units=d_model)
    )
    return pre_attention_decoder

def prepare_attention_input(encoder_activations, decoder_activations, inputs):
    """ Prepare queries, keys, values and mask for attention.

    Args:
        encoder_activations fastnp.array(batch_size, padded_input_length, d_model): output from the input encoder
        decoder_activations fastnp.array(batch_size, padded_input_length, d_model): output from the pre_attention decoder
        inputs fastnp.array(batch_size, padded_input_length): input tokens

    Returns:
        queries, keys, values and mask for attention

    """
    # set the keys and values to the encoder activation
    keys = encoder_activations
    values = encoder_activations

    # set the queries to the decoder activations
    queries = decoder_activations

    # generate the mask to the distinguish real tokens from padding
    # inputs is positive for real tokens and 0 where they are padding
    mask = inputs > 0

    # add axes to the mask for attention hands and decoder length.
    mask = fastnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))

    # broadcast so mask shape is [batch size, attention heads, decoder-len, encoder-len.
    mask = mask + fastnp.zeros((1, 1, decoder_activations.shape[1], 1))
    return queries, keys, values, mask

    def AttentionQKV(d_feature, n_heads=1, dropout=0.0, mode="train"):
        """ Returns a leyer that maps (q, k, v, mask) to (activations. mask)
        Args:
            d_feature: Depth/dimensionality of feature embedding.
            n_heads: Number of attention heads.
            dropout: Probabilistic rate for internal dropout applied to attention
                activations (based on query-key pairs) before dotting them with values.
            mode: Either 'train' pr 'eval'.
        """
        return cb.Serial(
            cb.Parallel(
                core.Dense(d_feature),
                core.Dense(d_feature),
                core.Dense(d_feature).
            ),
            PureAttention( #pyLint: disable-no-value-for-parameter
                n_heads=n_heads,
                dropout=dropout,
                mode=mode
                ),
            core.Dense(d_feature)

        )
