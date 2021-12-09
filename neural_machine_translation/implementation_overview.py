def NMTAttn(input_vocab=333000,
            target_vocab=333000,
            d_model=1024,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_attention_heads=4,
            attention_dropout=0.0,
            mode="train"):
    """Returns  an LSTM  sequence-to-sequence model with attention.

    The input to the model is a pair (inputs tokens, target tokens), e.g.,
    an English sentence (tokenize) an it's translation into German (tokenized).

    Args:
        input_vocab_size (int): vocab size of the input
        target_vocab_size (int) : vocab size of the target
        d_model (int):depth of embedding (n_units in the LSTM cell)
        n_encoder_layers (int): number of LSTM layers in the encoder
        n_decoder_layers (int): number of LSTM layers in the decoder after attention
        n_attention_heads (int): number of attention heads
        attention_dropout (float): dropout for the attention layer
        mode (str): 'train', 'eval', 'predict', predict mode is for fast inference

    Returns:
        An LSTM sequence-to-sequence model with attention.
    """
    input_encoder = input_encoder_fn(input_vocab_size, d_model, n_encoder_layers)

    pre_attention_decoder = pre_attention_decoder_fn(mode, target_vocab_size, d_model)

    model = tl.Seria(
        tl.Select([0, 1, 0, 1]),
        tl.Parallel(input_encoder, pre_attention_decoder),
        tl.Fn("PrepareAttentionInput", prepare_attention_input, n_out=4),
        tl.Residual(tl.AttentionQKV(d_model, n_heads=n_attention_heads, dropout=attention_dropout, mode=None)),
        tl.Select([0, 2]),
        [tl.LSTM(d_model) for _ in range(n_decoder_layers)],
        tl.Dense(target_vocab_size),
        tl.LogSoftmax()
    )
    return model
