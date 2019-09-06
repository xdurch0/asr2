import os

import tensorflow as tf
import tensorflow.keras.layers as layers


def make_w2l_model(vocab_size, data_format):
    """Creates a Keras model that does the W2L forward computation.

    Just goes from mel spectrogram input to logits output.

    Parameters:
        data_format: channels_first or channels_last.
        vocab_size: Size of the vocabulary. Output layer will have this size + 1
                    for the blank label.

    Returns:
        Keras sequential model.

    TODO could allow model configs etc. For now, architecture is hardcoded

    """
    channel_ax = 1 if data_format == "channels_first" else -1

    layer_list = [
        layers.Conv1D(256, 48, 2, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(256, 7, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(256, 7, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(256, 7, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(256, 7, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(256, 7, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(256, 7, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(256, 7, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(256, 7, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(2048, 32, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(2048, 1, 1, "same", data_format),
        layers.BatchNormalization(channel_ax),
        layers.ReLU(),
        layers.Conv1D(vocab_size + 1, 1, 1, "same", data_format)
        ]

    return tf.keras.Sequential(layer_list, name="w2l")


def w2l_forward(features, model, data_format):
    """Simple forward pass of a W2L model to compute logits.

    Parameters:
        features: Dict of features as returned by the tf dataset.
        model: tf.keras.Sequential model to transform spectrograms to logits.
        data_format: channels_first/last.

    Returns:
        Result of applying model to audio.

    """
    audio = features["audio"]
    if data_format == "channels_last":
        audio = tf.transpose(audio, [0, 2, 1])

    return model(audio)


def w2l_train_step(features, labels, model, optimizer, data_format, on_gpu):
    """Implements train step of the W2L model.

    Parameters:
        features: Dict of features as returned by the tf dataset.
        labels: Similarly, dict of labels.
        model: tf.keras.Sequential model to transform spectrograms to logits.
        optimizer: Optimizer instance to do training with.
        data_format: channels_first/last.
        on_gpu: Bool, whether to run on GPU. This changes how the transcriptions
                are handled.

    """

    audio, audio_lengths = features["audio"], features["length"]
    transcrs, transcr_lengths = labels["transcriptions"], labels["length"]
    if data_format == "channels_last":
        audio = tf.transpose(audio, [0, 2, 1])

    with tf.GradientTape() as tape:
        logits = model(audio)
        # after this we need logits in shape time x batch_size x vocab_size
        if data_format == "channels_first":  # bs x v x t -> t x bs x v
            logits_tm = tf.transpose(logits, [2, 0, 1],
                                     name="logits_time_major")
        else:  # channels last: bs x t x v -> t x bs x v
            logits_tm = tf.transpose(logits, [1, 0, 2],
                                     name="logits_time_major")

        audio_lengths = tf.cast(audio_lengths / 2, tf.int32)

        if on_gpu:
            ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels=transcrs, logits=logits_tm, label_length=transcr_lengths,
                logit_length=audio_lengths, logits_time_major=True,
                blank_index=0), name="avg_loss")
        else:
            transcrs_sparse = dense_to_sparse(transcrs, sparse_val=-1)
            ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels=transcrs_sparse, logits=logits_tm, label_length=None,
                logit_length=audio_lengths, logits_time_major=True,
                blank_index=0), name="avg_loss")

    grads = tape.gradient(ctc_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def w2l_train_full(dataset, model, steps, data_format, adam_params, on_gpu,
                   model_dir):
    data_step_limited = dataset.take(steps)
    opt = tf.optimizers.Adam(*adam_params)
    for features, labels in data_step_limited:
        w2l_train_step(features, labels, model, opt, data_format, on_gpu)
    # TODO store the model regularly or something? lol
    model.save(os.path.join(model_dir, "final.h5"))


def w2l_decode(features, model, data_format):
    """Wrapper to decode using W2L model.

    Parameters:
        features: Dict of features as returned by the tf dataset.
        model: tf.keras.Sequential model to transform spectrograms to logits.
        data_format: channels_first/last.

    Returns:
        Sparse or dense tensor with the top predictions.

    """
    logits = w2l_forward(features, model, data_format)

    return ctc_decode_top(logits, features["length"], pad_val=-1)


def ctc_decode_top(logits, seq_lengths, beam_width=100, merge_repeated=False,
                   pad_val=0, as_sparse=False):
    """Simpler version of ctc decoder that only returns the top result.

    Parameters:
        logits: Passed straight to ctc decoder. This has to be time-major and
                channels_last!!
        seq_lengths: Same.
        beam_width: Same.
        merge_repeated: Same.
        pad_val: Value to use to pad dense tensor. No effect if as_sparse is
                 True.
        as_sparse: If True, return results as sparse tensor.

    Returns:
        Sparse or dense tensor with the top predictions.

    """
    with tf.name_scope("decoding"):
        decoded_sparse_list, _ = tf.nn.ctc_beam_search_decoder(
            logits, seq_lengths, beam_width=beam_width, top_paths=1,
            merge_repeated=merge_repeated)
        decoded_sparse = decoded_sparse_list[0]
        decoded_sparse = tf.cast(decoded_sparse, tf.int32)
        if as_sparse:
            return decoded_sparse
        else:
            # this should result in a bs x t matrix of predicted classes
            return tf.sparse.to_dense(decoded_sparse,
                                      default_value=pad_val,
                                      name="dense_decoding")


def dense_to_sparse(dense_tensor, sparse_val=0):
    """Inverse of tf.sparse_to_dense.

    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.

    Returns:
        SparseTensor equivalent to the dense input.

    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                               name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                   name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape",
                               out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)

