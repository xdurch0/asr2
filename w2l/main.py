import os

import tensorflow as tf

from .utils.data import read_data_config
from .utils.errors import letter_error_rate_corpus, word_error_rate_corpus
from .utils.vocab import parse_vocab
from .model import make_w2l_model, w2l_train_full, w2l_decode
from .input import w2l_input_fn_npy


def run_asr(mode, data_config, model_dir, data_format="channels_first",
            cpu=False,
            adam_params=(1e-4, 0.9, 0.9, 1e-8), batch_size=16, clipping=500,
            fix_lr=False, normalize=False, steps=300000, threshold=0.,
            which_sets=None):
    """
    All of these parameters can be passed from w2l_cli. Please check
    that one for docs on what they are.

    Returns:
        Depends on mode!
        If train, eval-current or eval-all: Nothing is returned.
        If predict: Returns a generator over predictions for the requested set.
        If return: Return the estimator object. Use this if you want access to
                   the variables or their values, for example.
        If container: Returns a generator over predictions for the given
                      container.

    """
    data_config_dict = read_data_config(data_config)
    csv_path, array_dir, vocab_path, mel_freqs = (
        data_config_dict["csv_path"], data_config_dict["array_dir"],
        data_config_dict["vocab_path"], data_config_dict["n_freqs"])

    ch_to_ind, ind_to_ch = parse_vocab(vocab_path)
    ind_to_ch[-1] = "<PAD>"

    if os.path.isdir(model_dir) and os.listdir(model_dir):
        model = tf.keras.models.load_model(os.path.join(model_dir, "final.h5"))
    else:
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model = make_w2l_model(len(ch_to_ind), mel_freqs, data_format)

    if mode == "return":
        return model

    dataset = w2l_input_fn_npy(csv_path, array_dir, which_sets, mode == "train",
                               ch_to_ind, mel_freqs, batch_size, threshold,
                               normalize)

    if mode == "train":
        w2l_train_full(dataset, model, steps, data_format, adam_params,
                       not cpu, model_dir)

    elif mode == "predict" or mode == "errors":
        def gen():
            for features, labels in dataset:
                pred_batch = w2l_decode(features["audio"],
                                        features["audio_length"],
                                        model, data_format).numpy()
                label_batch = labels["transcription"].numpy()
                for ind in range(pred_batch.shape[0]):
                    predictions_repacked = dict()
                    predictions_repacked["input_length"] = features["length"][ind]

                    # remove padding and convert to chars
                    pred = pred_batch[ind]
                    pred = [p for p in pred if p != -1]
                    pred_ch = "".join([ind_to_ch[ind] for ind in pred])
                    predictions_repacked["decoding"] = pred_ch

                    true = [t for t in label_batch[ind] if t != -1]
                    true_ch = "".join([ind_to_ch[ind] for ind in true])
                    predictions_repacked["true"] = true_ch
                    yield predictions_repacked

    if mode == "predict":
        return gen()

    if mode == "errors":
        true = []
        predicted = []
        for p in gen():
            true.append(p["true"])
            predicted.append(p["decoding"][0])
        ler = letter_error_rate_corpus(true, predicted)
        wer = word_error_rate_corpus(true, predicted)
        print("LER: {}\nWER: {}".format(ler, wer))
