import os

import numpy as np
import tensorflow as tf


def w2l_input_fn_npy(csv_path, array_base_path, which_sets, train, vocab,
                     n_freqs, batch_size, threshold, normalize):
    """Builds a TF dataset for the preprocessed data.

    NOTE: The data on disk is assumed to be stored channels_first.
    Parameters:
        csv_path: Should be the path to an appropriate csv file linking sound
                  files and label data. This csv should contain lines as
                  follows:
                    file_id,file_path,transcription,set where:
                      file_id: unique identifier for the file.
                      file_path: Relative path to the file within corpus
                                 directory. This is not needed here but is
                                 originally used to create the array directory,
                                 so this entry is assumed to be in the csv.
                      transcription: The target string.
                      set: Subset this piece of data belongs to.
        array_base_path: Base path to where the .npy arrays are stored.
        which_sets: Iterable (e.g. list, tuple or set) that contains all the
                    subsets to be considered (e.g. train-clean-360 etc.).
        train: Whether to shuffle and repeat data.
        vocab: Dictionary mapping characters to indices.
        n_freqs: Number of frequencies/"channels" in the data that will be
                 loaded. Needs to be set so that the model has this
                 information.
        batch_size: How big the batches should be.
        threshold: Float to use for thresholding input arrays.
                   See the _pyfunc further below for some important notes.
        normalize: Bool; if True, normalize each input array to mean 0, std 1.

    Returns:
        tf.data.Dataset

    """
    print("Building dataset for {} set using file {}...".format(
        which_sets, csv_path))
    # first read the csv and keep the useful stuff
    with open(csv_path, mode="r") as corpus:
        lines_split = [line.strip().split(",") for line in corpus]
    print("\t{} entries found.".format(len(lines_split)))

    if which_sets:
        print("\tFiltering requested subset...")
        lines_split = [line for line in lines_split if line[3] in which_sets]
    if not lines_split:
        raise ValueError("Filtering resulted in size-0 dataset! Maybe you "
                         "specified an invalid subset? You supplied "
                         "'{}'.".format(which_sets))
    print("\t{} entries remaining.".format(len(lines_split)))

    print("\tCreating the dataset...")
    ids, _, transcrs, subsets = zip(*lines_split)
    files = [os.path.join(array_base_path, fid + ".npy") for fid in ids]

    def _to_arrays(fname, trans):
        return _pyfunc_load_arrays_map_transcriptions(
            fname, trans, vocab, threshold, normalize)

    def gen():  # dummy to be able to use from_generator
        for file_name, transcr in zip(files, transcrs):
            # byte encoding is necessary in python 3, see TF 1.4 known issues
            yield file_name.encode("utf-8"), transcr.encode("utf-8")

    data = tf.data.Dataset.from_generator(
        gen, (tf.string, tf.string))

    if train:
        # this basically shuffles the full dataset
        data = data.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=2 ** 18))

    output_types = [tf.float32, tf.int32, tf.int32, tf.int32]
    data = data.map(
        lambda fid, trans: tuple(tf.numpy_function(
            _to_arrays, [fid, trans], output_types)),
        num_parallel_calls=3)
    # NOTE 1: padding value of 0 for element 1 and 3 is just a dummy (since
    #         sequence lengths are always scalar)
    # NOTE 2: changing padding value of -1 for element 2 requires changes
    # in the model as well!
    pad_shapes = ((n_freqs, -1), (), (-1,), ())
    pad_values = (np.log(1e-11).astype(np.float32), 0, -1, 0)
    data = data.padded_batch(
        batch_size, padded_shapes=pad_shapes, padding_values=pad_values)
    map_fn = pack_inputs_in_dict
    data = data.map(map_fn, num_parallel_calls=3)
    data = data.prefetch(2)  # 2 batches

    return data


def _pyfunc_load_arrays_map_transcriptions(file_name, trans, vocab,
                                           threshold, normalize):
    """Mapping function to go from file names to numpy arrays.

    Goes from file_id, transcriptions to a tuple np_array, coded_transcriptions
    (integers).
    NOTE: Files are assumed to be stored channels_first. If this is not the
          case, this will cause trouble down the line!!
    Parameters:
        file_name: Path built from ID taken from data csv, should match npy
                   file names. Expected to be utf-8 encoded as bytes.
        trans: Transcription. Also utf-8 bytes.
        vocab: Dictionary mapping characters to integers.
        threshold: Float to use for thresholding the array. Any values more
                   than this much under the maximum will be clipped. E.g. if
                   the max is 15 and the threshold is 50, any value below -35
                   would be clipped to -35. It is your responsibility to pass a
                   reasonable value here -- this can vary heavily depending on
                   the scale of the data (it is however invariant to shifts).
                   Passing 0 or any "False" value here disables thresholding.
                   NOTE: You probably don't want to use this with
                   pre-normalized data since in that case, each example is
                   essentially on its own scale (one that results in mean 0 and
                   std 1, or whatever normalization was used) so a single
                   threshold value isn't really applicable.
        normalize: Bool; if True, normalize the array to mean 0, std 1.

    Returns:
        Tuple of 2D numpy array (n_freqs x seq_len), scalar (seq_len),
        1D array (label_len)

    """
    array = np.load(file_name.decode("utf-8"))
    trans_mapped = np.array([vocab[ch] for ch in trans.decode("utf-8")],
                            dtype=np.int32)
    length = np.int32(array.shape[-1])
    trans_length = np.int32(len(trans_mapped))

    if threshold:
        clip_val = np.max(array) - threshold
        array = np.maximum(array, clip_val)
    if normalize:
        array = (array - np.mean(array)) / np.std(array)

    return_vals = (array.astype(np.float32), length, trans_mapped,
                   trans_length)

    return return_vals


def pack_inputs_in_dict(audio, length, trans, trans_length):
    """For estimator interface (only allows one input -> pack into dict)."""
    return ({"audio": audio, "length": length},
            {"transcription": trans, "length": trans_length})
