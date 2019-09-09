import librosa
import numpy as np


DATA_CONFIG_EXPECTED_ENTRIES = {
    "csv_path", "array_dir", "vocab_path", "n_freqs", "window_size",
    "hop_length", "normalize", "resample_rate"}
DATA_CONFIG_INT_ENTRIES = {"n_freqs", "window_size", "hop_length",
                           "resample_rate"}
DATA_CONFIG_BOOL_ENTRIES = {"normalize"}


def read_data_config(config_path):
    """Read a config file with information about the data.

    The file should be in csv format and contain the following entries (order
    doesn't matter):
        csv_path: Path to a file like created in prepare_data.py
        array_dir: Path to the directory containing the corresponding numpy
                   arrays, as processed in prepare_data.py
        vocab_path: Path to a vocabulary file such as one created in vocab.py.
        n_freqs: Frequencies (e.g. STFT or mel bins) to be expected in the
                 data. Will lead to problems if this does not match with
                 reality.
        window_size: Window size to use for STFT (n_fft argument in librosa).
                     Relevant for preprocessing only (and for you to know the
                     parameters of the data).
        hop_length: STFT hop length. See window_size.
        normalize: Whether to normalize data in preprocessing. True or False.
        resample_rate: Sample rate to resample the data to. Use 0 to do no
                       resampling.

    Entries can be in any order. Missing or superfluous entries will result in
    a crash. You can add comments via lines starting with '#'.

    Returns:
        dict with config file entries. Numerical entries are converted to int.
    """
    config_dict = dict()
    with open(config_path) as data_config:
        for line in data_config:
            if line[0] == "#":
                continue
            key, val = line.strip().split(",")
            config_dict[key] = val
    found_entries = set(config_dict.keys())
    for f_entry in found_entries:
        if f_entry not in DATA_CONFIG_EXPECTED_ENTRIES:
            raise ValueError("Entry {} found in config file which should not "
                             "be there.".format(f_entry))
    for e_entry in DATA_CONFIG_EXPECTED_ENTRIES:
        if e_entry not in found_entries:
            raise ValueError("Entry {} expected in config file, but not "
                             "found.".format(e_entry))
    for i_entry in DATA_CONFIG_INT_ENTRIES:
        config_dict[i_entry] = int(config_dict[i_entry])

    def str_to_bool(string):
        if string == "True":
            return True
        elif string == "False":
            return False
        else:
            raise ValueError("Invalid bool string {}. Use 'True' or "
                             "'False'.".format(string))

    for b_entry in DATA_CONFIG_BOOL_ENTRIES:
        config_dict[b_entry] = str_to_bool(config_dict[b_entry])

    return config_dict


def extract_transcriptions(csv_path, which_sets):
    """Just return a list of transcriptions from a corpus csv as strings.

    Parameters:
        csv_path: Path to corpus csv that has all the transcriptions.
        which_sets: Iterable (e.g. list, tuple or set) that contains all the
                    subsets to be considered (e.g. train-clean-360 etc.).
    Returns:
        list of strings, the transcriptions (in order!).

    """
    with open(csv_path, mode="r") as corpus:
        lines = [line.strip().split(",") for line in corpus]
    if which_sets:
        transcrs = [line[2] for line in lines if line[3] in which_sets]
    else:
        transcrs = [line[2] for line in lines]

    if not transcrs:
        raise ValueError("Filtering resulted in size-0 dataset! Maybe you "
                         "specified an invalid subset? You supplied "
                         "'{}'.".format(which_sets))
    return transcrs


def raw_to_mel(audio, sampling_rate, window_size, hop_length, n_freqs,
               normalize):
    """Go from 1D numpy array containing audio waves to mel spectrogram.

    To be precise, this is a log power mel spectrogram. It is NOT decibel-scaled
    but this can easily be corrected via multiplication by a constant factor.

    Parameters:
        audio: 1D numpy array containing the audio.
        sampling_rate: Sampling rate of audio.
        window_size: STFT window size.
        hop_length: Distance between successive STFT windows.
        n_freqs: Number of mel frequency bins.
        normalize: If set, normalize log power spectrogram to mean 0, std 1.

    Returns:
        Log mel spectrogram.

    """
    spectro = librosa.stft(audio, n_fft=window_size, hop_length=hop_length)
    power = np.abs(spectro)**2
    mel = librosa.feature.melspectrogram(S=power, sr=sampling_rate,
                                         n_mels=n_freqs)
    logmel = np.log(mel + 1e-11)
    if normalize:
        logmel = (logmel - np.mean(logmel)) / np.std(logmel)
    return logmel
