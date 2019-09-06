import argparse


def make_vocab(csv_path, out_path):
    """Create a human-readable character-index mapping for a given corpus.

    Parameters:
        csv_path: Path to the corpus csv (see e.g. w2l_inputs for a more
                  detailed description of how this should look).
        out_path: Path to store the vocabulary to.

    """
    with open(csv_path, mode="r") as corpus:
        lines_split = [line.strip().split(",") for line in corpus]
    transcrs = [line[2] for line in lines_split]

    char_set = sorted(set("".join(transcrs)))

    with open(out_path, mode="w") as vocab_file:
        for ind, char in enumerate(char_set, start=1):
            vocab_file.write(char + "," + str(ind) + "\n")
    print("Written vocabulary of size {} to {}.".format(len(char_set),
                                                        out_path))


def parse_vocab(vocab_path):
    """Turn a human-readable character-index mapping into python dictionaries.

    Parameters:
        vocab_path: Path to vocabulary file.

    Returns:
        Python dictionaries mapping character -> index as well as
        index -> character.

    """

    def process_line(line):
        ch, ind = line.rstrip().split(",")
        return ch, int(ind)

    with open(vocab_path, mode="r") as vocab_file:
        map_ch_ind = dict(process_line(line) for line in vocab_file)
    # could be a list but it's easier to just revert the dict
    map_ind_ch = dict((ind, ch) for ch, ind in map_ch_ind.items())
    return map_ch_ind, map_ind_ch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a human-readable "
                                                 "character-index mapping for "
                                                 "a given corpus.")
    parser.add_argument("csv_path", help="Path to corpus csv for which to "
                                         "build the vocabulary.")
    parser.add_argument("out_path", help="Path to store the vocabulary to.")
    args = parser.parse_args()

    make_vocab(args.csv_path, args.out_path)
