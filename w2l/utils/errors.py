import editdistance


def letter_error_rate_corpus(targets, predictions):
    total_error_count = 0
    total_expected_count = 0
    for true, predicted in zip(targets, predictions):
        total_error_count += editdistance.eval(true, predicted)
        total_expected_count += len(true)
    return total_error_count / total_expected_count


def word_error_rate_corpus(targets, predictions):
    word_targets = [true.split() for true in targets]
    word_predictions = [predicted.split() for predicted in predictions]
    return letter_error_rate_corpus(word_targets, word_predictions)


def letter_error_rate_single(true, predicted):
    return letter_error_rate_corpus([true], [predicted])


def word_error_rate(true, predicted):
    return word_error_rate_corpus([true], [predicted])
