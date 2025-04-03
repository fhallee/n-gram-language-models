"""Defines a unigram language model."""

import math


def load_tokens(file_path):
    """
    Loads tokens from a file.

    Args:
        file_path (str): File path to pre-tokenized file with tokens
                        separated by spaces
    Returns:
        list: A list of all tokens (str) extracted from the file
    """
    tokens = []
    with open(file_path, "r", encoding="latin-1") as file:
        for line in file:
            tokens.extend(line.strip().split())
    return tokens


def count_unigrams(tokens):
    """
    Counts the occurrences of each unigram in a list.

    Args:
        tokens (list): A list of tokens (str)
    Returns:
        dict: A dictionary with unigrams (str) as keys and
            token counts (int) as values
    """
    unigram_counts = {}
    for token in tokens:
        if token not in unigram_counts:
            unigram_counts[token] = 1
        else:
            unigram_counts[token] += 1
    return unigram_counts


def compute_unigram_log_thetas(unigram_counts, alpha):
    """
    Calculates the log probabilities for unigrams with add-alpha smoothing.

    Args:
        unigram_counts (dict): A dictionary with unigrams (str) as keys
                            and token counts (int) as values
        alpha (float): The smoothing parameter
    Returns:
        dict: A dictionary with unigrams (str) as keys and log
            probabilities (float) as values. Contains a special
            "<UNSEEN>" key for the probability of unseen unigrams
    """
    log_thetas = {}
    total_token_count = sum(unigram_counts.values())
    total_type_count = len(unigram_counts)
    for w, w_count in unigram_counts.items():
        log_thetas[w] = math.log(
            (w_count + alpha) / (total_token_count + alpha * total_type_count)
        )
    log_thetas["<UNSEEN>"] = math.log(
        alpha / (total_token_count + alpha * total_type_count)
    )
    return log_thetas


def calculate_log_probability(test_tokens, log_thetas):
    """
    Calculates the log probability of a sequence of tokens.

    Args:
        test_tokens (list): A list of tokens (str)
        log_thetas (dict): A dictionary with unigrams (str) as keys and log
                        probabilities (float) as values. Contains a special
                        "<UNSEEN>" key for the probability of unseen unigrams
    Returns:
        float: The log probability of the sequence of tokens
    """
    log_probability = 0
    for token in test_tokens:
        if token in log_thetas:
            log_probability += log_thetas[token]
        else:
            log_probability += log_thetas["<UNSEEN>"]
    return log_probability


def find_best_alpha(unigram_counts, dev_tokens, a, b, tolerance=1e-5):
    """
    Golden-selection search to find the value of alpha
    that maximizes the log probability of a held out dataset.

    Adapted from https://en.wikipedia.org/wiki/Golden-section_search

    Args:
        unigram_counts (dict): A dictionary with unigrams (str) as keys
                            and token counts (int) as values
        dev_tokens (list): A list of tokens (str) from the development dataset
        a (float): The lower bound of the search interval
        b (float): The upper bound of the search interval
        tolerance (float, optional): Tolerance parameter

    Returns:
        float: The optimal parameter for add-alpha smoothing
    """
    invphi = (math.sqrt(5) - 1) / 2
    while b - a > tolerance:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        if calculate_log_probability(
            dev_tokens, compute_unigram_log_thetas(unigram_counts, c)
        ) > calculate_log_probability(
            dev_tokens, compute_unigram_log_thetas(unigram_counts, d)
        ):
            b = d
        else:
            a = c
    return round((b + a) / 2, 1)


class UnigramLM:
    """
    A unigram language model.

    Attributes:
        log_thetas (dict): A dictionary with unigrams (str) as keys and log
                        probabilities (float) as values. Contains a special
                        "<UNSEEN>" key for the probability of unseen unigrams
        alpha (float): The smoothing parameter for add-alpha smoothing
    """

    def __init__(self):
        self.log_thetas = {}
        self.alpha = 1

    def train(self, train_tokens, dev_tokens=None, alpha=None):
        """
        Trains the model by calculating unigram log probabilities
        using add-alpha smoothing. Either a specified alpha value
        or a development dataset must be provided. If a dev set and
        no value for alpha is provided, the function automatically
        finds the optimal alpha value.

        Args:
            train_tokens (list): A list of tokens (str) from the
                                training dataset
            dev_tokens (list, optional): A list of tokens (str) from
                                        the development dataset
            alpha (float, optional): Desired smoothing parameter for
                                    add-alpha smoothing
        """
        train_unigram_counts = count_unigrams(train_tokens)

        if alpha:
            self.alpha = alpha
        elif dev_tokens:
            self.alpha = find_best_alpha(
                train_unigram_counts, dev_tokens, 0.1, 3
            )
        else:
            raise ValueError(
                "Please provide the dev dataset or a value for alpha"
            )

        self.log_thetas = compute_unigram_log_thetas(
            train_unigram_counts, self.alpha
        )

    def log_probability(self, test_tokens):
        return calculate_log_probability(test_tokens, self.log_thetas)
