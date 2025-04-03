"""Defines a bigram language model."""

import math
import unigram


def prepare_bigram_data(file_path):
    """
    Prepares data for processing by padding sentences from a file
    with "*START*" and "*STOP*" tokens.

    Args:
        file_path (str): File path to pre-tokenized file with tokens
                        separated by spaces and one sentence per line
    Returns:
        list: A list of padded sentences, where each padded sentence is
            a list of tokens (str)
    """
    padded_sentences = []
    with open(file_path, "r", encoding="latin-1") as file:
        for line in file:
            padded_sentences.append(["*START*"] + line.strip().split() + ["*STOP*"])
    return padded_sentences


def count_bigrams(padded_sentences):
    """
    Counts the occurrence of each bigram in a list of padded sentences.

    Args:
        padded_sentences (list): A list of padded sentences, where each
            padded sentence is a list of tokens (str)

    Returns:
        dict: A dictionary with bigrams (tuples of two unigrams (str)) and
            bigram counts (int) as values
    """
    bigram_counts = {}
    for sentence in padded_sentences:
        for w in range(len(sentence) - 1):
            bigram = sentence[w], sentence[w + 1]
            if bigram in bigram_counts:
                bigram_counts[bigram] += 1
            else:
                bigram_counts[bigram] = 1
    return bigram_counts


def compute_bigram_log_theta(
    w, w_prime, unigram_counts, unigram_log_thetas, bigram_counts, beta
):
    """
    Calculates the log probability of a given bigram with smoothing.

    Args:
        w (str): The first unigram in the bigram
        w_prime (str): The second unigram in the bigram
        unigram_counts (dict): A dictionary with unigrams (str) as keys and
                            token counts (int) as values
        unigram_log_thetas(dict): A dictionary with unigrams (str) as keys and
                                log probabilities (float) as values. Contains
                                a special "<UNSEEN>" key for the probability
                                of unseen unigrams
        bigram_counts (dict): A dictionary with bigrams (tuples of two
                            unigrams (str)) and bigram counts (int) as values
        beta (float): The bigram smoothing parameter

    Returns:
        float: The log probability of the bigram
    """
    bigram_count = bigram_counts.get((w, w_prime), 0)
    theta_w_prime = math.exp(
        unigram_log_thetas.get(w_prime, unigram_log_thetas["<UNSEEN>"])
    )
    w_count = unigram_counts.get(w, 0)
    return math.log((bigram_count + beta * theta_w_prime) / (w_count + beta))


def calculate_log_probability(
    test_data, unigram_counts, unigram_log_thetas, bigram_counts, beta
):
    """
    Calculates the log probability of a dataset.

    Args:
        test_data (list): A list of padded sentences, where each padded
                        sentence is a list of tokens (str)
        unigram_counts (dict): A dictionary with unigrams (str) as keys and
                            token counts (int) as values
        unigram_log_thetas(dict): A dictionary with unigrams (str) as keys and
                                log probabilities (float) as values. Contains
                                a special "<UNSEEN>" key for the probability
                                of unseen unigrams
        bigram_counts (dict): A dictionary with bigrams (tuples of two
                            unigrams (str)) and bigram counts (int) as values
        beta (float): The bigram smoothing parameter

    Returns:
        float: The log probability of the dataset
    """
    log_probability = 0
    for sentence in test_data:
        for w in range(len(sentence) - 1):
            log_probability += compute_bigram_log_theta(
                sentence[w],
                sentence[w + 1],
                unigram_counts,
                unigram_log_thetas,
                bigram_counts,
                beta,
            )
    return log_probability


def find_best_beta(
    unigram_counts,
    unigram_log_thetas,
    bigram_counts,
    dev_data,
    a,
    b,
    tolerance=1e-5,
):
    """
    Golden-selection search to find the value of beta
    that maximizes the log probability of a held out dataset.

    Adapted from https://en.wikipedia.org/wiki/Golden-section_search

    Args:
        unigram_counts (dict): A dictionary with unigrams (str) as keys
                            and token counts (int) as values
        unigram_log_thetas(dict): A dictionary with unigrams (str) as keys and
                                log probabilities (float) as values. Contains
                                a special "<UNSEEN>" key for the probability
                                of unseen unigrams
        bigram_counts (dict): A dictionary with bigrams (tuples of two
                            unigrams (str)) and bigram counts (int) as values
        dev_data (list): A list of padded sentences, where each padded
                        sentence is a list of tokens (str)
        a (float): The lower bound of the search interval
        b (float): The upper bound of the search interval
        tolerance (float, optional): Tolerance parameter

    Returns:
        float: The optimal parameter for add-beta smoothing
    """
    invphi = (math.sqrt(5) - 1) / 2
    while b - a > tolerance:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        if calculate_log_probability(
            dev_data, unigram_counts, unigram_log_thetas, bigram_counts, c
        ) > calculate_log_probability(
            dev_data, unigram_counts, unigram_log_thetas, bigram_counts, d
        ):
            b = d
        else:
            a = c
    return round((b + a) / 2, 1)


class BigramLM:
    """
    A bigram language model.

    Attributes:
        alpha (float): The unigram smoothing parameter
        beta (float): The bigram smoothing parameter
        unigram_counts (dict): A dictionary with unigrams (str) as keys
                            and token counts (int) as values
        unigram_log_thetas(dict): A dictionary with unigrams (str) as keys and
                                log probabilities (float) as values. Contains
                                a special "<UNSEEN>" key for the probability
                                of unseen unigrams
        bigram_counts (dict): A dictionary with bigrams (tuples of two
                            unigrams (str)) and bigram counts (int) as values
    """

    def __init__(self):
        self.alpha = 1.6
        self.beta = 1
        self.unigram_counts = {}
        self.unigram_log_thetas = {}
        self.bigram_counts = {}

    def train(self, train_data, alpha, dev_data=None, beta=None):
        """
        Trains the model by calculating unigram and bigram counts and unigram
        probabilities. Either a specified beta value or a development dataset
        must be provided. If a dev set and no value for beta is provided, the
        function automatically finds the optimal beta value.

        Args:
            train_data (list): A list of padded sentences, where each padded
                        sentence is a list of tokens (str)
            alpha (float): The unigram smoothing parameter
            dev_data (list, optional): A list of padded sentences, where each
                                    padded sentence is a list of tokens (str)
            beta (float, optional): The bigram smoothing parameter
        """
        self.alpha = alpha
        self.unigram_counts = unigram.count_unigrams(
            [token for sentence in train_data for token in sentence]
        )
        self.unigram_log_thetas = unigram.compute_unigram_log_thetas(
            self.unigram_counts, self.alpha
        )
        self.bigram_counts = count_bigrams(train_data)

        if beta:
            self.beta = beta
        elif dev_data:
            self.beta = find_best_beta(
                self.unigram_counts,
                self.unigram_log_thetas,
                self.bigram_counts,
                dev_data,
                0.1,
                1000,
            )
        else:
            raise ValueError("Please provide the dev dataset or a value for beta")

    def log_probability(self, test_data):
        return calculate_log_probability(
            test_data,
            self.unigram_counts,
            self.unigram_log_thetas,
            self.bigram_counts,
            self.beta,
        )

