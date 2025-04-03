"""Unit tests for unigram model."""

import math
import unittest

import unigram


class UnigramTest(unittest.TestCase):
    def test_seen_token(self):
        train_tokens = ["My", "name", "is", "Bob"]
        unigram_lm = unigram.UnigramLM()
        unigram_lm.train(train_tokens, alpha=1)
        predicted_probability = unigram_lm.log_probability(["Bob"])

        # See unigram.compute_unigram_log_thetas for equation
        calculated_probability = math.log((1 + 1) / (4 + 1 * 4))

        self.assertAlmostEqual(predicted_probability, calculated_probability)

    def test_unseen_token(self):
        train_tokens = ["now", "brown", "cow", "cow"]
        unigram_lm = unigram.UnigramLM()
        unigram_lm.train(train_tokens, alpha=2)
        predicted_probability = unigram_lm.log_probability(["moose"])

        calculated_probability = math.log((0 + 2) / (4 + 2 * 3))

        self.assertAlmostEqual(predicted_probability, calculated_probability)

    def test_multiple_tokens(self):
        train_tokens = ["now", "brown", "cow", "cow"]
        unigram_lm = unigram.UnigramLM()
        unigram_lm.train(train_tokens, alpha=1)
        predicted_probability = unigram_lm.log_probability(["brown", "cow"])

        # Log prob b + log prob c
        calculated_probability = math.log((1 + 1) / (4 + 1 * 3)) + math.log(
            (2 + 1) / (4 + 1 * 3)
        )

        self.assertAlmostEqual(predicted_probability, calculated_probability)


if __name__ == "__main__":
    unittest.main()
