"""Unit tests for bigram model."""

import math
import unittest

import bigram


class BigramTest(unittest.TestCase):
    def test_train_and_predict(self):
        train_data = [
            ["*START*", "I", "am", "Sam", "*STOP*"],
            ["*START*", "Sam", "I", "am", "*STOP*"],
            [
                "*START*",
                "I",
                "do",
                "not",
                "like",
                "green",
                "eggs",
                "and",
                "ham",
                "*STOP*",
            ],
        ]
        bigram_lm = bigram.BigramLM()
        bigram_lm.train(train_data, alpha=1, beta=1)
        predicted_probability = bigram_lm.log_probability([["*START*", "I"]])

        w_prime_theta = (3 + 1) / (20 + 1 * 12)
        calculated_probability = math.log((2 + 1 * w_prime_theta) / (3 + 1))

        self.assertAlmostEqual(predicted_probability, calculated_probability)


if __name__ == "__main__":
    unittest.main()
