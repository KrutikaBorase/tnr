import unittest

from task2.ranking import retrieval_ranking


class RetrievalRankingTests(unittest.TestCase):
    def test_top1_and_mrr_at_five_include_misses(self):
        details = [
            {"gold_passage": "A B", "top5": [{"text": "a  b"}]},
            {"gold_passage": "target", "top5": [{"text": "other"}, {"text": "target"}]},
            {"gold_passage": "missing", "top5": [{"text": "other"}]},
        ]
        top1, mrr = retrieval_ranking(details)
        self.assertEqual(top1, 1)
        self.assertAlmostEqual(mrr, 0.5)

    def test_empty_questions_rejected(self):
        with self.assertRaises(ValueError):
            retrieval_ranking([])


if __name__ == "__main__":
    unittest.main()
