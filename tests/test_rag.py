import json
import unittest
from pathlib import Path

from task2.rag import (
    CHUNK_CONFIGS,
    CORPUS,
    QUESTIONS,
    ensure_corpus,
    extract_sections,
    make_chunks,
)


class RfcPassageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_corpus()
        source = CORPUS.read_text(encoding="utf-8")
        cls.sections = extract_sections(source)
        cls.questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    def test_all_gold_passages_are_in_their_cited_sections(self):
        self.assertEqual(len(self.questions), 15)
        for question in self.questions:
            with self.subTest(question=question["id"]):
                section = self.sections[question["gold_section"]]
                self.assertIn(question["gold_passage"], section)

    def test_gold_passages_survive_both_chunk_sizes(self):
        for size, overlap in CHUNK_CONFIGS:
            chunks = make_chunks(self.sections, size, overlap)
            for question in self.questions:
                with self.subTest(size=size, question=question["id"]):
                    self.assertTrue(
                        any(
                            chunk["section"] == question["gold_section"]
                            and question["gold_passage"] in chunk["text"]
                            for chunk in chunks
                        )
                    )


if __name__ == "__main__":
    unittest.main()
