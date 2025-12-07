from dataset_processing import DatasetPreProcessor
import unittest
import tempfile
import shutil
import os
import pandas as pd


class TestDatasetPreProcessor(unittest.TestCase):
    def setUp(self):
        """Creates a temporary directory and a JSONL dataset file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_filepath = os.path.join(self.temp_dir, "test_dataset.jsonl")
        self.columns_to_process = [
            "declarative_sentence",
            "question",
            "answer",
        ]
        data = [
            {
                "id": 1,
                "declarative_sentence": "  The quick brown fox.  ",
                "question": "What does the fox say?",
                "answer": "Ring ding ding",
                "question_category": "Subject-Verb-Object",
                "statement_category": "Subject-Verb-Object",
            },
            {
                "id": 2,
                "declarative_sentence": "New York is a big city.",
                "question": "Where is New York?",
                "answer": "In the United States Of America",
                "question_category": "Subject-Verb-Object",
                "statement_category": "Subject-Verb-Object",
            },
            {
                "id": 3,
                "declarative_sentence": "This sentence has more than five words in it.",
                "question": "Is this sentence long?",
                "answer": "Yes",
                "question_category": "Subject-Verb-Object",
                "statement_category": "Subject-Verb-Object",
            },
            {
                "id": 4,
                "declarative_sentence": "This sentence has a verylongwordinit.",
                "question": "Does it have a long word?",
                "answer": "Indeed",
                "question_category": "Subject-Verb-Object",
                "statement_category": "Subject-Verb-Object",
            },
            {
                "id": 5,
                "declarative_sentence": "the English call -ed New Amsterdam New York after its capture",
                "question": "What did the English call New Amsterdam after its capture?",
                "answer": "New York",
                "question_category": "Subject-Verb-Object",
                "statement_category": "Subject-Verb-Object",
            },
            {
                "id": 6,
                "declarative_sentence": "a copper statue of Christ be in front of the Notre Dame Main Building",
                "question": "? what be in front of the Notre Dame Main Building",
                "answer": "a copper statue of Christ",
                "question_category": "Subject-Verb-Object",
                "statement_category": "Subject-Verb-Object",
            },
            {
                "id": 7,
                "declarative_sentence": "a dog is a mammal",
                "question": "? tell me a mammal",
                "answer": "dog",
                "question_category": "Subject-Verb-Object",
                "statement_category": "Subject-Verb-Object",
            },
        ]

        with open(self.dataset_filepath, "w") as f:
            for item in data:
                f.write(str(item).replace("'", '"') + "\n")

        dataset = pd.read_json(self.dataset_filepath, lines=True)
        for col in self.columns_to_process:
            dataset[f"{col}_formatted"] = dataset[col]

        self.preprocessor = DatasetPreProcessor(
            dataset=dataset,
            max_words_limit=5,
            max_word_length_limit=10,
            columns_to_process=self.columns_to_process,
        )

    def tearDown(self):
        """Removes the temporary directory and its contents."""
        shutil.rmtree(self.temp_dir)

    def test_remove_specialCharacters(self):
        text_to_process = "what be in front of the Notre Dame Main Building ?"
        expected_result = "what be in front of the Notre Dame Main Building"
        actual_result = self.preprocessor.remove_special_characters(text_to_process)
        self.assertEqual(actual_result, expected_result)

    def test_add_question_mark(self):
        text_to_process = "what be in front of the Notre Dame Main Building"
        expected_result = "? what be in front of the Notre Dame Main Building"
        actual_result = self.preprocessor.add_question_mark_to_start(text_to_process)
        self.assertEqual(actual_result, expected_result)

    def test_remove_whitespace_from_dataframe(self):
        """Tests that leading/trailing whitespace is removed from string columns."""
        self.preprocessor.remove_whitespace_from_dataframe()
        expected = "The quick brown fox."
        actual = self.preprocessor.dataset.loc[0, "declarative_sentence"]
        self.assertEqual(actual, expected)

    def test_join_concurrent_capitalized_words(self):
        """Tests that consecutive capitalized words are joined with hyphens."""
        self.preprocessor.join_entity_words()
        expected_sentence = (
            "the English call -ed New_Amsterdam New_York after its capture"
        )
        actual_sentence = self.preprocessor.dataset.loc[
            4, "declarative_sentence_formatted"
        ]
        self.assertEqual(actual_sentence, expected_sentence)

        expected_answer = "New_York"
        actual_answer = self.preprocessor.dataset.loc[4, "answer_formatted"]
        self.assertEqual(actual_answer, expected_answer)

    def test_join_concurrent_capitalized_words_without_the(self):
        """Tests that consecutive capitalized words are joined with hyphens."""
        self.preprocessor.join_entity_words()
        expected_sentence = (
            "a copper statue of Christ be in front of the Notre_Dame Main Building"
        )
        actual_sentence = self.preprocessor.dataset.loc[
            5, "declarative_sentence_formatted"
        ]
        self.assertEqual(actual_sentence, expected_sentence)

        expected_answer = "a copper statue of Christ"
        actual_answer = self.preprocessor.dataset.loc[5, "answer_formatted"]
        self.assertEqual(actual_answer, expected_answer)

    def test_filter_dataset_by_limits_word_count(self):
        """Tests filtering based on the maximum number of words."""
        initial_rows = len(self.preprocessor.dataset)
        # This row should be removed: "This sentence has more than five words in it."
        self.preprocessor.max_words_limit = 6
        self.preprocessor.filter_dataset_by_limits()
        self.assertLess(len(self.preprocessor.dataset), initial_rows)
        self.assertNotIn(3, self.preprocessor.dataset["id"].values)

    def test_filter_dataset_by_limits_word_length(self):
        """Tests filtering based on the maximum word length."""
        initial_rows = len(self.preprocessor.dataset)
        # This row should be removed: "This sentence has a verylongwordinit."
        self.preprocessor.max_word_length_limit = 15
        self.preprocessor.filter_dataset_by_limits()
        self.assertLess(len(self.preprocessor.dataset), initial_rows)
        self.assertNotIn(4, self.preprocessor.dataset["id"].values)

    def test_preprocess_data(self):
        """Tests the full preprocessing pipeline."""
        self.preprocessor.preprocess_data()
        # After all preprocessing with limits (5 words, 10 length), only row 1 should remain
        self.assertEqual(len(self.preprocessor.dataset), 1)
        self.assertEqual(self.preprocessor.dataset.iloc[0]["id"], 7)

        # Check that whitespace has been handled
        processed_row = self.preprocessor.dataset.iloc[0]
        self.assertEqual(
            processed_row["declarative_sentence_formatted"],
            "a dog be a mammal",
        )

    def test_select_pretraining_data(self):
        """Tests selecting a subset of the dataset for pretraining."""
        # Do not preprocess here to keep 5 rows and avoid reducing dataset to 1
        percent = 50
        original_len = len(self.preprocessor.dataset)
        expected_selected = int(original_len * percent / 100)

        self.preprocessor.select_pretraining_data(
            percentage_of_pretraining_samples=percent
        )

        # Construct the selected subset and perform assertions
        self.assertIn("is_pretraining", self.preprocessor.dataset.columns)
        self.assertTrue(self.preprocessor.dataset["is_pretraining"].dtype == bool)

        pretraining_data = self.preprocessor.dataset[
            self.preprocessor.dataset["is_pretraining"] == True
        ]
        self.assertEqual(len(pretraining_data), expected_selected)
        self.assertIn("declarative_sentence_formatted", pretraining_data.columns)
        self.assertIn("question_formatted", pretraining_data.columns)
        self.assertIn("answer_formatted", pretraining_data.columns)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)