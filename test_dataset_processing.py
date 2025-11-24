from dataset_processing import DatasetPreProcessor
from commands import (
    AnnabellQuestionCommandGenerator,
    AnnabellBaseCommandGenerator,
    AnnabellTestingCommandGenerator,
    AnnabellTrainingCommandGenerator,
)
import unittest
import tempfile
import shutil
import os
import pandas as pd


class TestAbstractAnnabellCommandGenerator(unittest.TestCase):

    def setUp(self):
        """Set up a common instance for testing."""
        self.sample_id = "test_01"
        self.declarative_sentence = "the sky is blue with patches of grey"
        self.question = "what color is the sky?"
        self.short_answer = "blue"
        self.long_answer = "blue with patches of grey"
        self.long_question = (
            "? what was the trade -ing post that precede -d New-York-City call -ed"
        )
        self.long_declarative_sentence = "the trade -ing post that precede -d New-York-City was call -ed New-Amsterdam"

    def test_remove_stopwords(self):
        """Test the static method remove_stopwords."""
        self.assertEqual(
            AnnabellQuestionCommandGenerator.remove_stopwords(
                "this is a test sentence"
            ),
            "test sentence",
        )
        self.assertEqual(
            AnnabellQuestionCommandGenerator.remove_stopwords("missing stopwords"),
            "missing stopwords",
        )
        self.assertEqual(AnnabellQuestionCommandGenerator.remove_stopwords(""), "")

    def test_remove_suffixes(self):
        """Test the static method remove_suffixes."""
        self.assertEqual(
            AnnabellQuestionCommandGenerator.remove_suffixes("this is for test -ing"),
            "this is for test",
        )
        self.assertEqual(
            AnnabellQuestionCommandGenerator.remove_suffixes("no suffixes here"),
            "no suffixes here",
        )
        self.assertEqual(AnnabellQuestionCommandGenerator.remove_suffixes(""), "")

    def test_remove_question_mark(self):
        """Test the static method remove_question_mark."""
        self.assertEqual(
            AnnabellQuestionCommandGenerator.remove_question_mark("is this a test?"),
            "is this a test",
        )
        self.assertEqual(
            AnnabellQuestionCommandGenerator.remove_question_mark("? is this a test"),
            "is this a test",
        )
        self.assertEqual(
            AnnabellQuestionCommandGenerator.remove_question_mark("no question mark"),
            "no question mark",
        )
        self.assertEqual(AnnabellQuestionCommandGenerator.remove_question_mark("?"), "")

    def test_create_list_of_commands_short_answer(self):
        """Test command generation for an answer with fewer than 4 words."""
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, self.declarative_sentence, self.question, self.short_answer
        )
        commands = generator.create_list_of_commands()

        expected_commands = [
            "#id: test_01",
            "the sky is blue with patches of grey",
            "\n",
            "what color is the sky?",
            ".wg sky",
            ".ph the sky is blue with patches of grey",
            ".wg blue",
            ".rw",
            "\n",
        ]

        self.assertEqual(commands, expected_commands)

    def test_create_list_of_commands_long_answer(self):
        """Test command generation for an answer with more than 3 words."""
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, self.declarative_sentence, self.question, self.long_answer
        )
        commands = generator.create_list_of_commands()

        expected_commands = [
            "#id: test_01",
            "the sky is blue with patches of grey",
            "\n",
            "what color is the sky?",
            ".wg sky",
            ".ph the sky is blue with patches of grey",
            ".wg blue with patches",
            ".prw",
            ".wg of grey",
            ".rw",
            "\n",
        ]

        self.assertEqual(commands, expected_commands)

    def test_write_question_commands_for_phrase(self):
        """Test the write_question_commands_for_phrase method."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, max_words=5
        )
        generator.write_commands()
        expected_commands = [
            "what color is the sky?",
            ".sctx what color is the sky?",
            ".ph the sky is blue with",
            ".wg sky",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_question_commands_for_context(self):
        """Test the write_question_commands_for_context method."""
        generator = AnnabellQuestionCommandGenerator(
            self.long_declarative_sentence,
            self.long_question,
            max_words=5,
        )
        generator.write_commands()
        expected_commands = [
            "? what was the trade",
            "-ing post that precede -d",
            "New-York-City call -ed",
            ".sctx ? what was the trade",
            ".wg trade",
            ".sctx -ing post that precede -d",
            ".wg post",
            ".ph the trade -ing post that",
            ".sctx precede -d New-York-City was call",
            ".wg precede",
            ".sctx New-York-City call -ed",
            ".ph the trade -ing post that",
            ".sctx precede -d New-York-City was call",
            ".wg New-York-City",
            ".ph the trade -ing post that",
            ".sctx precede -d New-York-City was call",
            ".wg call",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_question_commands_wg_not_in_phrase(self):
        """consider the following inout phrase:
        a golden statue of the_Virgin_Mary sit on top of the_Main_Building
        at Notre_Dame
        and this question and commands:

        ? what sit on top of the_Main_Building at Notre_Dame
        .wg sit
        .wg top
        .wg the_Main_Building
        .wg Notre_Dame

        the following phrase retrieval fails because Notre_Dame is not in the phrase
        .ph a golden statue of the_Virgin_Mary sit on top of the_Main_Building

        the solution is to ensure that .wg commands are only created for words present in the lookup phrase.
        then for any remaining words in the question context that are not in the phrase, move to the subsequent phrases in the context and apply the same logic
        """

        question = "? what sit on top of the_Main_Building at Notre_Dame"
        declarative_sentence = "a golden statue of the_Virgin_Mary sit on top of the_Main_Building at Notre_Dame"

        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence,
            question,
            max_words=10,
        )
        generator.write_question_commands()

        expected_commands = [
            ".sctx ? what sit on top of the_Main_Building at Notre_Dame",
            ".wg sit",
            ".wg top",
            ".wg the_Main_Building",
            ".ph a golden statue of the_Virgin_Mary sit on top of the_Main_Building",
            ".sctx at Notre_Dame",
            ".wg Notre_Dame",
        ]

        self.assertEqual(expected_commands, generator.commands)

    def test_write_question_commands_short_question(self):
        """Test the write_question_commands method with a short question."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, max_words=5
        )
        generator.write_question_commands()
        expected_commands = [
            ".sctx what color is the sky?",
            ".ph the sky is blue with",
            ".wg sky",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_question_commands_long_question(self):
        """Test the write_question_commands method with a long question."""
        generator = AnnabellQuestionCommandGenerator(
            self.long_declarative_sentence,
            self.long_question,
            max_words=5,
        )
        generator.write_question_commands()
        expected_commands = [
            ".sctx ? what was the trade",
            ".wg trade",
            ".sctx -ing post that precede -d",
            ".wg post",
            ".ph the trade -ing post that",
            ".sctx precede -d New-York-City was call",
            ".wg precede",
            ".sctx New-York-City call -ed",
            ".ph the trade -ing post that",
            ".sctx precede -d New-York-City was call",
            ".wg New-York-City",
            ".ph the trade -ing post that",
            ".sctx precede -d New-York-City was call",
            ".wg call",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_question_short_question(self):
        """Test the write_question method with a short question."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, max_words=5
        )
        generator.write_question()
        self.assertEqual(generator.commands, [self.question])

    def test_write_question_long_question(self):
        """Test the write_question method with a long question."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence,
            self.long_question,
            max_words=5,
        )
        generator.write_question()
        expected_phrases = [
            "? what was the trade",
            "-ing post that precede -d",
            "New-York-City call -ed",
        ]
        self.assertEqual(generator.commands, expected_phrases)

    def test_write_answer_commands_short_sentence_short_answer(self):
        """Test write_answer_commands with a short sentence and short answer."""
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, "the sky is blue", self.question, "blue", max_words=10
        )
        generator.write_answer_commands()
        expected_commands = [".ph the sky is blue", ".wg blue", ".rw"]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_answer_commands_short_sentence_long_answer(self):
        """Test write_answer_commands with a short sentence and long answer."""
        declarative_sentence = "the color of the sky is blue and sometimes grey"
        answer = "blue and sometimes grey"
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, declarative_sentence, self.question, answer, max_words=10
        )
        generator.write_answer_commands()
        expected_commands = [
            f".ph {declarative_sentence}",
            ".wg blue and sometimes",
            ".prw",
            ".wg grey",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_answer_commands_long_sentence(self):
        """Test write_answer_commands with a long sentence where the answer is split across phrases."""
        declarative_sentence = "the sky is a brilliant blue with some patches of grey"
        answer = "blue with some patches of grey"
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, declarative_sentence, self.question, answer, max_words=5
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ph the sky is a brilliant",
            ".sctx blue with some patches of",
            ".wg blue with some",
            ".prw",
            ".wg patches of",
            ".prw",
            ".sctx grey",
            ".wg grey",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_answer_commands_long_sentence_2(self):
        """Test write_answer_commands with a long sentence where the answer is split across phrases."""
        declarative_sentence = (
            "the Grotto at Notre_Dame be a marian place of prayer and reflection"
        )
        answer = "a marian place of prayer and reflection"
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, declarative_sentence, self.question, answer, max_words=10
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ph the Grotto at Notre_Dame be a marian place of prayer",
            ".wg a marian place",
            ".prw",
            ".wg of prayer",
            ".prw",
            ".sctx and reflection",
            ".wg and reflection",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)


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
        self.assertEqual(self.preprocessor.dataset.iloc[0]["id"], 2)

        # Check that whitespace has been handled
        processed_row = self.preprocessor.dataset.iloc[0]
        self.assertEqual(
            processed_row["declarative_sentence_formatted"],
            "New_York be a big city",
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


class TestAnnabellTestCommandGenerator(unittest.TestCase):
    def setUp(self):
        """Set up a common instance for testing."""
        self.sample_id = "5733be284776f41900661180"
        self.question = "? the Basilica of the sacred heart at Notre_Dame be beside to which structure"
        self.command_generator = AnnabellTestingCommandGenerator(
            self.sample_id, self.question
        )

    def test_write_testing_command(self):

        expected_commands = [
            "#id: 5733be284776f41900661180",
            "? the Basilica of the sacred heart at Notre_Dame",
            "be beside to which structure",
            ".x",
            "#END OF TESTING SAMPLE",
            "\n",
        ]
        self.command_generator.create_list_of_commands()
        self.assertEqual(self.command_generator.commands, expected_commands)


class TestAnnabellTrainingCommandGenerator(unittest.TestCase):
    def setUp(self):
        """Set up a common instance for testing."""
        self.sample_id = "5733be284776f41900661180"
        self.declarative_sentence = (
            "the Basilica of the Sacred Heart at Notre_Dame be beside the_Main_Building"
        )
        self.command_generator = AnnabellTrainingCommandGenerator(
            self.sample_id, self.declarative_sentence
        )

    def test_write_training_command(self):

        expected_commands = [
            "#id: 5733be284776f41900661180",
            "the Basilica of the Sacred Heart at Notre_Dame be",
            "beside the_Main_Building",
            "\n",
        ]
        self.command_generator.create_list_of_commands()
        self.assertEqual(self.command_generator.commands, expected_commands)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)