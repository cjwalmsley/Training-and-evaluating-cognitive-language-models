import unittest
from dataset_processing import AnnabellCommandGenerator


class TestAnnabellCommandGenerator(unittest.TestCase):

    def setUp(self):
        """Set up a common instance for testing."""
        self.sample_id = "test_01"
        self.declarative_sentence = "the sky is blue with patches of grey"
        self.question = "what color is the sky?"
        self.short_answer = "blue"
        self.long_answer = "blue with patches of grey"
        self.long_question = "? what was the trade -ing post that precede -d New-York-City call -ed"

    def test_remove_stopwords(self):
        """Test the static method remove_stopwords."""
        self.assertEqual(
            AnnabellCommandGenerator.remove_stopwords("this is a test sentence"),
            "test sentence"
        )
        self.assertEqual(
            AnnabellCommandGenerator.remove_stopwords("missing stopwords"),
            "missing stopwords"
        )
        self.assertEqual(AnnabellCommandGenerator.remove_stopwords(""), "")

    def test_remove_suffixes(self):
        """Test the static method remove_suffixes."""
        self.assertEqual(
            AnnabellCommandGenerator.remove_suffixes("this is for test -ing"),
            "this is for test"
        )
        self.assertEqual(
            AnnabellCommandGenerator.remove_suffixes("no suffixes here"),
            "no suffixes here"
        )
        self.assertEqual(AnnabellCommandGenerator.remove_suffixes(""), "")

    def test_remove_question_mark(self):
        """Test the static method remove_question_mark."""
        self.assertEqual(
            AnnabellCommandGenerator.remove_question_mark("is this a test?"),
            "is this a test"
        )
        self.assertEqual(
            AnnabellCommandGenerator.remove_question_mark("? is this a test"),
            "is this a test"
        )
        self.assertEqual(
            AnnabellCommandGenerator.remove_question_mark("no question mark"),
            "no question mark"
        )
        self.assertEqual(AnnabellCommandGenerator.remove_question_mark("?"), "")

    def test_create_list_of_commands_short_answer(self):
        """Test command generation for an answer with fewer than 4 words."""
        generator = AnnabellCommandGenerator(
            self.sample_id, self.declarative_sentence, self.question, self.short_answer
        )
        commands = generator.create_list_of_commands()

        expected_commands = [
            "#id: test_01",
            "the sky is blue with patches of grey",
            "\n",
            "what color is the sky?",
            ".wg color",
            ".wg sky",
            ".ph the sky is blue with patches of grey",
            ".wg blue",
            ".rw",
            "\n"
        ]

        self.assertEqual(commands, expected_commands)

    def test_create_list_of_commands_long_answer(self):
        """Test command generation for an answer with more than 3 words."""
        generator = AnnabellCommandGenerator(
            self.sample_id, self.declarative_sentence, self.question, self.long_answer
        )
        commands = generator.create_list_of_commands()

        expected_commands = [
            "#id: test_01",
            "the sky is blue with patches of grey",
            "\n",
            "what color is the sky?",
            ".wg color",
            ".wg sky",
            ".ph the sky is blue with patches of grey",
            ".wg blue with patches",
            ".prw",
            ".wg of grey",
            ".rw",
            "\n"
        ]

        self.assertEqual(commands, expected_commands)

    def test_write_question_commands_for_phrase(self):
        """Test the write_question_commands_for_phrase method."""
        generator = AnnabellCommandGenerator(
            self.sample_id, self.declarative_sentence, self.question, self.short_answer
        )
        generator.write_question_commands_for_phrase("what color is the sky?")
        expected_commands = [
            ".wg color",
            ".wg sky",
        ]
        self.assertEqual(generator.commands, expected_commands)

    def test_write_question_commands_for_context(self):
        """Test the write_question_commands_for_context method."""
        generator = AnnabellCommandGenerator(
            self.sample_id, self.declarative_sentence, self.long_question, self.short_answer, max_words=5
        )
        generator.write_question_commands_for_context(self.long_question)
        expected_commands = [
            ".sctx ? what was the trade",
            ".wg trade",
            ".sctx -ing post that precede -d",
            ".wg post",
            ".wg precede",
            ".sctx New-York-City call -ed",
            ".wg New-York-City",
            ".wg call",
        ]
        self.assertEqual(generator.commands, expected_commands)

    def test_write_question_commands_short_question(self):
        """Test the write_question_commands method with a short question."""
        generator = AnnabellCommandGenerator(
            self.sample_id, self.declarative_sentence, self.question, self.short_answer
        )
        generator.write_question_commands()
        expected_commands = [
            ".wg color",
            ".wg sky",
        ]
        self.assertEqual(generator.commands, expected_commands)

    def test_write_question_commands_long_question(self):
        """Test the write_question_commands method with a long question."""
        generator = AnnabellCommandGenerator(
            self.sample_id, self.declarative_sentence, self.long_question, self.short_answer, max_words=5
        )
        generator.write_question_commands()
        expected_commands = [
            ".sctx ? what was the trade",
            ".wg trade",
            ".sctx -ing post that precede -d",
            ".wg post",
            ".wg precede",
            ".sctx New-York-City call -ed",
            ".wg New-York-City",
            ".wg call",
        ]
        self.assertEqual(generator.commands, expected_commands)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)