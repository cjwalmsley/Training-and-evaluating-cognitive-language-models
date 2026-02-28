import unittest
from unittest.mock import patch

from numpy.ma.testutils import assert_equal

from annabell_utilities import AnnabellLogfileInterpreter
from commands import (
    AnnabellQuestionCommandGenerator,
    AnnabellBaseCommandGenerator,
    AnnabellTestingCommandGenerator,
    AnnabellTrainingCommandGenerator,
    AnnabellQuestionContext,
    AnnabellDeclarativeContext,
    AnnabellAnswerContext,
    AnnabellAnswerCommandGenerator,
    LIFOQueue,
    MissingAnswerWordsException,
    ToFewLookupWordGroupsException,
)


class AbstractAnnabellTestCase(unittest.TestCase):
    """Abstract base class for Annabell command generator tests."""

    def setUp(self):
        self.log_stats_patcher = patch(
            "commands.global_config.log_stats", return_value=False
        )
        self.exclude_patcher = patch(
            "commands.global_config.exclude_samples_with_fewer_than_2_lookups",
            return_value=False,
        )
        self.write_non_lookup_commands_patcher = patch(
            "commands.global_config.write_non_lookup_commands",
            return_value=True,
        )

        self.goal_stack_limit_patcher = patch(
            "commands.global_config.goal_stack_limit",
            return_value=100,
        )

        # 2. Start the patchers and store the mock objects
        self.mock_log_stats = self.log_stats_patcher.start()
        self.mock_exclude = self.exclude_patcher.start()
        self.mock_write_non_lookup_commands = (
            self.write_non_lookup_commands_patcher.start()
        )
        self.mock_goal_stack_limit = self.goal_stack_limit_patcher.start()

    def tearDown(self):
        # Stop the patchers after each test
        self.log_stats_patcher.stop()
        self.exclude_patcher.stop()
        self.write_non_lookup_commands_patcher.stop()
        self.goal_stack_limit_patcher.stop()


class TestAbstractAnnabellCommandGenerator(AbstractAnnabellTestCase):

    def setUp(self):
        super().setUp()
        """Set up a common instance for testing."""
        self.sample_id = "test_01"
        self.declarative_sentence = "the sky is blue with patches of grey"
        self.question = "? what color is the sky"

        self.short_answer = "blue"
        self.long_answer = "blue with patches of grey"

        self.long_question = (
            "? what was the trade -ing post that precede -d New-York-City call -ed"
        )
        self.long_declarative_sentence = "the trade -ing post that precede -d New-York-City was call -ed New-Amsterdam"

    def test_remove_stopwords(self):
        """Test the static method remove_stopwords."""
        self.assertEqual(
            AnnabellQuestionContext.remove_stopwords("this is a test sentence"),
            "test sentence",
        )
        self.assertEqual(
            AnnabellQuestionContext.remove_stopwords("missing stopwords"),
            "missing stopwords",
        )
        self.assertEqual(AnnabellQuestionContext.remove_stopwords(""), "")

    def test_remove_suffixes(self):
        """Test the static method remove_suffixes."""
        self.assertEqual(
            AnnabellQuestionContext.remove_suffixes("this is for test -ing"),
            "this is for test",
        )
        self.assertEqual(
            AnnabellQuestionContext.remove_suffixes("no suffixes here"),
            "no suffixes here",
        )
        self.assertEqual(AnnabellQuestionContext.remove_suffixes(""), "")

    def test_remove_question_mark(self):
        """Test the static method remove_question_mark."""
        self.assertEqual(
            AnnabellQuestionContext.remove_question_mark("is this a test?"),
            "is this a test",
        )
        self.assertEqual(
            AnnabellQuestionContext.remove_question_mark("? is this a test"),
            "is this a test",
        )
        self.assertEqual(
            AnnabellQuestionContext.remove_question_mark("no question mark"),
            "no question mark",
        )
        self.assertEqual(AnnabellQuestionContext.remove_question_mark("?"), "")

    def test_create_list_of_commands_short_answer(self):
        """Test command generation for an answer with fewer than 4 words."""
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, self.declarative_sentence, self.question, self.short_answer
        )
        commands = generator.create_list_of_commands()

        expected_commands = [
            "#id: test_01",
            "the sky is blue with patches of grey",
            f"{AnnabellLogfileInterpreter.end_of_declaration_string()}",
            "\n",
            "? what color is the sky",
            f"{AnnabellLogfileInterpreter.end_of_question_string()}",
            ".pg sky",
            ".ggp",
            ".ph the sky is blue with patches of grey",
            ".drop_goal",
            ".wg blue",
            ".rw",
            "\n",
            f"{AnnabellLogfileInterpreter.end_of_commands_string()}",
            ".time",
            f"{AnnabellLogfileInterpreter.end_of_time_string()}",
        ]

        self.assertEqual(expected_commands, commands)

    def test_create_list_of_commands_long_answer(self):
        """Test command generation for an answer with more than 3 words."""
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, self.declarative_sentence, self.question, self.long_answer
        )
        commands = generator.create_list_of_commands()

        expected_commands = [
            "#id: test_01",
            "the sky is blue with patches of grey",
            f"{AnnabellLogfileInterpreter.end_of_declaration_string()}",
            "\n",
            "? what color is the sky",
            f"{AnnabellLogfileInterpreter.end_of_question_string()}",
            ".pg sky",
            ".ggp",
            ".ph the sky is blue with patches of grey",
            ".drop_goal",
            ".wg blue with patches of",
            ".prw",
            ".wg grey",
            ".rw",
            "\n",
            f"{AnnabellLogfileInterpreter.end_of_commands_string()}",
            ".time",
            f"{AnnabellLogfileInterpreter.end_of_time_string()}",
        ]

        self.assertEqual(expected_commands, commands)

    def test_write_question_commands_for_phrase(self):
        """Test the write_question_commands_for_phrase method."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, self.short_answer, max_words=5
        )
        generator.write_commands()
        expected_commands = [
            "? what color is the",
            "sky",
            "#END OF QUESTION",
            ".sctx sky",
            ".pg sky",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_question_commands_for_context(self):
        """Test the write_question_commands_for_context method."""
        generator = AnnabellQuestionCommandGenerator(
            self.long_declarative_sentence,
            self.long_question,
            self.short_answer,
            max_words=5,
        )

        generator.write_commands()
        expected_commands = [
            "? what was the trade",
            "-ing post that precede -d",
            "New-York-City call -ed",
            "#END OF QUESTION",
            ".sctx ? what was the trade",
            ".pg trade",
            ".sctx -ing post that precede -d",
            ".pg post",
            ".sctx New-York-City call -ed",
            ".pg New-York-City",
            ".pg call",
            ".wg precede",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_question_commands_wg_not_in_phrase(self):

        question = "? what sit on top of the_Main_Building at Notre_Dame"
        declarative_sentence = "a golden statue of the_Virgin_Mary sit on top of the_Main_Building at Notre_Dame"
        answer = "a golden statue of the_Virgin_Mary"

        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence,
            question,
            answer,
            max_words=10,
        )
        generator.write_question_commands()

        expected_commands = [
            ".pg sit on top",
            ".pg the_Main_Building",
            ".wg Notre_Dame",
        ]

        self.assertEqual(expected_commands, generator.commands)

    def test_write_question_commands_short_question(self):
        """Test the write_question_commands method with a short question."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, self.short_answer, max_words=5
        )
        generator.write_question_commands()
        expected_commands = [".sctx sky", ".pg sky"]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_question_commands_long_question(self):
        """Test the write_question_commands method with a long question."""
        generator = AnnabellQuestionCommandGenerator(
            self.long_declarative_sentence,
            self.long_question,
            self.short_answer,
            max_words=5,
        )

        # ("? what was the trade"
        # " -ing post that precede -d"
        # " New-York-City call -ed")

        # "the trade -ing post that"
        # " precede -d New-York-City was call"
        # " -ed New-Amsterdam"

        generator.write_question_commands()
        expected_commands = [
            ".sctx ? what was the trade",
            ".pg trade",
            ".sctx -ing post that precede -d",
            ".pg post",
            ".sctx New-York-City call -ed",
            ".pg New-York-City",
            ".pg call",
            ".wg precede",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_question(self):
        """Test the write_question method with a short question."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, self.short_answer, max_words=5
        )

        expected_commands = ["? what color is the", "sky"]

        generator.write_question()
        self.assertEqual(generator.commands, expected_commands)

        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, self.short_answer, max_words=6
        )

        expected_commands = ["? what color is the sky"]

        generator.write_question()
        self.assertEqual(generator.commands, expected_commands)

    def test_write_question_long_question(self):
        """Test the write_question method with a long question."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence,
            self.long_question,
            self.short_answer,
            max_words=5,
        )
        generator.write_question()
        expected_phrases = [
            "? what was the trade",
            "-ing post that precede -d",
            "New-York-City call -ed",
        ]
        self.assertEqual(generator.commands, expected_phrases)

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
            ".wg blue and sometimes grey",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_answer_commands_long_sentence_2(self):
        """Test write_answer_commands with a long sentence where the answer is split across phrases."""
        declarative_sentence = (
            "the Grotto at Notre_Dame be a marian place of prayer and reflection"
        )
        question = "? what is the grotto at Notre_Dame"
        answer = "a marian place of prayer and reflection"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=10
        )
        question_generator.write_commands()
        answer_generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=10
        )
        answer_generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph the Grotto at Notre_Dame be a marian place of prayer",
            ".drop_goal",
            ".wg a marian place of",
            ".prw",
            ".wg prayer",
            ".prw",
            ".sctx and reflection",
            ".wg and reflection",
            ".rw",
        ]
        self.assertEqual(expected_commands, answer_generator.commands)


class TestAnnabellTestCommandGenerator(AbstractAnnabellTestCase):
    def setUp(self):
        super().setUp()
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


class TestAnnabellTrainingCommandGenerator(AbstractAnnabellTestCase):
    def setUp(self):
        super().setUp()
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
            "#END OF DECLARATION",
            "\n",
        ]
        self.command_generator.create_list_of_commands()
        self.assertEqual(expected_commands, self.command_generator.commands)


class TestAnnabellQuestionContext(AbstractAnnabellTestCase):

    def setUp(self):
        super().setUp()
        self.declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        self.question = "? what sit on top of the Main_Building at Notre_Dame"
        self.context = AnnabellQuestionContext(self.question, max_words_per_phrase=9)

    def test_split_2_line_question_into_phrases(self):

        expected_phrases = ["? what sit on top of the Main_Building at", "Notre_Dame"]

        context_phrases = [phrase.text for phrase in self.context.phrases]

        self.assertEqual(context_phrases, expected_phrases)

    def test_word_group_chunks_matching_declarative_sentence(self):
        expected_chunks = [["sit", "on", "top"]]
        declarative_sentence_context = AnnabellDeclarativeContext(
            "a golden statue of the Virgin_Mary sit on top", max_words_per_phrase=9
        )
        question_context = AnnabellQuestionContext(
            "? what sit on top of the Main_Building at", max_words_per_phrase=9
        )
        word_group_chunks = question_context.word_group_chunks_matching_sentence(
            declarative_sentence_context
        )
        assert_equal(word_group_chunks, expected_chunks)

    def test_word_group_chunks_matching_sentence_non_consecutive_keywords(self):
        expected_chunks = [["New-York-City"], ["call"]]
        declarative_sentence_context = AnnabellDeclarativeContext(
            "precede -d New-York-City was call", max_words_per_phrase=9
        )
        question_context = AnnabellQuestionContext(
            "New-York-City call -ed", max_words_per_phrase=9
        )
        word_group_chunks = question_context.phrases[
            0
        ].word_group_chunks_matching_sentence(declarative_sentence_context.phrases[0])
        assert_equal(word_group_chunks, expected_chunks)


class TestAnnabellDeclarativeContext(AbstractAnnabellTestCase):

    def test_split_2_line_question_into_phrases(self):
        declarative_statement = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        expected_phrases = [
            "a golden statue of the Virgin_Mary sit on top",
            "of the Main_Building at Notre_Dame",
        ]
        context = AnnabellDeclarativeContext(
            declarative_statement, max_words_per_phrase=9
        )

        context_phrases = [phrase.text for phrase in context.phrases]

        self.assertEqual(context_phrases, expected_phrases)
        self.assertEqual(context.text, declarative_statement)


class TestAnnabellAnswerContext(AbstractAnnabellTestCase):
    def test_create_answer_context(self):
        answer = "a golden statue of the Virgin_Mary"
        expected_phrases = ["a golden statue of the Virgin_Mary"]
        context = AnnabellAnswerContext(answer)
        context_phrases = [phrase.text for phrase in context.phrases]
        self.assertEqual(context_phrases, expected_phrases)
        self.assertEqual(context.text, answer)


class TestLIFOQueue(unittest.TestCase):
    def test_lifo_queue_operations(self):
        queue = LIFOQueue()
        queue.enqueue("first")
        queue.enqueue("second")
        queue.enqueue("third")

        self.assertEqual(queue.dequeue(), "third")
        self.assertEqual(queue.dequeue(), "second")
        self.assertEqual(queue.dequeue(), "first")
        self.assertTrue(queue.is_empty())


class TestAnnabellQuestionCommandGenerator(AbstractAnnabellTestCase):

    def setUp(self):
        super().setUp()
        self.declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        self.question = "? what sit on top of the Main_Building at Notre_Dame"
        self.answer = "a golden statue of the Virgin_Mary"
        self.generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, self.answer, max_words=9
        )

    def tearDown(self):
        # Stop the patchers after each test
        self.log_stats_patcher.stop()
        self.exclude_patcher.stop()
        self.write_non_lookup_commands_patcher.stop()
        self.goal_stack_limit_patcher.stop()
        super().tearDown()

    def test_goal_stack_is_correct(self):
        self.assertTrue(self.generator.goal_stack.is_empty())
        self.generator.write_commands()
        expected_goal_stack_items = [["sit on top"], ["Main_Building"]]
        self.assertEqual(expected_goal_stack_items, self.generator.goal_stack.items())

    def test_word_group_is_set_correctly(self):
        self.assertTrue(len(self.generator.current_word_group) == 0)
        self.generator.write_commands()
        expected_goal_item = ["Main_Building"]
        expected_word_group = ["Notre_Dame"]
        self.assertEqual(expected_word_group, self.generator.current_word_group)
        self.assertEqual(expected_goal_item, self.generator.goal_stack.peek())

    def test_write_long_question_long_declaration_commands(self):
        declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        answer = "a golden statue of the Virgin_Mary"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        generator.write_commands()

        expected_commands = [
            "? what sit on top of the Main_Building at",
            "Notre_Dame",
            "#END OF QUESTION",
            ".sctx ? what sit on top of the Main_Building at",
            ".pg sit on top",
            ".pg Main_Building",
            ".sctx Notre_Dame",
            ".wg Notre_Dame",
        ]
        self.assertEqual(expected_commands, generator.commands)

    @patch(
        "commands.global_config.exclude_samples_with_fewer_than_2_lookups",
        return_value=True,
    )
    def test_write_short_question_short_declaration_commands(self, mock_method):
        declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        answer = "a golden statue of the Virgin_Mary"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=14
        )
        generator.write_question_commands()
        expected_commands = [
            ".pg sit on top",
            ".wg Main_Building at Notre_Dame",
        ]
        self.assertEqual(expected_commands, generator.commands)

        declarative_sentence = "a golden statue of the Virgin_Mary sit on top"
        question = "? what sit on top of the Main_Building at"
        answer = "a golden statue of the Virgin_Mary"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )

        with self.assertRaises(ToFewLookupWordGroupsException):
            generator.write_question_commands()

    def test_write_commands_single_phrase_question_multi_phrase_statement(self):
        declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        answer = "a golden statue of the Virgin_Mary"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=10
        )
        generator.write_commands()
        expected_commands = [
            "? what sit on top of the Main_Building at Notre_Dame",
            "#END OF QUESTION",
            ".pg sit on top",
            ".wg Main_Building at Notre_Dame",
        ]
        self.assertEqual(expected_commands, generator.commands)

    from config.global_config import global_config

    def test_write_commands_single_phrase_question_single_phrase_statement(self):

        # from config.global_config import override_global_config
        # with override_global_config(experiments__exclude_samples_with_fewer_than_2_lookups=False):

        declarative_sentence = "Notre_Dames_Juggler be publish twice"
        question = "? how often be Notre_Dames the Juggler publish"
        answer = "twice"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        generator.write_question_commands()
        expected_commands = [
            ".pg publish",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_multi_phrase_question_single_phrase_statement(self):
        declarative_sentence = (
            "Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        )
        question = "? what sit on top of the Main_Building at Notre_Dame"
        answer = "Virgin_Mary"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        generator.write_question_commands()
        expected_commands = [
            ".sctx ? what sit on top of the Main_Building at",
            ".pg sit on top",
            ".pg Main_Building",
            ".sctx Notre_Dame",
            ".wg Notre_Dame",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_multi_phrase_question_single_phrase_statement2(
        self,
    ):

        declarative_sentence = (
            "Beyonce have sell over 118_million record throughout the world"
        )
        question = "? how many record have Beyonce sell throughout the world"
        answer = "over 118_million"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? how many record have Beyonce sell throughout the",
            "world",
            "#END OF QUESTION",
            ".sctx ? how many record have Beyonce sell throughout the",
            ".pg record",
            ".pg Beyonce",
            ".pg sell",
            ".pg throughout",
            ".sctx world",
            ".wg world",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement(self):
        declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        answer = "a golden statue of the Virgin_Mary"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_question_commands()
        expected_commands = [
            ".sctx ? what sit on top of the Main_Building at",
            ".pg sit on top",
            ".pg Main_Building",
            ".sctx Notre_Dame",
            ".wg Notre_Dame",
        ]
        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement2(self):

        declarative_sentence = "a golden statue of the virgin mary sit on top of the main building at notre_dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        answer = "a golden statue of the Virgin Mary"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? what sit on top of the Main_Building at",
            "Notre_Dame",
            "#END OF QUESTION",
            ".sctx ? what sit on top of the Main_Building at",
            ".pg top",
            ".pg sit",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement3(
        self,
    ):

        declarative_sentence = "before the creation of the College_of_Engineering similar study be carry out at the College_of_Science"
        question = "? before the creation of the College_of_Engineering similar study be carry out at which Notre_Dame college"
        answer = "the College_of_Science"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? before the creation of the College_of_Engineering similar study",
            "be carry out at which Notre_Dame college",
            "#END OF QUESTION",
            ".sctx be carry out at which Notre_Dame college",
            ".pg carry",
            ".sctx ? before the creation of the College_of_Engineering similar study",
            ".pg creation of the College_of_Engineering",
            ".wg similar study",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement4(
        self,
    ):

        declarative_sentence = "the change to national standard at Notre_Dame in the early_20th_century take place over three_year"
        question = "? over how many year do the change to national standard undertake at Notre_Dame in the early_20th_century take place"
        answer = "three_year"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? over how many year do the change to",
            "national standard undertake at Notre_Dame in the early_20th_century take",
            "place",
            "#END OF QUESTION",
            ".sctx national standard undertake at Notre_Dame in the early_20th_century "
            "take",
            ".pg early_20th_century take",
            ".sctx place",
            ".pg place",
            ".sctx ? over how many year do the change to",
            ".pg change",
            ".sctx national standard undertake at Notre_Dame in the early_20th_century "
            "take",
            ".pg national standard",
            ".wg Notre_Dame",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement5(
        self,
    ):

        # todo make a change to preferentially choose the goal lookup word group to be that with the longest numbr of letters - in this case, Notre_Dame and daily_student paper instead, instead of call and Notre_Dame

        declarative_sentence = (
            "the daily student paper at Notre_Dame be call the Observer"
        )
        question = "? what be the daily student paper at Notre_Dame call"
        answer = "the Observer"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? what be the daily student paper at Notre_Dame",
            "call",
            "#END OF QUESTION",
            ".sctx ? what be the daily student paper at Notre_Dame",
            ".pg daily student paper",
            ".pg Notre_Dame",
            ".sctx call",
            ".wg call",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement6(
        self,
    ):

        declarative_sentence = "the number of department within the Stinson - Remick Hall of Engineering be five"
        question = (
            "? how many department be within the Stinson - Remick Hall of Engineering"
        )
        answer = "five"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? how many department be within the Stinson -",
            "Remick Hall of Engineering",
            "#END OF QUESTION",
            ".sctx Remick Hall of Engineering",
            ".pg Hall of Engineering",
            ".sctx ? how many department be within the Stinson -",
            ".pg department",
            ".pg within the Stinson",
            ".sctx Remick Hall of Engineering",
            ".wg Remick",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement7(self):

        declarative_sentence = "the entity that provide help with the management of time for new student at Notre_Dame be Learning_Resource_Center"
        question = "? what entity provide help with the management of time for new student at Notre_Dame"
        answer = "Learning_Resource_Center"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? what entity provide help with the management of",
            "time for new student at Notre_Dame",
            "#END OF QUESTION",
            ".sctx time for new student at Notre_Dame",
            ".pg time for new student",
            ".pg Notre_Dame",
            ".sctx ? what entity provide help with the management of",
            ".pg entity",
            ".pg provide help",
            ".wg management",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    @patch(
        "commands.global_config.goal_stack_limit",
        return_value=1,
    )
    def test_write_commands_multi_phrase_question_multi_phrase_statement7_with_goal_stack_limit(
        self, mock_method1
    ):

        declarative_sentence = "the entity that provide help with the management of time for new student at Notre_Dame be Learning_Resource_Center"
        question = "? what entity provide help with the management of time for new student at Notre_Dame"
        answer = "Learning_Resource_Center"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? what entity provide help with the management of",
            "time for new student at Notre_Dame",
            "#END OF QUESTION",
            ".sctx ? what entity provide help with the management of",
            ".pg provide help",
            ".wg management",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement8(
        self,
    ):

        declarative_sentence = "Forbescom place Notre_Dame at 8th position compare to other US research university"
        question = "? Forbescom place Notre_Dame at what position compare to other US research university"
        answer = "8th"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? Forbescom place Notre_Dame at what position compare to",
            "other US research university",
            "#END OF QUESTION",
            ".sctx other US research university",
            ".pg US research university",
            ".sctx ? Forbescom place Notre_Dame at what position compare to",
            ".pg Forbescom place Notre_Dame",
            ".wg position compare",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement9(
        self,
    ):

        declarative_sentence = "the percentage of food serve at Notre_Dame that be locally grow be 40_percent"
        question = "? what percentage of the food serve at Notre_Dame be locally grow"
        answer = "40_percent"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )

        question_generator.write_commands()

        expected_commands = [
            "? what percentage of the food serve at Notre_Dame",
            "be locally grow",
            "#END OF QUESTION",
            ".sctx be locally grow",
            ".pg locally grow",
            ".sctx ? what percentage of the food serve at Notre_Dame",
            ".pg percentage of",
            ".wg food serve at Notre_Dame",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement10(
        self,
    ):

        declarative_sentence = (
            "the encounter between Notre_Dame student and the KKK occur in South_Bend"
        )
        question = "? where do Notre_Dame student and the KKK have their encounter"
        answer = "south Bend"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? where do Notre_Dame student and the KKK have",
            "their encounter",
            "#END OF QUESTION",
            ".sctx ? where do Notre_Dame student and the KKK have",
            ".pg Notre_Dame student",
            ".pg KKK",
            ".sctx their encounter",
            ".wg encounter",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement11(
        self,
    ):

        declarative_sentence = (
            "in front of the Notre_Dame_Main_Building stand a copper statue of Christ"
        )
        question = "? what be in front of the Notre_Dame_Main_Building"
        answer = "a copper statue of Christ"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? what be in front of the Notre_Dame_Main_Building",
            "#END OF QUESTION",
            ".pg front of the Notre_Dame_Main_Building",
        ]

        self.assertEqual(expected_commands, question_generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement12(self):

        declarative_sentence = "Father John_Francis OHara become the Vice - President of the University_of_Notre_Dame in 1933"
        question = "? which person become vice - president of Notre_Dame in 1933"
        answer = "Father John_Francis OHara"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? which person become vice - president of Notre_Dame",
            "in 1933",
            "#END OF QUESTION",
            ".sctx in 1933",
            ".pg 1933",
            ".sctx ? which person become vice - president of Notre_Dame",
            ".pg become",
        ]

        self.assertEqual(expected_commands, question_generator.commands)


class TestAnnabellAnswerCommandGenerator(AbstractAnnabellTestCase):

    def setUp(self):
        super().setUp()
        self.declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        self.answer = "a golden statue of the Virgin_Mary"
        self.question = "? what sit on top of the Main_Building at Notre_Dame"
        self.question_generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, self.answer, max_words=9
        )
        self.question_generator.write_commands()
        self.answer_context = AnnabellAnswerContext(self.answer)

    def test_create_answer_generator(self):
        answer_generator = AnnabellAnswerCommandGenerator(
            self.declarative_sentence,
            self.answer,
            self.question_generator,
            max_words=9,
        )
        self.assertEqual(
            answer_generator.declarative_sentence.text, self.declarative_sentence
        )
        self.assertEqual(answer_generator.answer.text, self.answer)
        self.assertEqual(answer_generator.question_generator, self.question_generator)

    def test_split_long_answer_into_phrases(self):
        answer = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        context = AnnabellAnswerContext(answer)
        expected_phrases = [
            "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame",
        ]
        context_phrases = [phrase.text for phrase in context.phrases]
        self.assertEqual(context_phrases, expected_phrases)
        self.assertEqual(context.text, answer)

    def test_write_commands_short_answer_single_phrase_statement(self):
        question = "? how many record have Beyonce sell throughout the world"
        declarative_sentence = (
            "Beyonce have sell over 118_million record throughout the world"
        )
        answer = "over 118_million"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph Beyonce have sell over 118_million record throughout the world",
            ".drop_goal",
            ".drop_goal",
            ".drop_goal",
            ".drop_goal",
            ".wg over 118_million",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_long_answer_single_phrase_statement(self):
        question = "? what sit on top of the Main_Building"
        declarative_sentence = "a golden statue of the Virgin_Mary sit on top"
        answer = "a golden statue of the Virgin_Mary sit on top of the Main_Building"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_commands_long_answer_single_phrase_statement()
        expected_commands = [
            ".ph a golden statue of the Virgin_Mary sit on top",
            ".wg a golden statue of",
            ".prw",
            ".wg the Virgin_Mary sit on",
            ".prw",
            ".wg top",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_short_answer_multi_phrase_statement(self):
        answer = "the Virgin_Mary"
        generator = AnnabellAnswerCommandGenerator(
            self.declarative_sentence, answer, self.question_generator, max_words=9
        )
        generator.write_answer_commands()
        # generator.write_commands_short_answer_multi_phrase_statement()
        expected_commands = [
            ".ggp",
            ".ph of the Main_Building at Notre_Dame",
            ".drop_goal",
            ".sctx a golden statue of the Virgin_Mary sit on top",
            ".drop_goal",
            ".wg the Virgin_Mary",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_short_answer_multi_phrase_statement2(self):
        declarative_sentence = "before the creation of the College_of_Engineering similar study be carry out at the College_of_Science"
        question = "? before the creation of the College_of_Engineering similar study be carry out at which Notre_Dame college"
        answer = "the College_of_Science"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph before the creation of the College_of_Engineering similar study be",
            ".drop_goal",
            ".sctx carry out at the College_of_Science",
            ".drop_goal",
            ".wg the College_of_Science",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_short_answer_multi_phrase_statement3(self):
        declarative_sentence = "the change to national standard at Notre_Dame in the early_20th_century take place over three_year"
        question = "? over how many year do the change to national standard undertake at Notre_Dame in the early_20th_century take place"
        answer = "three_year"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph the change to national standard at Notre_Dame in the",
            ".drop_goal",
            ".drop_goal",
            ".sctx early_20th_century take place over three_year",
            ".drop_goal",
            ".drop_goal",
            ".wg three_year",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_short_answer_multi_phrase_statement4(self):
        # todo: handles this case where an answer word group is split across multiple phrases ("the", "Observer")
        declarative_sentence = (
            "the daily student paper at Notre_Dame be call the Observer"
        )
        question = "? what be the daily student paper at Notre_Dame call"
        answer = "the Observer"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        with self.assertRaises(MissingAnswerWordsException):
            generator.write_answer_commands()
        expected_commands = [
            "? what be the daily student paper at Notre_Dame call",
            ".drop_goal",
            ".drop_goal",
            ".sctx the Observer",
            ".drop_goal",
            ".drop_goal",
            ".wg the Observer",
            ".rw",
        ]
        # self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_short_answer_multi_phrase_statement5(self):

        declarative_sentence = "the number of department within the Stinson - Remick Hall of Engineering be five"
        question = (
            "? how many department be within the Stinson - Remick Hall of Engineering"
        )
        answer = "five"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()

        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph the number of department within the Stinson - Remick",
            ".drop_goal",
            ".drop_goal",
            ".sctx Hall of Engineering be five",
            ".drop_goal",
            ".wg five",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_short_answer_multi_phrase_statement6(self):

        declarative_sentence = "the entity that provide help with the management of time for new student at Notre_Dame be Learning_Resource_Center"
        question = "? what entity provide help with the management of time for new student at Notre_Dame"
        answer = "Learning_Resource_Center"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()

        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph the entity that provide help with the management of",
            ".drop_goal",
            ".drop_goal",
            ".sctx time for new student at Notre_Dame be Learning_Resource_Center",
            ".drop_goal",
            ".drop_goal",
            ".wg Learning_Resource_Center",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_short_answer_multi_phrase_statement7(self):

        declarative_sentence = "Forbescom place Notre_Dame at 8th position compare to other US research university"
        question = "? Forbescom place Notre_Dame at what position compare to other US research university"
        answer = "8th"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()

        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph Forbescom place Notre_Dame at 8th position compare to other",
            ".drop_goal",
            ".drop_goal",
            ".wg 8th",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_short_answer_multi_phrase_statement8(self):

        declarative_sentence = "Father John_Francis OHara become the Vice - President of the University_of_Notre_Dame in 1933"
        question = "? which person become vice - president of Notre_Dame in 1933"
        answer = "Father John_Francis OHara"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()

        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph Father John_Francis OHara become the Vice - President of",
            ".drop_goal",
            ".drop_goal",
            ".wg Father John_Francis OHara",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_long_answer_multi_phrase_statement(self):

        question_generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, self.answer, max_words=9
        )
        question_generator.write_commands()

        generator = AnnabellAnswerCommandGenerator(
            self.declarative_sentence, self.answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph of the Main_Building at Notre_Dame",
            ".drop_goal",
            ".sctx a golden statue of the Virgin_Mary sit on top",
            ".drop_goal",
            ".wg a golden statue of",
            ".prw",
            ".wg the Virgin_Mary",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_long_answer_multi_phrase_statement2(self):

        declarative_sentence = (
            "in front of the Notre_Dame_Main_Building stand a copper statue of Christ"
        )
        question = "? what be in front of the Notre_Dame_Main_Building"
        answer = "a copper statue of Christ"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()

        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
            ".ggp",
            ".ph in front of the Notre_Dame_Main_Building stand a copper statue",
            ".drop_goal",
            ".wg a copper statue",
            ".prw",
            ".sctx of Christ",
            ".wg of Christ",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    @patch(
        "commands.global_config.exclude_samples_with_fewer_than_2_lookups",
        return_value=False,
    )
    def test_write_commands_long_answer_multi_phrase_statement_no_matching_word_group(
        self, mock_method
    ):

        declarative_sentence = (
            "the daily student paper at Notre_Dame be call the Observer"
        )
        question = "? what be the daily student paper at Notre_Dame call"
        answer = "the Observer"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        answer_generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        with self.assertRaises(MissingAnswerWordsException):
            answer_generator.write_answer_commands()

    def test_write_commands_long_answer_multi_phrase_statement_answer_stopword_at_end(
        self,
    ):
        # todo: handles this case where the answer is split across multiple phrases and the last answer word of the first phrase is a stopword ("the", "Observer")
        declarative_sentence = "a golden statue of the virgin mary sit on top of the main building at notre_dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        answer = "a golden statue of the Virgin Mary"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()
        answer_generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        with self.assertRaises(MissingAnswerWordsException):
            answer_generator.write_answer_commands()

    def test_write_commands_long_answer_multi_phrase_statement3(self):
        """Put a test case for this sample
        2026-02-26 23:15:36,540 - dataset_processing - INFO - Answer 'subject–verb–object' not found in declarative statement: 'the sentence structure of czech syntax follows the subject-verb-object order.', for sample ID: 572b9c6fbe1ee31400cb8465
        """

        """2026-02-26 23:15:54,577 - commands - WARNING - Not all answer words were found in the declarative sentence. missing answer words: ['up']
2026-02-26 23:15:54,577 - commands - ERROR - Error creating commands for sample 57313d51e6313a140071cd75: Not all answer words were found in the declarative sentence. missing answer words: ['up'] Declarative sentence: 'government assistance be necessary in Tuvalu for clean - up after the cyclone' Question: '? for what action be government assistance necessary after the Tuvalu cyclone' Answer: 'clean - up"""

        declarative_sentence = "Sicily play a key role in transfer the lute from Arabian to european culture"
        question = (
            "? what country helped transfer the lute from Arabian to european culture"
        )
        answer = "sicily"

        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, answer, max_words=9
        )
        question_generator.write_commands()

        answer_generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        with self.assertRaises(MissingAnswerWordsException):
            answer_generator.write_answer_commands()


class TestAnnabellBaseCommandGenerator(AbstractAnnabellTestCase):
    def setUp(self):
        super().setUp()
        """Set up a common instance for testing."""
        self.sample_id = "test_01"
        self.declarative_sentence = "the sky is blue with patches of grey"
        self.question = "? what color is the sky"

        self.short_answer = "blue"
        self.long_answer = "blue with patches of grey"

        self.long_question = (
            "? what was the trade -ing post that precede -d New-York-City call -ed"
        )
        self.long_declarative_sentence = "the trade -ing post that precede -d New-York-City was call -ed New-Amsterdam"

    def test_create_base_command_generator(self):
        generator = AnnabellBaseCommandGenerator(
            self.sample_id,
            self.declarative_sentence,
            self.question,
            self.short_answer,
            is_pre_training=True,
            max_words=10,
        )
        self.assertEqual(generator.sample_id, self.sample_id)
        self.assertEqual(generator.declarative_sentence.text, self.declarative_sentence)
        self.assertEqual(generator.question, self.question)
        self.assertEqual(generator.answer, self.short_answer)
        self.assertTrue(generator.is_pre_training)
        self.assertEqual(generator.max_words, 10)

    @patch(
        "commands.global_config.exclude_samples_with_fewer_than_2_lookups",
        return_value=True,
    )
    def test_write_question_commands(self, mock_method):
        generator = AnnabellBaseCommandGenerator(
            self.sample_id,
            self.declarative_sentence,
            self.question,
            self.short_answer,
            max_words=10,
        )
        with self.assertRaises(ToFewLookupWordGroupsException):
            generator.write_question_commands()
        # expected_commands = ["? what color is the sky", "#END OF QUESTION", ".pg sky"]
        # self.assertEqual(expected_commands, generator.commands)

    def test_write_answer_commands(self):
        generator = AnnabellBaseCommandGenerator(
            self.sample_id,
            self.declarative_sentence,
            self.question,
            self.short_answer,
            max_words=10,
        )
        generator.write_question_commands()
        generator.write_answer_commands()

        expected_commands = [
            "? what color is the sky",
            "#END OF QUESTION",
            ".pg sky",
            ".ggp",
            ".ph the sky is blue with patches of grey",
            ".drop_goal",
            ".wg blue",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_long_answer(self):
        """Test write_answer_commands with a long sentence where the answer is split across phrases."""
        declarative_sentence = "the sky is a brilliant blue with some patches of grey"
        answer = "blue with some patches of grey"
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, declarative_sentence, self.question, answer, max_words=5
        )
        generator.write_question_commands()
        generator.write_answer_commands()
        expected_commands = [
            "? what color is the",
            "sky",
            "#END OF QUESTION",
            ".sctx sky",
            ".pg sky",
            ".ggp",
            ".ph the sky is a brilliant",
            ".drop_goal",
            ".sctx blue with some patches of",
            ".wg blue with some patches",
            ".prw",
            ".wg of",
            ".prw",
            ".sctx grey",
            ".wg grey",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_timing_in_commands(self):
        """Test that timing commands are correctly added."""
        generator = AnnabellBaseCommandGenerator(
            self.sample_id,
            self.declarative_sentence,
            self.question,
            self.short_answer,
            max_words=9,
        )
        generator.create_list_of_commands()

        expected_commands = [
            "#id: test_01",
            "the sky is blue with patches of grey",
            f"{AnnabellLogfileInterpreter.end_of_declaration_string()}",
            "\n",
            "? what color is the sky",
            f"{AnnabellLogfileInterpreter.end_of_question_string()}",
            ".pg sky",
            ".ggp",
            ".ph the sky is blue with patches of grey",
            ".drop_goal",
            ".wg blue",
            ".rw",
            "\n",
            f"{AnnabellLogfileInterpreter.end_of_commands_string()}",
            f"{AnnabellBaseCommandGenerator.time_command()}",
            f"{AnnabellLogfileInterpreter.end_of_time_string()}",
        ]

        self.assertEqual(expected_commands, generator.commands)

    @patch(
        "commands.global_config.log_stats",
        return_value=True,
    )
    def test_stat_in_commands(self, mock_method1):
        generator = AnnabellBaseCommandGenerator(
            self.sample_id,
            self.declarative_sentence,
            self.question,
            self.short_answer,
            max_words=9,
        )
        generator.create_list_of_commands()

        expected_commands = [
            "#id: test_01",
            "the sky is blue with patches of grey",
            f"{AnnabellLogfileInterpreter.end_of_declaration_string()}",
            "\n",
            "? what color is the sky",
            f"{AnnabellLogfileInterpreter.end_of_question_string()}",
            ".pg sky",
            ".ggp",
            ".ph the sky is blue with patches of grey",
            ".drop_goal",
            ".wg blue",
            ".rw",
            "\n",
            f"{AnnabellLogfileInterpreter.end_of_commands_string()}",
            f"{AnnabellBaseCommandGenerator.time_command()}",
            f"{AnnabellLogfileInterpreter.end_of_time_string()}",
            f"{AnnabellBaseCommandGenerator.stat_command()}",
            f"{AnnabellLogfileInterpreter.end_of_stats_string()}",
        ]

        self.assertEqual(expected_commands, generator.commands)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
