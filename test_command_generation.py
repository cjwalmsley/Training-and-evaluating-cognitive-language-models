import unittest

from numpy.ma.testutils import assert_equal

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
)


class TestAbstractAnnabellCommandGenerator(unittest.TestCase):

    def setUp(self):
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
            "\n",
            "? what color is the sky",
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
            "? what color is the sky",
            ".wg sky",
            ".ph the sky is blue with patches of grey",
            ".wg blue with patches of",
            ".prw",
            ".wg grey",
            ".rw",
            "\n",
        ]

        self.assertEqual(expected_commands, commands)

    def test_write_question_commands_for_phrase(self):
        """Test the write_question_commands_for_phrase method."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, max_words=5
        )
        generator.write_commands()
        expected_commands = [
            "? what color is the",
            "sky",
            ".sctx sky",
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
            ".pg trade",
            ".sctx -ing post that precede -d",
            ".pg post",
            ".pg precede",
            ".sctx New-York-City call -ed",
            ".pg New-York-City",
            ".wg call",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_question_commands_wg_not_in_phrase(self):

        question = "? what sit on top of the_Main_Building at Notre_Dame"
        declarative_sentence = "a golden statue of the_Virgin_Mary sit on top of the_Main_Building at Notre_Dame"

        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence,
            question,
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
            self.declarative_sentence, self.question, max_words=5
        )
        generator.write_question_commands()
        expected_commands = [".sctx sky", ".wg sky"]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_question_commands_long_declarative_sentence(self):
        pass

    def test_write_question_commands_long_question(self):
        """Test the write_question_commands method with a long question."""
        generator = AnnabellQuestionCommandGenerator(
            self.long_declarative_sentence,
            self.long_question,
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
            ".pg precede",
            ".sctx New-York-City call -ed",
            ".pg New-York-City",
            ".wg call",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_question(self):
        """Test the write_question method with a short question."""
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, max_words=5
        )

        expected_commands = ["? what color is the", "sky"]

        generator.write_question()
        self.assertEqual(generator.commands, expected_commands)

        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, max_words=6
        )

        expected_commands = ["? what color is the sky"]

        generator.write_question()
        self.assertEqual(generator.commands, expected_commands)

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
            declarative_sentence, question, max_words=10
        )
        question_generator.write_commands()
        answer_generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=10
        )
        answer_generator.write_answer_commands()
        expected_commands = [
            ".ph the Grotto at Notre_Dame be a marian place of prayer",
            ".wg a marian place of",
            ".prw",
            ".wg prayer",
            ".prw",
            ".sctx and reflection",
            ".wg and reflection",
            ".rw",
        ]
        self.assertEqual(expected_commands, answer_generator.commands)


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


class TestAnnabellQuestionContext(unittest.TestCase):

    def setUp(self):
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


class TestAnnabellDeclarativeContext(unittest.TestCase):
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


class TestAnnabellAnswerContext(unittest.TestCase):
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


class TestAnnabellQuestionCommandGenerator(unittest.TestCase):

    def setUp(self):
        self.declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        self.question = "? what sit on top of the Main_Building at Notre_Dame"
        self.generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, max_words=9
        )

    def test_goal_stack_is_correct(self):
        self.assertTrue(self.generator.goal_stack.is_empty())
        self.generator.write_commands()
        expected_goal_stack_items = [["sit on top"], ["Main_Building"]]
        self.assertEqual(expected_goal_stack_items, self.generator.goal_stack.items())

    def test_word_group_is_set_correctly(self):
        self.assertTrue(len(self.generator.current_word_group) == 0)
        self.generator.write_commands()
        expected_word_group = ["Notre_Dame"]
        self.assertEqual(expected_word_group, self.generator.current_word_group)

    def test_write_long_question_long_declaration_commands(self):
        declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=9
        )
        generator.write_commands()

        expected_commands = [
            "? what sit on top of the Main_Building at",
            "Notre_Dame",
            ".sctx ? what sit on top of the Main_Building at",
            ".pg sit on top",
            ".pg Main_Building",
            ".sctx Notre_Dame",
            ".wg Notre_Dame",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_short_question_short_declaration_commands(self):
        declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=14
        )
        generator.write_question_commands()
        expected_commands = [
            ".pg sit on top",
            ".wg Main_Building at Notre_Dame",
        ]
        self.assertEqual(expected_commands, generator.commands)

        declarative_sentence = "a golden statue of the Virgin_Mary sit on top"
        question = "? what sit on top of the Main_Building at"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=9
        )
        generator.write_commands_single_phrase_question_single_phrase_statement()
        expected_commands = [
            ".wg sit on top",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_single_phrase_question_multi_phrase_statement(self):
        declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=10
        )
        generator.write_commands_single_phrase_question_multi_phrase_statement()
        expected_commands = [
            ".pg sit on top",
            ".wg Main_Building at Notre_Dame",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_single_phrase_question_single_phrase_statement(self):
        declarative_sentence = "Notre_Dames_Juggler be publish twice"
        question = "? how often be Notre_Dames the Juggler publish"
        answer = "twice"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=9
        )
        generator.write_question_commands()
        expected_commands = [
            ".wg publish",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_multi_phrase_question_single_phrase_statement(self):
        declarative_sentence = (
            "Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        )
        question = "? what sit on top of the Main_Building at Notre_Dame"
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=9
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
        # answer = "over 118_million"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? how many record have Beyonce sell throughout the",
            "world",
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
        generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=9
        )
        generator.write_commands_multi_phrase_question_multi_phrase_statement()
        expected_commands = [
            ".sctx ? what sit on top of the Main_Building at",
            ".pg sit on top",
            ".pg Main_Building",
            ".sctx Notre_Dame",
            ".wg Notre_Dame",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_multi_phrase_question_multi_phrase_statement2(
        self,
    ):

        declarative_sentence = "a golden statue of the virgin mary sit on top of the main building at notre_dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        # answer = "a golden statue of the Virgin Mary"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=9
        )
        question_generator.write_commands()
        expected_commands = [
            "? what sit on top of the Main_Building at",
            "Notre_Dame",
            ".sctx ? what sit on top of the Main_Building at",
            ".pg sit",
            ".wg top",
        ]

        self.assertEqual(expected_commands, question_generator.commands)


class TestAnnabellAnswerCommandGenerator(unittest.TestCase):

    def setUp(self):
        self.declarative_sentence = "a golden statue of the Virgin_Mary sit on top of the Main_Building at Notre_Dame"
        self.answer = "a golden statue of the Virgin_Mary"
        self.question = "? what sit on top of the Main_Building at Notre_Dame"
        self.question_generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence, self.question, max_words=9
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
            declarative_sentence, question, max_words=9
        )
        question_generator.write_commands()
        generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        generator.write_answer_commands()
        expected_commands = [
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
            declarative_sentence, question, max_words=9
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
        generator.write_commands_long_answer_multi_phrase_statement()
        # generator.write_commands_short_answer_multi_phrase_statement()
        expected_commands = [
            ".ph of the Main_Building at Notre_Dame",
            ".drop_goal",
            # todo test alternative commands - .ph instead of .sctx which results in the best generalisation?
            ".ph a golden statue of the Virgin_Mary sit on top",
            # ".sctx a golden statue of the Virgin_Mary sit on top",
            ".drop_goal",
            ".wg the Virgin_Mary",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_long_answer_multi_phrase_statement(self):

        generator = AnnabellAnswerCommandGenerator(
            self.declarative_sentence, self.answer, self.question_generator, max_words=9
        )
        generator.write_commands_long_answer_multi_phrase_statement()
        expected_commands = [
            ".ph of the Main_Building at Notre_Dame",
            ".drop_goal",
            # todo test alternative commands - .ph instead of .sctx which results in the best generalisation?
            ".ph a golden statue of the Virgin_Mary sit on top",
            # ".sctx a golden statue of the Virgin_Mary sit on top",
            ".drop_goal",
            ".wg a golden statue of",
            ".prw",
            ".wg the Virgin_Mary",
            ".rw",
        ]
        self.assertEqual(expected_commands, generator.commands)

    def test_write_commands_long_answer_multi_phrase_statement_no_matching_word_group(
        self,
    ):

        declarative_sentence = "a golden statue of the virgin mary sit on top of the main building at notre_dame"
        question = "? what sit on top of the Main_Building at Notre_Dame"
        answer = "a golden statue of the Virgin Mary"
        question_generator = AnnabellQuestionCommandGenerator(
            declarative_sentence, question, max_words=9
        )
        question_generator.write_commands()
        answer_generator = AnnabellAnswerCommandGenerator(
            declarative_sentence, answer, question_generator, max_words=9
        )
        with self.assertRaises(MissingAnswerWordsException):
            answer_generator.write_commands_long_answer_multi_phrase_statement()


class TestAnnabellBaseCommandGenerator(unittest.TestCase):
    def setUp(self):
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

    def test_write_question_commands(self):
        generator = AnnabellBaseCommandGenerator(
            self.sample_id,
            self.declarative_sentence,
            self.question,
            self.short_answer,
            max_words=10,
        )
        generator.write_question_commands()
        expected_commands = [
            "? what color is the sky",
            ".wg sky",
        ]
        self.assertEqual(expected_commands, generator.commands)

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
            ".wg sky",
            ".ph the sky is blue with patches of grey",
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
            ".sctx sky",
            ".wg sky",
            ".ph the sky is a brilliant",
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


# todo add test case for the following - 2026-01-08 10:28:43,739 - commands - ERROR - Error creating commands for sample 5733bf84d058e614000b61c0: Not all answer words were found in the declarative sentence. missing answer words: ['the', 'Observer'] Declarative sentence: 'the daily student paper at Notre_Dame be call the Observer' Question: '? what be the daily student paper at Notre_Dame call' Answer: 'the Observer

"""#id: 5733a6424776f41900660f4f
before the creation of the College_of_Engineering similar study be
carry out at the College_of_Science

 >>> End context

 >>> End context
? before the creation of the College_of_Engineering similar study
be carry out at which Notre_Dame college"""

"""#id: 5733a70c4776f41900660f64
? what entity provide help with the management of
time for new student at Notre_Dame
.x
 -> management
.
management
#END OF TESTING SAMPLE"""

"""#id: 5733a6424776f41900660f50
? how many department be within the Stinson -
Remick Hall of Engineering
.x

#END OF TESTING SAMPLE"""

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)