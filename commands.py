import logging
import nltk
from nltk.corpus import stopwords
from config.global_config import GlobalConfig

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class AbstractAnnabellCommandGenerator:

    def __init__(self, declarative_sentence_text, max_words):
        self.max_words = max_words
        self.declarative_sentence = AnnabellDeclarativeContext(
            declarative_sentence_text, self.max_words
        )
        self.declarative_sentence_type = (
            self.declarative_sentence.get_declarative_sentence_type()
        )

        self.commands = []

    def create_list_of_commands(self):
        raise NotImplementedError(
            "Subclasses must implement create_list_of_commands method"
        )

    def phrases_in_context(self, context):
        phrases = []
        context_words = context.split()
        number_of_phrases = (len(context_words) + self.max_words - 1) // self.max_words
        for i in range(number_of_phrases):
            phrase_words = context_words[i * self.max_words : (i + 1) * self.max_words]
            phrase = " ".join(phrase_words)
            phrases.append(phrase)
        return phrases

    def get_declarative_sentence_type(self):
        if len(self.declarative_sentence.text.split()) > self.max_words:
            return LongDeclarativeSentenceType()
        else:
            return ShortDeclarativeSentenceType()


class AnnabellBaseCommandGenerator(AbstractAnnabellCommandGenerator):
    # creates the set of commands required to train annabell for a single training sample
    def __init__(
        self,
        sample_id,
        declarative_sentence,
        question,
        answer,
        is_pre_training=True,
        max_words=global_config.maximum_phrase_length(),
    ):
        super().__init__(declarative_sentence, max_words)

        self.sample_id = sample_id
        self.question = question
        self.answer = answer
        self.is_pre_training = is_pre_training
        self.answer_command_generator = self.answer_command_generator()
        self.question_command_generator = self.question_command_generator()

    def question_command_generator(self):
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence.text, self.question, self.max_words
        )
        return generator

    def answer_command_generator(self):
        generator = AnnabellAnswerCommandGenerator(
            self.declarative_sentence.text, self.answer, self.max_words
        )
        return generator

    @staticmethod
    def blank_line():
        return "\n"

    def write_declarative_sentence(self):

        for phrase in self.declarative_sentence.phrases:
            self.commands.append(phrase.text)

    @staticmethod
    def informational_non_pretraining_command():
        return "# This is a non-pretraining sample. No commands to execute."

    def write_question_commands(self):
        self.question_command_generator.write_commands()
        self.commands.extend(self.question_command_generator.commands)

    def write_answer_commands(self):
        self.answer_command_generator.write_answer_commands()
        self.commands.extend(self.answer_command_generator.commands)

    def write_sample_id(self):
        self.commands.append("#id: " + str(self.sample_id))

    def create_list_of_commands(self):

        self.commands = []

        if self.is_pre_training:
            self.write_sample_id()
            self.write_declarative_sentence()
            # add a blank line to terminate the context
            self.commands.append(self.blank_line())
            self.write_question_commands()
            self.write_answer_commands()
            # add a blank line to terminate the context
            self.commands.append(self.blank_line())
        else:
            self.commands.append(self.informational_non_pretraining_command())
        return self.commands


class AnnabellTestingCommandGenerator(AnnabellBaseCommandGenerator):
    # creates the commands required to test annabell for a single testing sample
    def __init__(
        self,
        sample_id,
        question,
    ):
        super().__init__(
            sample_id, declarative_sentence="", question=question, answer=None
        )

    def answer_command_generator(self):
        return None

    def create_list_of_commands(self):

        self.commands = []
        self.write_sample_id()
        self.write_question_commands()
        # add a blank line to terminate the context
        self.commands.append(".x")
        self.commands.append("#END OF TESTING SAMPLE")
        self.commands.append(self.blank_line())

        return self.commands

    def write_question_commands(self):
        self.question_command_generator.write_question()
        self.commands.extend(self.question_command_generator.commands)


class AnnabellTrainingCommandGenerator(AnnabellBaseCommandGenerator):
    # creates the commands required to test annabell for a single testing sample
    def __init__(
        self,
        sample_id,
        declarative_sentence,
    ):
        super().__init__(sample_id, declarative_sentence, question=None, answer=None)

        """"#id: 5733be284776f41900661180",
        "the Basilica of the Sacred Heart at Notre_Dame be",
        " beside the_Main_Building",
        "\n","""

    def answer_command_generator(self):
        return None

    def question_command_generator(self):
        return None

    def create_list_of_commands(self):

        self.commands = []
        self.write_sample_id()
        self.write_training_commands()
        # add a blank line to terminate the context
        self.commands.append(self.blank_line())

        return self.commands

    def write_training_commands(self):
        self.write_declarative_sentence()


class AnnabellAnswerCommandGenerator(AbstractAnnabellCommandGenerator):
    def __init__(
        self,
        declarative_sentence,
        answer,
        max_words,
    ):
        super().__init__(declarative_sentence, max_words)

        self.answer = AnnabellAnswerContext(answer, max_words)
        self.answer_type = self.answer.get_answer_type()

    @staticmethod
    def answer_command_for_phrase_index(index):
        if index == 0:
            command = ".ph"
        else:
            command = ".sctx"

        return command

    def write_answer_commands(self):
        self.answer_type.write_answer_commands(self)

    def write_short_declaration_long_answer_commands(self):
        raise NotImplementedError(
            "Short declarative sentence with long answer not supported"
        )

    def write_long_declaration_long_answer_commands(self):
        answer_words_remaining = self.answer.words().copy()
        # look up the first phrase of the context
        self.commands.append(f".ph {self.declarative_sentence.first_phrase().text}")
        # for each remaining phrase in the declarative sentence search the context for the phrase and write the answer words that are in that phrase.
        for declarative_phrase in self.declarative_sentence.phrases:
            self.commands.append(f".sctx {declarative_phrase.text}")
            for declarative_word in declarative_phrase.words():
                if declarative_word in answer_words_remaining:
                    self.commands.append(f".wg {declarative_word}")
                    answer_words_remaining.remove(declarative_word)
                    if len(answer_words_remaining) > 0:
                        self.commands.append(".prw")
                    else:
                        continue
        self.commands.append(".rw")

    def write_short_declaration_short_answer_commands(self):
        self.commands.append(f".ph {self.declarative_sentence.text}")
        # the model can only hold 4 words in its focus of attention, so the answer must be split and rewarded and outputted incrementally in chunks if the answer has more than 4 words

        if len(self.answer.words()) < 4:
            self.commands.append(f".wg {self.answer.text}")
        else:
            self.commands.append(f".wg {" ".join(self.answer.words()[:3])}")
            self.commands.append(".prw")
            self.commands.append(f".wg {" ".join(self.answer.words()[3:])}")
        self.commands.append(".rw")

    def write_long_declaration_short_answer_commands(self):
        # for each phrase in the declarative sentence, write the phrase command and the answer words
        for phrase_index, (phrase, phrase_answer_words) in enumerate(
            self.declarative_sentence.phrases_and_answer_words(
                self.answer.words()
            ).items()
        ):
            if len(phrase_answer_words) == 0 and phrase_index > 0:
                continue
            else:
                self.commands.append(
                    f"{self.answer_command_for_phrase_index(phrase_index)} {phrase.text}"
                )
                if len(phrase_answer_words) == 0:
                    continue
                elif len(phrase_answer_words) < 4:
                    self.commands.append(f".wg {" ".join(phrase_answer_words)}")
                else:
                    self.commands.append(f".wg {" ".join(phrase_answer_words[:3])}")
                    self.commands.append(".prw")
                    self.commands.append(f".wg {" ".join(phrase_answer_words[3:])}")
                    self.commands.append(".prw")
        self.commands.append(".rw")


class AnnabellQuestionCommandGenerator(AbstractAnnabellCommandGenerator):
    def __init__(
        self,
        declarative_sentence,
        question,
        max_words,
    ):
        super().__init__(declarative_sentence, max_words)
        self.question = AnnabellQuestionContext(question, max_words)
        self.question_type = self.question.get_question_type()

    def write_commands(self):
        self.commands = []
        self.write_question()
        self.write_question_commands()
        return self.commands

    def write_question(self):
        self.question_type.write_question(self)

    def write_question_commands(self):
        self.question_type.write_question_commands(self)

    def append_question(self):
        self.commands.append(self.question.text)

    def append_question_phrases(self):
        # split the question into phrases of max_words length
        for phrase in self.question.phrases_in_context():
            self.commands.append(phrase.text)

    def write_commands_single_phrase_question_single_phrase_statement(self):

        word_group_chunks = (
            self.question.word_group_chunks_matching_declarative_sentence(
                self.declarative_sentence
            )
        )

        if len(word_group_chunks) == 1:
            self.commands.append(f".wg {' '.join(word_group_chunks[0])}")
        elif len(word_group_chunks) > 1:
            self.commands.append(f".pg {' '.join(word_group_chunks[0])}")
            self.commands.append(f".wg {' '.join(word_group_chunks[-1])}")

    def write_commands_single_phrase_question_multi_phrase_statement(self):
        raise NotImplementedError(
            "Single phrase question with multi-phrase statement not supported"
        )

    def write_commands_multi_phrase_question_single_phrase_statement(self):
        raise NotImplementedError(
            "Multi phrase question with single-phrase statement not supported"
        )

    def write_commands_multi_phrase_question_multi_phrase_statement(self):
        raise NotImplementedError(
            "Multi phrase question with multi-phrase statement not supported"
        )

    def write_question_commands_for_phrase(self, phrase, declarative_context):

        declarative_phrases = declarative_context.phrases
        first_declarative_context_phrase = declarative_phrases[0]
        key_words = phrase.key_words()

        self.commands.append(f".sctx {phrase.text}")
        for word in key_words.split():
            if word in first_declarative_context_phrase.text.split():
                self.commands.append(f".wg {word}")
                # remove the lookup word once it has been added to the commands
                key_words = key_words.replace(word, "").strip()
            # write a command to look up the first phrase in the lookup context
            else:
                self.commands.append(f".ph {first_declarative_context_phrase.text}")

                for index, declarative_phrase in enumerate(declarative_phrases):
                    if index == 0 or len(key_words) == 0:
                        # ignore the first phrase in the lookup context as this has already been processed
                        continue
                    else:
                        if word in declarative_phrase.text.split():
                            self.commands.append(f".sctx {declarative_phrase.text}")
                            self.commands.append(f".wg {word}")
                            key_words = key_words.replace(word, "").strip()


class ShortDeclarativeSentenceType:
    @staticmethod
    def write_question_commands_for_single_phrase_question(command_generator):
        command_generator.write_commands_single_phrase_question_single_phrase_statement()

    @staticmethod
    def write_question_commands_for_multi_phrase_question(command_generator):
        command_generator.write_commands_multi_phrase_question_single_phrase_statement(
            command_generator.question, command_generator.declarative_sentence
        )

    @staticmethod
    def write_answer_commands_for_long_answer(command_generator):
        command_generator.write_short_declaration_long_answer_commands()

    @staticmethod
    def write_answer_commands_for_short_answer(command_generator):
        command_generator.write_short_declaration_short_answer_commands()


class LongDeclarativeSentenceType:

    @staticmethod
    def write_question_commands_for_single_phrase_question(command_generator):
        command_generator.write_commands_single_phrase_question_multi_phrase_statement()

    @staticmethod
    def write_question_commands_for_multi_phrase_question(command_generator):
        command_generator.write_commands_multi_phrase_question_multi_phrase_statement()

    @staticmethod
    def write_answer_commands_for_long_answer(command_generator):
        command_generator.write_long_declaration_long_answer_commands()

    @staticmethod
    def write_answer_commands_for_short_answer(command_generator):
        command_generator.write_long_declaration_short_answer_commands()


class ShortQuestionType:

    @staticmethod
    def write_question(command_generator):
        command_generator.append_question()

    @staticmethod
    def write_question_commands(command_generator):

        command_generator.declarative_sentence_type.write_question_commands_for_single_phrase_question(
            command_generator
        )


class LongQuestionType:

    @staticmethod
    def write_question(command_generator):
        command_generator.append_question_phrases()

    @staticmethod
    def write_question_commands(command_generator):
        command_generator.declarative_sentence_type.write_question_commands_for_multi_phrase_question(
            command_generator
        )


class ShortAnswerType:

    @staticmethod
    def write_answer_commands(command_generator):

        command_generator.declarative_sentence_type.write_answer_commands_for_short_answer(
            command_generator
        )


class LongAnswerType:

    @staticmethod
    def write_answer_commands(command_generator):
        command_generator.declarative_sentence_type.write_answer_commands_for_long_answer(
            command_generator
        )


class AnnabellAbstractWordCollection:
    def __init__(self, text):
        self.text = text

    @staticmethod
    def remove_stopwords(a_string):
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        words = a_string.split()
        cleaned_string = " ".join([word for word in words if word not in stop_words])
        return cleaned_string

    @staticmethod
    def remove_suffixes(a_string):
        # remove any words prefixed with "-"
        words = a_string.split()
        cleaned_string = " ".join([word for word in words if not word.startswith("-")])
        return cleaned_string

    @staticmethod
    def remove_question_mark(a_string):
        # remove any question marks from the string
        cleaned_string = a_string.replace("?", "").strip()
        return cleaned_string

    def key_words(self):
        key_words = self.remove_stopwords(self.text)
        key_words = self.remove_suffixes(key_words)
        key_words = self.remove_question_mark(key_words)
        return key_words

    def words(self):
        return self.text.split()


class AnnabellAbstractPhrase(AnnabellAbstractWordCollection):
    def __init__(self, text, context):
        super().__init__(text)
        self.context = context


class AnnabellQuestionPhrase(AnnabellAbstractPhrase):
    pass


class AnnabellDeclarativePhrase(AnnabellAbstractPhrase):
    pass


class AnnabellAnswerPhrase(AnnabellAbstractPhrase):
    pass


class AnnabellAbstractContext(AnnabellAbstractWordCollection):
    def __init__(self, text, max_words_per_phrase):
        super().__init__(text)
        self.max_words_per_phrase = max_words_per_phrase
        self.phrases = self.phrases_in_context()

    def phrase_class(self):
        raise NotImplementedError("Subclasses must implement phrase_class method")

    def first_phrase(self):

        return self.phrases[0]

    def tail_phrases(self):
        return self.phrases[1:]

    def phrases_in_context(self):
        phrases = []
        context_words = self.text.split()
        number_of_phrases = (
            len(context_words) + self.max_words_per_phrase - 1
        ) // self.max_words_per_phrase
        for i in range(number_of_phrases):
            phrase_words = context_words[
                i * self.max_words_per_phrase : (i + 1) * self.max_words_per_phrase
            ]
            phrase_text = " ".join(phrase_words)
            phrase = self.phrase_class()(phrase_text, self)
            phrases.append(phrase)
        return phrases


class AnnabellQuestionContext(AnnabellAbstractContext):

    def get_question_type(self):
        if len(self.text.split()) > self.max_words_per_phrase:
            return LongQuestionType()
        else:
            return ShortQuestionType()

    def phrase_class(self):
        return AnnabellQuestionPhrase

    def word_group_chunks_matching_declarative_sentence(
        self, declarative_sentence_context
    ):
        word_group_chunks = []
        # find any consecutive word groups in chunks of 4 that are in the first phrase of the question context
        key_words_in_question_phrase = self.key_words().split()
        # remove any key words that are not in the declarative sentence
        key_words_in_question_phrase = [
            word
            for word in key_words_in_question_phrase
            if word in declarative_sentence_context.key_words()
        ]
        word_group = []

        for index, word in enumerate(self.words()):
            if word in key_words_in_question_phrase and len(word_group) == 0:
                word_group.append(word)
                key_words_in_question_phrase.remove(word)
            elif 0 < len(word_group) < 4:
                words_remaining_in_phrase_chunk = self.words()[
                    index : index + (4 - len(word_group))
                ]
                any_key_words_in_phrase_chunk = any(
                    w in key_words_in_question_phrase
                    for w in words_remaining_in_phrase_chunk
                )
                if any_key_words_in_phrase_chunk:
                    word_group.append(word)
                    if word in key_words_in_question_phrase:
                        key_words_in_question_phrase.remove(word)
                else:
                    word_group_chunks.append(word_group)
                    word_group = []
        if len(word_group) > 0:
            word_group_chunks.append(word_group)

        return word_group_chunks


class AnnabellDeclarativeContext(AnnabellAbstractContext):

    def get_declarative_sentence_type(self):
        if len(self.text.split()) > self.max_words_per_phrase:
            return LongDeclarativeSentenceType()
        else:
            return ShortDeclarativeSentenceType()

    def phrases_and_answer_words(self, answer_words):
        # construct a dictionary that contains each phrase as the key and the list of words from the answer that are in that phrase as the value
        phrase_answer_words = {}
        for phrase in self.phrases:
            phrase_answer_words[phrase] = []
            for word in answer_words:
                if word in phrase.text.split():
                    phrase_answer_words[phrase].append(word)
        return phrase_answer_words

    def phrase_class(self):
        return AnnabellDeclarativePhrase


class AnnabellAnswerContext(AnnabellAbstractContext):

    def get_answer_type(self):
        if len(self.text.split()) > self.max_words_per_phrase:
            return LongAnswerType()
        else:
            return ShortAnswerType()

    def phrase_class(self):
        return AnnabellAnswerPhrase