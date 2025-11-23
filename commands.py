import logging
import nltk
from nltk.corpus import stopwords
from config.global_config import GlobalConfig

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class AbstractAnnabellCommandGenerator:

    def __init__(self, declarative_sentence, max_words=10):
        self.declarative_sentence = declarative_sentence
        self.max_words = max_words
        self.commands = []

    def create_list_of_commands(self):
        raise NotImplementedError(
            "Subclasses must implement create_list_of_commands method"
        )

    def sentence_word_length(self):
        return len(self.declarative_sentence.split())

    def phrases_in_context(self, context):
        phrases = []
        context_words = context.split()
        number_of_phrases = (len(context_words) + self.max_words - 1) // self.max_words
        for i in range(number_of_phrases):
            phrase_words = context_words[i * self.max_words : (i + 1) * self.max_words]
            phrase = " ".join(phrase_words)
            phrases.append(phrase)
        return phrases


class AnnabellBaseCommandGenerator(AbstractAnnabellCommandGenerator):
    # creates the set of commands required to train annabell for a single training sample
    def __init__(
        self,
        sample_id,
        declarative_sentence,
        question,
        answer,
        is_pre_training=True,
        max_words=10,
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
            self.declarative_sentence, self.question, self.max_words
        )
        return generator

    def answer_command_generator(self):
        generator = AnnabellAnswerCommandGenerator(
            self.declarative_sentence, self.answer, self.max_words
        )
        return generator

    @staticmethod
    def blank_line():
        return "\n"

    def write_declarative_sentence(self):

        for phrase in self.phrases_in_context(self.declarative_sentence):
            self.commands.append(phrase)

    @staticmethod
    def informational_non_pretraining_command():
        return "# This is a non-pretraining sample. No commands to execute."

    def write_question_commands(self):
        self.question_command_generator.write_commands()
        self.commands.extend(self.question_command_generator.commands)

    def write_answer_commands(self):
        self.answer_command_generator.write_answer_commands()
        self.commands.extend(self.answer_command_generator.commands)

    def create_list_of_commands(self):

        self.commands = []

        if self.is_pre_training:
            self.commands.append("#id: " + str(self.sample_id))
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


class AnnabellAnswerCommandGenerator(AbstractAnnabellCommandGenerator):
    def __init__(
        self,
        declarative_sentence,
        answer,
        max_words,
    ):
        super().__init__(declarative_sentence, max_words)

        self.answer = answer
        self.declarative_sentence_type = self.get_declarative_sentence_type()
        self.answer_type = self.get_answer_type()

    def get_declarative_sentence_type(self):
        if len(self.declarative_sentence.split()) > self.max_words:
            return LongDeclarativeSentenceType()
        else:
            return ShortDeclarativeSentenceType()

    def get_answer_type(self):
        if len(self.answer.split()) > self.max_words:
            return LongQuestionType()
        else:
            return ShortQuestionType()

    @staticmethod
    def phrases_and_answer_words(phrases, answer_words):
        # construct a dictionary that contains each phrase as the key and the list of words from the answer that are in that phrase as the value
        phrase_answer_words = {}
        for phrase in phrases:
            phrase_answer_words[phrase] = []
            for word in answer_words:
                if word in phrase.split():
                    phrase_answer_words[phrase].append(word)
        return phrase_answer_words

    @staticmethod
    def answer_command_for_phrase_index(index):
        if index == 0:
            command = ".ph"
        else:
            command = ".sctx"

        return command

    def write_answer_commands(self):
        self.declarative_sentence_type.write_answer_commands(self)
        self.commands.append(".rw")

    def write_short_answer_commands(self):
        self.commands.append(f".ph {self.declarative_sentence}")
        # the model can only hold 4 words in its focus of attention, so the answer must be split and rewarded and outputted incrementally in chunks if the answer has more than 4 words

        answer_words = self.answer.split()
        if len(answer_words) < 4:
            self.commands.append(f".wg {self.answer}")
        else:
            self.commands.append(f".wg {" ".join(answer_words[:3])}")
            self.commands.append(".prw")
            self.commands.append(f".wg {" ".join(answer_words[3:])}")

    def write_long_answer_commands(self):
        # todo - instead of looking for the answer on the phrase via .ph instead look up the start of the context with .ph then sctx to find the phrase in the context that has the answer words
        answer_words = self.answer.split()
        phrases = self.phrases_in_context(self.declarative_sentence)
        # construct a dictionary that contains each phrase as the key and the list of words from the answer that are in that phrase as the value
        phrase_answer_words = self.phrases_and_answer_words(phrases, answer_words)
        # for each phrase in the declarative sentence, write the phrase command and the answer words

        for phrase_index, (phrase, answer_words) in enumerate(
            phrase_answer_words.items()
        ):
            if len(answer_words) == 0 and phrase_index > 0:
                continue
            else:
                self.commands.append(
                    f"{self.answer_command_for_phrase_index(phrase_index)} {phrase}"
                )
                if len(answer_words) == 0:
                    continue
                elif len(answer_words) < 4:
                    self.commands.append(f".wg {" ".join(answer_words)}")
                else:
                    self.commands.append(f".wg {" ".join(answer_words[:3])}")
                    self.commands.append(".prw")
                    self.commands.append(f".wg {" ".join(answer_words[3:])}")
                    self.commands.append(".prw")


class AnnabellQuestionCommandGenerator(AbstractAnnabellCommandGenerator):
    def __init__(
        self,
        declarative_sentence,
        question,
        max_words,
    ):
        super().__init__(declarative_sentence, max_words)
        self.question = question
        self.declarative_sentence_type = self.get_declarative_sentence_type()
        self.question_type = self.get_question_type()

    def get_declarative_sentence_type(self):
        if len(self.declarative_sentence.split()) > self.max_words:
            return LongDeclarativeSentenceType()
        else:
            return ShortDeclarativeSentenceType()

    def get_question_type(self):
        if len(self.question.split()) > self.max_words:
            return LongQuestionType()
        else:
            return ShortQuestionType()

    def question_word_length(self):
        return len(self.question.split())

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
        self.commands.append(self.question)

    def append_question_phrases(self):
        # split the question into phrases of max_words length
        for phrase in self.phrases_in_context(self.question):
            self.commands.append(phrase)

    def key_words_in_phrase(self, phrase):
        key_words = self.remove_stopwords(phrase)
        key_words = self.remove_suffixes(key_words)
        key_words = self.remove_question_mark(key_words)
        return key_words

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

    def write_question_commands_for_single_phrase_context(self):
        self.declarative_sentence_type.write_question_commands_for_single_phrase_question(
            self
        )

    def write_commands_single_phrase_question_single_phrase_statement(
        self, question, lookup_context
    ):
        key_words = self.key_words_in_phrase(question)
        for word in key_words.split():
            if word in lookup_context.split():
                self.commands.append(f".wg {word}")

    def write_commands_single_phrase_question_multi_phrase_statement(
        self, question, lookup_context
    ):

        self.write_question_commands_for_phrase(question, lookup_context)

    def write_commands_multi_phrase_question_single_phrase_statement(
        self, question_context, lookup_context
    ):
        for phrase in self.phrases_in_context(question_context):
            self.commands.append(f".sctx {phrase}")
            for word in self.key_words_in_phrase(phrase).split():
                if word in lookup_context.split():
                    self.commands.append(f".wg {word}")

    def write_commands_multi_phrase_question_multi_phrase_statement(
        self, question_context, lookup_context
    ):
        for phrase in self.phrases_in_context(question_context):
            self.commands.append(f".sctx {phrase}")
            self.write_commands_single_phrase_question_multi_phrase_statement(
                phrase, lookup_context
            )

    def write_question_commands_for_long_context(
        self, question_context, lookup_context
    ):
        # split the context into phrases of max_words length
        for phrase in self.phrases_in_context(question_context):
            self.write_question_commands_for_phrase(phrase, lookup_context)

    def write_question_commands_for_phrase(self, phrase, statement_context):
        statement_phrases = self.phrases_in_context(statement_context)
        first_lookup_context_phrase = statement_phrases[0]
        key_words = self.key_words_in_phrase(phrase)

        self.commands.append(f".sctx {phrase}")
        for word in key_words.split():
            if word in first_lookup_context_phrase.split():
                self.commands.append(f".wg {word}")
                # remove the lookup word once it has been added to the commands
                key_words = key_words.replace(word, "").strip()
            # write a command to look up the first phrase in the lookup context
            else:
                self.commands.append(f".ph {first_lookup_context_phrase}")

                for index, statement_phrase in enumerate(statement_phrases):
                    if index == 0 or len(key_words) == 0:
                        # ignore the first phrase in the lookup context as this has already been processed
                        continue
                    else:
                        if word in statement_phrase.split():
                            self.commands.append(f".sctx {statement_phrase}")
                            self.commands.append(f".wg {word}")
                            key_words = key_words.replace(word, "").strip()


class ShortDeclarativeSentenceType:
    @staticmethod
    def write_question_commands_for_single_phrase_question(command_generator):
        command_generator.write_commands_single_phrase_question_single_phrase_statement(
            command_generator.question, command_generator.declarative_sentence
        )

    @staticmethod
    def write_question_commands_for_multi_phrase_question(command_generator):
        command_generator.write_commands_multi_phrase_question_single_phrase_statement(
            command_generator.question, command_generator.declarative_sentence
        )

    @staticmethod
    def write_answer_commands(command_generator):
        command_generator.write_short_answer_commands()


class LongDeclarativeSentenceType:

    @staticmethod
    def write_question_commands_for_single_phrase_question(command_generator):
        command_generator.write_commands_single_phrase_question_multi_phrase_statement(
            command_generator.question, command_generator.declarative_sentence
        )

    @staticmethod
    def write_question_commands_for_multi_phrase_question(command_generator):
        command_generator.write_commands_multi_phrase_question_multi_phrase_statement(
            command_generator.question, command_generator.declarative_sentence
        )

    @staticmethod
    def write_answer_commands(command_generator):
        command_generator.write_long_answer_commands()


class ShortQuestionType:

    @staticmethod
    def write_question(command_generator):
        command_generator.append_question()

    @staticmethod
    def write_question_commands(command_generator):

        command_generator.write_question_commands_for_single_phrase_context()


class LongQuestionType:

    @staticmethod
    def write_question(command_generator):
        command_generator.append_question_phrases()

    @staticmethod
    def write_question_commands(command_generator):
        command_generator.write_question_commands_for_long_context(
            command_generator.question, command_generator.declarative_sentence
        )