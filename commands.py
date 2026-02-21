import logging
from operator import indexOf

import nltk
from nltk.corpus import stopwords

from annabell_utilities import AnnabellLogfileInterpreter
from config.global_config import GlobalConfig
from collections import deque

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class MissingAnswerWordsException(Exception):
    pass


class ToFewLookupWordGroupsException(Exception):
    pass


class LIFOQueue:
    def __init__(self):
        self._items = deque()

    def enqueue(self, item):
        self._items.append(item)

    def dequeue(self):
        return self._items.pop()

    def is_empty(self):
        return len(self._items) == 0

    def items(self):
        return list(self._items)

    def peek(self):
        if not self.is_empty():
            return self._items[-1]
        else:
            return None


class AnnabellGoalStack(LIFOQueue):
    def __init__(self, limit):
        self.limit = limit
        super().__init__()

    def size(self):
        return len(self.items())


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

    @staticmethod
    def phrase_command():
        return ".ph"

    @staticmethod
    def search_context_command():
        return ".sctx"

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
        self.question_command_generator = self.question_command_generator()
        self.answer_command_generator = self.answer_command_generator()

    def question_command_generator(self):
        generator = AnnabellQuestionCommandGenerator(
            self.declarative_sentence.text, self.question, self.answer, self.max_words
        )
        return generator

    def answer_command_generator(self):
        generator = AnnabellAnswerCommandGenerator(
            self.declarative_sentence.text,
            self.answer,
            self.question_command_generator,
            self.max_words,
        )
        return generator

    @staticmethod
    def blank_line():
        return "\n"

    @staticmethod
    def time_command():
        return ".time"

    def write_declarative_sentence(self):
        for phrase in self.declarative_sentence.phrases:
            self.commands.append(phrase.text)
        self.commands.append(AnnabellLogfileInterpreter.end_of_declaration_string())

    @staticmethod
    def informational_non_pretraining_command():
        return "# This is a non-pretraining sample. No commands to execute."

    @staticmethod
    def error_generating_pretraining_command():
        return "# There was an error generating commands for this pre-training sample."

    @staticmethod
    def stat_command():
        return ".stat"

    def write_question_commands(self):
        self.question_command_generator.write_commands()
        self.commands.extend(self.question_command_generator.commands)

    def write_answer_commands(self):
        self.answer_command_generator.write_answer_commands()
        self.commands.extend(self.answer_command_generator.commands)

    def write_sample_id(self):
        self.commands.append("#id: " + str(self.sample_id))

    def write_time_command(self):
        self.commands.append(self.time_command())
        self.commands.append(AnnabellLogfileInterpreter.end_of_time_string())

    def write_stat_command(self):
        if global_config.log_stats():
            self.commands.append(self.stat_command())
            self.commands.append(AnnabellLogfileInterpreter.end_of_stats_string())

    def create_list_of_commands(self):
        try:
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
                self.commands.append(
                    AnnabellLogfileInterpreter.end_of_commands_string()
                )
                self.write_time_command()
                self.write_stat_command()

            else:
                self.commands.append(self.informational_non_pretraining_command())
            return self.commands
        except Exception as e:
            logger.error(
                f"Error creating commands for sample {self.sample_id}: {e} Declarative sentence: '{self.declarative_sentence.text}' Question: '{self.question}' Answer: '{self.answer}"
            )
            self.commands = []
            self.commands.append(self.error_generating_pretraining_command())

        return self.commands


class AnnabellTestingCommandGenerator(AnnabellBaseCommandGenerator):
    # creates the commands required to test annabell for a single testing sample
    def __init__(
        self,
        sample_id,
        question,
    ):
        super().__init__(
            sample_id, declarative_sentence="", question=question, answer=""
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
    def __init__(self, declarative_sentence, answer, question_generator, max_words):
        super().__init__(declarative_sentence, max_words)

        self.answer = AnnabellAnswerContext(answer)
        self.answer_type = self.answer.get_answer_type()
        self.question_generator = question_generator

    @staticmethod
    def answer_command_for_phrase_index(index):
        if index == 0:
            command = ".ph"
        else:
            command = ".sctx"

        return command

    def write_answer_commands(self):
        self.answer_type.write_answer_commands(self)

    def write_commands_short_answer_single_phrase_statement(self):
        self.write_commands_long_answer_multi_phrase_statement()

    def write_commands_long_answer_single_phrase_statement(self):
        self.commands.append(f".ph {self.declarative_sentence.text}")
        answer_word_group_chunks = self.answer.word_group_chunks_matching_sentence(
            self.declarative_sentence
        )
        for index, word_group_chunk in enumerate(answer_word_group_chunks):
            word_group_text = " ".join(word_group_chunk)
            self.commands.append(f".wg {word_group_text}")
            if index < len(answer_word_group_chunks) - 1:
                self.commands.append(".prw")
            else:
                self.commands.append(".rw")

    def lookup_declarative_sentence(self):
        candidate_target_sentences = []
        drop_goal = False
        for declarative_sentence in self.declarative_sentence.phrases:
            if declarative_sentence.contains_word_group(
                self.question_generator.current_word_group
            ):
                candidate_target_sentences.append(declarative_sentence)
            else:
                pass
        if len(candidate_target_sentences) == 0:
            logger.critical(
                f"No declarative sentences found for lookup word group: {self.question_generator.current_word_group}"
            )
            raise Exception
        elif len(candidate_target_sentences) > 1:
            target_sentence = candidate_target_sentences[0]
            logger.info(
                f"Multiple declarative sentences found for lookup word group: {self.question_generator.current_word_group}, checking goal stack for disambiguation."
            )
            for candidate_target_sentence in candidate_target_sentences:
                if candidate_target_sentence.contains(
                    self.question_generator.goal_stack.peek()
                ):
                    target_sentence = candidate_target_sentence
                    drop_goal = True
                else:
                    pass
        else:
            target_sentence = candidate_target_sentences[0]
            if self.question_generator.goal_stack.is_empty():
                drop_goal = False
            elif target_sentence.contains_word_group(
                self.question_generator.goal_stack.peek()
            ):
                drop_goal = True
        return target_sentence, drop_goal

    def lookup_declarative_sentence_with_word_group(self, word_group):
        found_declarative_sentence = None
        for declarative_sentence in self.declarative_sentence.phrases:
            if declarative_sentence.contains_word_group(word_group):
                found_declarative_sentence = declarative_sentence
                break
        return found_declarative_sentence

    def write_answer_words_in_chunks(
        self, answer_words_remaining, answer_word_group_chunks, lookup_phrase=None
    ):
        # check if all the remaining answer words are in the word group chunks. if so, drop any reaming goals.
        if (
            self.words_in_word_groups(answer_word_group_chunks)
            == self.answer.answer_words_remaining
            and not self.question_generator.goal_stack.is_empty()
        ):
            while not self.question_generator.goal_stack.is_empty():
                self.commands.append(self.drop_goal_command())
                self.question_generator.goal_stack.dequeue()

        if lookup_phrase:
            self.commands.append(
                f"{self.search_context_command()} {lookup_phrase.text}"
            )
        for index, word_group_chunk in enumerate(answer_word_group_chunks):
            if all(word in answer_words_remaining for word in word_group_chunk):
                word_group_text = " ".join(word_group_chunk)
                self.commands.append(f".wg {word_group_text}")
                for word in word_group_chunk:
                    answer_words_remaining.remove(word)
                if len(answer_words_remaining) > 0:
                    self.commands.append(".prw")
                else:
                    self.commands.append(".rw")

    def write_get_goal_phrase_command(self):
        self.commands.append(self.get_goal_phrase_command())

    @staticmethod
    def get_goal_phrase_command():
        return ".ggp"

    @staticmethod
    def drop_goal_command():
        return ".drop_goal"

    def write_commands_short_answer_multi_phrase_statement(self):
        self.write_commands_long_answer_multi_phrase_statement()

    @staticmethod
    def words_in_word_groups(list_of_word_groups):
        words_list = [word for word_group in list_of_word_groups for word in word_group]
        return words_list

    def drop_goals_for_current_sentence(self, declarative_sentence):
        drop_goal = True
        while drop_goal and not self.question_generator.goal_stack.is_empty():
            current_goal = self.question_generator.goal_stack.peek()
            if declarative_sentence.contains_word_group(current_goal):
                self.commands.append(self.drop_goal_command())
                self.question_generator.goal_stack.dequeue()
            else:
                drop_goal = False

    def write_commands_long_answer_multi_phrase_statement(self):
        if not self.question_generator.goal_stack.is_empty():
            self.write_get_goal_phrase_command()
        declarative_sentence, drop_goal = self.lookup_declarative_sentence()
        self.commands.append(f"{self.phrase_command()} {declarative_sentence.text}")
        if drop_goal:
            self.drop_goals_for_current_sentence(declarative_sentence)
        else:
            pass

        # check if any answer words are in the first
        answer_word_group_chunks = self.answer.word_group_chunks_matching_sentence(
            declarative_sentence
        )

        self.write_answer_words_in_chunks(
            self.answer.answer_words_remaining, answer_word_group_chunks
        )
        # todo: remove unnecessary passing of answer_words_remaining
        if (
            self.question_generator.goal_stack.is_empty()
            and self.answer.answer_words_remaining
        ):
            # check the remaining declarative sentences for any remaining answer words
            # todo: handles the case where an answer word group is split across multiple phrases
            for declarative_phrase in self.declarative_sentence.tail_phrases():
                answer_word_group_chunks = (
                    self.answer.word_group_chunks_matching_sentence(declarative_phrase)
                )
                self.write_answer_words_in_chunks(
                    self.answer.answer_words_remaining,
                    answer_word_group_chunks,
                    lookup_phrase=declarative_phrase,
                )

        # using the goal stack lookup remaining sentences and check them for remaining answer words
        while not self.question_generator.goal_stack.is_empty():
            current_goal = self.question_generator.goal_stack.peek()
            found_declarative_sentence = (
                self.lookup_declarative_sentence_with_word_group(current_goal)
            )
            if found_declarative_sentence:
                self.commands.append(
                    f"{self.search_context_command()} {found_declarative_sentence.text}"
                )
                self.drop_goals_for_current_sentence(found_declarative_sentence)
                answer_word_group_chunks = (
                    self.answer.word_group_chunks_matching_sentence(
                        found_declarative_sentence
                    )
                )
                self.write_answer_words_in_chunks(
                    self.answer.answer_words_remaining, answer_word_group_chunks
                )
            else:
                pass
        if len(self.answer.answer_words_remaining) != 0:
            # if there are still answer words remaining, raise an exception
            error_msg = f"Not all answer words were found in the declarative sentence. missing answer words: {self.answer.answer_words_remaining}"
            logger.critical(error_msg)
            raise MissingAnswerWordsException(error_msg)

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


class CandidateQuestionPhrase:
    def __init__(self, question_phrase, word_groups):
        self.question_phrase = question_phrase
        self.word_groups = word_groups
        self.declarative_lookup_word_groups = []

    def __repr__(self):
        # Displays the phrase text and how many word groups it has
        return f"<CandidateQuestionPhrase: '{self.question_phrase.text}' | Groups: {self.word_groups}>"

    def word_groups_text(self):
        return [" ".join(word_group) for word_group in self.word_groups]

    def question_phrase_text(self):
        return self.question_phrase.text

    def add_lookup_word_groups(self, lookup_word_groups):
        self.declarative_lookup_word_groups.extend(lookup_word_groups)

    def declarative_non_lookup_word_groups(self):
        word_groups = []
        for word_group in self.word_groups:
            if word_group not in self.declarative_lookup_word_groups:
                word_groups.append(word_group)
        return word_groups


class AnnabellQuestionCommandGenerator(AbstractAnnabellCommandGenerator):
    def __init__(
        self,
        declarative_sentence,
        question,
        answer,
        max_words,
    ):
        super().__init__(declarative_sentence, max_words)
        self.candidate_question_phrases_and_word_groups = []
        self.question = AnnabellQuestionContext(question, max_words)
        self.question_type = self.question.get_question_type()
        self.answer = AnnabellAnswerContext(answer)
        self.current_word_group = []
        self.goal_stack = AnnabellGoalStack(global_config.goal_stack_limit())
        self.current_context = None

    def current_word_group_text(self):
        return " ".join(self.current_word_group)

    def set_candidate_question_phrases_and_word_groups(self):
        self.candidate_question_phrases_and_word_groups = []
        for phrase_index, question_phrase in enumerate(self.question.phrases):

            declarative_context_word_group_chunks = []
            for declarative_phrase in self.declarative_sentence.phrases:
                word_group_chunks = question_phrase.word_group_chunks_matching_sentence(
                    declarative_phrase
                )
                declarative_context_word_group_chunks.extend(word_group_chunks)
            if len(declarative_context_word_group_chunks) == 0:
                pass
            else:
                self.add_candidate_phrase(
                    CandidateQuestionPhrase(
                        question_phrase, declarative_context_word_group_chunks
                    )
                )

        self.set_lookup_word_groups_for_candidate_phrases()

    def all_candidate_phrase_word_groups(self):
        word_groups = []
        for candidate_phrase in self.candidate_question_phrases_and_word_groups:
            word_groups.extend(candidate_phrase.word_groups)
        return word_groups

    def non_lookup_candidate_phrase_word_groups(self):
        word_groups = []
        for candidate_phrase in self.candidate_question_phrases_and_word_groups:
            word_groups.extend(candidate_phrase.declarative_non_lookup_word_groups())
        return word_groups

    def filter_candidate_question_phrases_and_word_groups(self):
        """if the number of word groups exceeds the goal stack limit + 1,
        then remove word groups according to the following rules:
        1) remove non-lookup word_groups first
        2) remove the word_groups with the shortest length first
        3) keep removing word groups until the number of word groups equals the
        size of the goal stack plus one"""
        while len(self.all_candidate_phrase_word_groups()) > self.goal_stack.limit + 1:
            self.remove_word_group_from_candidate_phrases()

    @staticmethod
    def shortest_word_group_in(list_of_word_groups):
        shortest_word_group = None
        for word_group in list_of_word_groups:

            if shortest_word_group is None or len(word_group) < len(
                shortest_word_group
            ):
                shortest_word_group = word_group
        return shortest_word_group

    def remove_word_group_from_candidate_phrases(self):
        if self.non_lookup_candidate_phrase_word_groups():
            shortest_word_group = self.shortest_word_group_in(
                self.non_lookup_candidate_phrase_word_groups()
            )
            self.remove_non_lookup_candidate_phrase(shortest_word_group)
        else:
            shortest_word_group = self.shortest_word_group_in(
                self.all_candidate_phrase_word_groups()
            )
            self.remove_lookup_candidate_phrase(shortest_word_group)

    def remove_non_lookup_candidate_phrase(self, word_group_to_remove):
        for candidate_phrase in self.candidate_question_phrases_and_word_groups:
            if (
                word_group_to_remove
                in candidate_phrase.declarative_non_lookup_word_groups()
            ):
                candidate_phrase.word_groups.remove(word_group_to_remove)

    def remove_lookup_candidate_phrase(self, word_group_to_remove):
        for candidate_phrase in self.candidate_question_phrases_and_word_groups:
            if word_group_to_remove in candidate_phrase.declarative_lookup_word_groups:
                candidate_phrase.word_groups.remove(word_group_to_remove)
                candidate_phrase.declarative_lookup_word_groups.remove(
                    word_group_to_remove
                )

    def add_candidate_phrase(self, candidate_question_phrase):
        self.candidate_question_phrases_and_word_groups.append(
            candidate_question_phrase
        )

    def get_declarative_candidate_phrases_and_word_groups(self):
        result = [
            candidate_phrase
            for candidate_phrase in self.candidate_question_phrases_and_word_groups
            if candidate_phrase.is_declaration_lookup_candidate_phrase()
        ]
        return result

    def candidate_phrases_with_declarative_lookup_words(self):
        result = [
            candidate_phrase
            for candidate_phrase in self.candidate_question_phrases_and_word_groups
            if candidate_phrase.declarative_lookup_word_groups
        ]
        return result

    def answer_sentence_word_group_chunks(self):
        return self.answer.word_group_chunks_matching_sentence(
            self.declarative_sentence
        )

    def last_phrase_index(self):
        return len(self.question.phrases) - 1

    def write_search_context_command(self, text):
        if self.current_context != text:
            self.commands.append(f".sctx {text}")
            self.current_context = text

    def sort_candidate_question_phrases_and_word_groups(self):
        # sort the phrases so that the phrase containing key words which are in the declarative phrase that contains the answer words comes first.

        answer_words = self.answer.words()
        target_keywords = set()

        phrases_with_answer_words = self.declarative_sentence.phrases_and_answer_words(
            answer_words
        )

        for phrase, found_answer_words in phrases_with_answer_words.items():
            if len(found_answer_words) > 0:
                target_keywords.update(phrase.key_words().split())

        def phrase_sort_key(candidate_phrase):
            total_matching_words = 0
            question_phrase_keywords = (
                candidate_phrase.question_phrase.key_words().split()
            )
            for keyword in question_phrase_keywords:
                if keyword in target_keywords:
                    total_matching_words += 1
            return (len(candidate_phrase.word_groups), -total_matching_words)

        self.candidate_question_phrases_and_word_groups.sort(key=phrase_sort_key)

    def all_lookup_word_groups(self):
        lookup_word_groups = []
        for candidate_phrase in self.candidate_question_phrases_and_word_groups:
            lookup_word_groups.extend(candidate_phrase.word_groups)
        return lookup_word_groups

    def get_candidate_lookup_declarative_phrase(self):
        lookup_word_groups = self.all_lookup_word_groups()
        # find the declarative_phrase that has the most word groups in the lookup word groups
        max_word_groups = 0
        selected_declarative_phrase = None
        for declarative_phrase in self.declarative_sentence.phrases:
            word_groups_for_declarative_phrase = []
            for word_group in lookup_word_groups:
                if declarative_phrase.contains_word_group(word_group):
                    word_groups_for_declarative_phrase.append(word_group)
            if len(word_groups_for_declarative_phrase) > max_word_groups:
                max_word_groups = len(word_groups_for_declarative_phrase)
                selected_declarative_phrase = declarative_phrase
        return selected_declarative_phrase

    def set_lookup_word_groups_for_candidate_phrases(self):
        declarative_phrase = self.get_candidate_lookup_declarative_phrase()
        for candidate_phrase in self.candidate_question_phrases_and_word_groups:
            word_groups_for_declarative_phrase = (
                candidate_phrase.question_phrase.word_group_chunks_matching_sentence(
                    declarative_phrase
                )
            )
            if word_groups_for_declarative_phrase:
                candidate_phrase.add_lookup_word_groups(
                    word_groups_for_declarative_phrase
                )

    def write_lookup_declarative_sentence_commands(self):

        for candidate_phrase_index, candidate_phrase in enumerate(
            self.candidate_question_phrases_and_word_groups
        ):
            if candidate_phrase.declarative_lookup_word_groups:
                self.question_type.write_search_context_command(
                    self, candidate_phrase.question_phrase_text()
                )
                for chunk_index, declarative_context_word_group_chunk in enumerate(
                    candidate_phrase.declarative_lookup_word_groups
                ):

                    word_group_text = " ".join(declarative_context_word_group_chunk)

                    if (
                        candidate_phrase_index
                        == len(self.candidate_phrases_with_declarative_lookup_words())
                        - 1
                        and chunk_index
                        == len(candidate_phrase.declarative_lookup_word_groups) - 1
                    ):
                        self.set_current_word_group(word_group_text)
                    else:
                        self.append_goal(f"{word_group_text}")

    def write_non_lookup_commands(self):

        for candidate_phrase in self.candidate_question_phrases_and_word_groups:
            non_lookup_word_group_chunks = (
                candidate_phrase.declarative_non_lookup_word_groups()
            )

            if len(non_lookup_word_group_chunks) > 0:
                self.question_type.write_search_context_command(
                    self, candidate_phrase.question_phrase_text()
                )
                for chunk_index, non_lookup_sentence_word_group_chunk in enumerate(
                    non_lookup_word_group_chunks
                ):
                    word_group_text = " ".join(non_lookup_sentence_word_group_chunk)
                    self.append_goal(f"{word_group_text}")

    def all_declarative_phrase_lookup_word_groups(self):
        lookup_word_groups = []
        for candidate_phrase in self.candidate_question_phrases_and_word_groups:
            for word_group in candidate_phrase.declarative_lookup_word_groups:
                lookup_word_groups.append(word_group)
        return lookup_word_groups

    def write_commands_from_candidate_question_phrases_and_word_groups(self):

        # self.sort_candidate_question_phrases_and_word_groups()
        self.set_candidate_question_phrases_and_word_groups()
        self.filter_candidate_question_phrases_and_word_groups()
        # write commands to set goals for word groups in that are not used in the initial lookup
        if global_config.write_non_lookup_commands():
            self.write_non_lookup_commands()
        else:
            pass
        # write the commands that will initially guide finding the declarative sentence phrase
        self.write_lookup_declarative_sentence_commands()

        if global_config.exclude_samples_with_fewer_than_2_lookups() and (
            len(self.all_declarative_phrase_lookup_word_groups()) < 2
        ):
            error_msg = f"Sample excluded due to having fewer than 2 lookup word groups in declarative sentence. Lookup word groups: {self.all_declarative_phrase_lookup_word_groups()}"
            logger.warning(error_msg)
            raise ToFewLookupWordGroupsException(error_msg)

        # finally write the word group command that will be used for the lookup of the first declarative sentence
        if len(self.all_declarative_phrase_lookup_word_groups()) > 1:
            self.write_word_group_command(self.current_word_group_text())
        else:
            self.append_goal(self.current_word_group_text())

    def write_commands(self):
        self.commands = []
        self.write_question()
        self.commands.append(AnnabellLogfileInterpreter.end_of_question_string())
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

    def append_goal(self, goal_word_group_text):
        goal_word_group = [goal_word_group_text]
        self.goal_stack.enqueue(goal_word_group)
        goal_word_group_command_text = (
            f"{self.push_goal_command()} {goal_word_group_text}"
        )
        self.commands.append(goal_word_group_command_text)

    def set_current_word_group(self, word_group_text):
        self.current_word_group.clear()
        self.current_word_group.append(word_group_text)

    def write_word_group_command(self, word_group_text):
        word_group_command_text = f"{self.word_group_command()} {word_group_text}"
        self.commands.append(word_group_command_text)

    @staticmethod
    def word_group_command():
        return ".wg"

    @staticmethod
    def push_goal_command():
        return ".pg"

    def write_commands_single_phrase_question_single_phrase_statement(self):
        self.write_commands_multi_phrase_question_multi_phrase_statement()

    def write_commands_single_phrase_question_multi_phrase_statement(self):
        phrases_and_word_group_chunks_for_question = (
            self.declarative_sentence.phrases_and_word_group_chunks_for_question(
                self.question
            )
        )

        word_group_texts = []
        for word_group_chunks in phrases_and_word_group_chunks_for_question.values():
            for word_group_chunk in word_group_chunks:
                word_group_text = " ".join(word_group_chunk)
                word_group_texts.append(word_group_text)

        last_text_index = len(word_group_texts) - 1
        for index, word_group_text in enumerate(word_group_texts):
            if index < last_text_index or len(word_group_texts) == 1:
                self.append_goal(word_group_text)
                self.set_current_word_group(word_group_text)
            else:
                self.set_current_word_group(word_group_text)
                self.write_word_group_command(word_group_text)

    def write_commands_multi_phrase_question_single_phrase_statement(self):
        self.write_commands_multi_phrase_question_multi_phrase_statement()

    def write_commands_multi_phrase_question_multi_phrase_statement(self):
        self.write_commands_from_candidate_question_phrases_and_word_groups()


class ShortDeclarativeSentenceType:
    @staticmethod
    def write_question_commands_for_single_phrase_question(command_generator):
        command_generator.write_commands_single_phrase_question_single_phrase_statement()

    @staticmethod
    def has_multiple_phrases():
        return False

    @staticmethod
    def write_question_commands_for_multi_phrase_question(command_generator):
        command_generator.write_commands_multi_phrase_question_single_phrase_statement()

    @staticmethod
    def write_answer_commands_for_long_answer(command_generator):
        command_generator.write_commands_long_answer_multi_phrase_statement()

    @staticmethod
    def write_answer_commands_for_short_answer(command_generator):
        command_generator.write_commands_short_answer_multi_phrase_statement()


class LongDeclarativeSentenceType:

    @staticmethod
    def has_multiple_phrases():
        return True

    @staticmethod
    def write_question_commands_for_single_phrase_question(command_generator):
        command_generator.write_commands_single_phrase_question_multi_phrase_statement()

    @staticmethod
    def write_question_commands_for_multi_phrase_question(command_generator):
        command_generator.write_commands_multi_phrase_question_multi_phrase_statement()

    @staticmethod
    def write_answer_commands_for_long_answer(command_generator):
        command_generator.write_commands_long_answer_multi_phrase_statement()

    @staticmethod
    def write_answer_commands_for_short_answer(command_generator):
        command_generator.write_commands_long_answer_multi_phrase_statement()


class ShortQuestionType:

    @staticmethod
    def has_multiple_phrases():
        return False

    @staticmethod
    def write_search_context_command(command_generator, text):
        pass

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
    def has_multiple_phrases():
        return True

    @staticmethod
    def write_search_context_command(command_generator, text):
        command_generator.write_search_context_command(text)

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

    def __repr__(self):
        # Displays the class name and the text content (e.g., <AnnabellQuestionPhrase: 'what is commands'>)
        # Slices text [:50] to keep the debug view clean if text is very long
        display_text = (self.text[:50] + "..") if len(self.text) > 50 else self.text
        return f"<{self.__class__.__name__}: '{display_text}'>"

    @staticmethod
    def max_words_per_word_group():
        return 4

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
    def is_stopword(a_word):
        stop_words = set(stopwords.words("english"))
        return a_word in stop_words

    def is_content_word(self, a_word):
        return not self.is_stopword(a_word)

    def contains_content_words(self, a_word_list):
        for word in a_word_list:
            if self.is_content_word(word):
                return True
        return False

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

    def key_words_in_phrase(self, sentence):
        key_words = self.key_words().split()
        # remove any keywords that are not in the declarative sentence
        key_words = [word for word in key_words if word in sentence.key_words().split()]
        return key_words

    def word_group_chunks_matching_sentence(self, sentence):
        word_group_chunks = []
        # find any consecutive word groups in chunks of 4 that are in the sentence
        key_words_in_sentence = self.key_words_in_phrase(sentence)
        word_group = []

        for index, word in enumerate(self.words()):
            if word in key_words_in_sentence and len(word_group) == 0:
                word_group.append(word)
                key_words_in_sentence.remove(word)
            elif 0 < len(word_group) < 4:
                words_remaining_in_phrase_chunk = self.words()[
                    index : index + (4 - len(word_group))
                ]
                any_key_words_in_phrase_chunk = any(
                    w in key_words_in_sentence for w in words_remaining_in_phrase_chunk
                )
                proposed_word_group = word_group + [word]
                if any_key_words_in_phrase_chunk and sentence.contains_word_group(
                    proposed_word_group
                ):
                    word_group.append(word)
                    if word in key_words_in_sentence:
                        key_words_in_sentence.remove(word)
                else:
                    word_group_chunks.append(word_group)
                    word_group = []
                    if word in key_words_in_sentence:
                        word_group.append(word)
                        key_words_in_sentence.remove(word)
                if len(word_group) == 4:
                    word_group_chunks.append(word_group)
                    word_group = []
        if len(word_group) > 0:
            word_group_chunks.append(word_group)

        return word_group_chunks

    def contains_word_group(self, word_group):
        word_group_text = " ".join(word_group)
        if word_group_text in self.text:
            return True
        else:
            return False

    def key_words_for_phrase(self, phrase):
        return phrase.key_words_in_context(self)


class AnnabellAbstractPhrase(AnnabellAbstractWordCollection):
    def __init__(self, text, context):
        super().__init__(text)
        self.context = context

    def key_words_in_context(self, context):
        return context.key_words()


class AnnabellQuestionPhrase(AnnabellAbstractPhrase):
    pass


class AnnabellDeclarativePhrase(AnnabellAbstractPhrase):
    pass


class AnnabellAnswerPhrase(AnnabellAbstractPhrase):
    def __init__(self, text, context):
        super().__init__(text, context)

    def answer_words_remaining(self):
        return self.context.answer_words_remaining

    def key_words(self):
        return self.text

    def key_words_in_phrase(self, sentence):
        return sentence.key_words_for_phrase(self)

    @staticmethod
    def common_consecutive_words(text1, text2):
        words1 = text1.split()
        words2 = text2.split()
        common_words = []
        for word in words1:
            candidate_words = common_words + [word]
            candidate_phrase = " ".join(candidate_words)
            if candidate_phrase in text2 and word in words2:
                common_words.append(word)
            else:
                break
        return common_words

    def key_words_in_context(self, context):
        key_words = []
        answer_words_in_context = []
        candidate_answer_words_in_context = self.common_consecutive_words(
            " ".join(self.answer_words_remaining()), context.text
        )
        # only include the answer word in the context if the next answer word is also in the context
        if len(candidate_answer_words_in_context) == 1:
            if indexOf(
                context.text.split(), candidate_answer_words_in_context[0]
            ) == len(context.words()) - 1 or not self.is_stopword(
                candidate_answer_words_in_context[0]
            ):
                answer_words_in_context.extend(candidate_answer_words_in_context)
            else:
                pass
        else:
            answer_words_in_context.extend(candidate_answer_words_in_context)

        if " ".join(answer_words_in_context) in context.text:
            key_words.extend(answer_words_in_context)
        else:
            for index, declarative_word in enumerate(context.words()):
                if (
                    declarative_word in answer_words_in_context
                    and len(answer_words_in_context) > 1
                    and index < len(context.words()) - 1
                ):
                    next_declarative_word = context.words()[index + 1]
                    if next_declarative_word == answer_words_in_context[1]:
                        key_words.append(declarative_word)
                        key_words.append(next_declarative_word)
                        answer_words_in_context.remove(declarative_word)
                        answer_words_in_context.remove(next_declarative_word)
                elif declarative_word in answer_words_in_context:
                    key_words.append(declarative_word)
                    answer_words_in_context.remove(declarative_word)
        return key_words


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

    def word_group_chunks_matching_sentence(self, sentence):
        word_group_chunks = []
        for phrase in self.phrases:
            word_group_chunks.extend(
                phrase.word_group_chunks_matching_sentence(sentence)
            )
        return word_group_chunks


class AnnabellQuestionContext(AnnabellAbstractContext):

    def get_question_type(self):
        if len(self.text.split()) > self.max_words_per_phrase:
            return LongQuestionType()
        else:
            return ShortQuestionType()

    def phrase_class(self):
        return AnnabellQuestionPhrase


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

            candidate_answer_words = []
            for word in answer_words:
                if word in phrase.text.split():
                    candidate_answer_words.append(word)
            if self.contains_content_words(candidate_answer_words):
                phrase_answer_words[phrase] = candidate_answer_words
        return phrase_answer_words

    def phrase_class(self):
        return AnnabellDeclarativePhrase

    def phrases_and_word_group_chunks_for_question(self, question_phrase):
        phrases_and_word_group_chunks = {}
        for phrase in self.phrases:
            word_group_chunks = question_phrase.word_group_chunks_matching_sentence(
                phrase
            )
            phrases_and_word_group_chunks[phrase] = word_group_chunks
        return phrases_and_word_group_chunks

    # def key_words_for_phrase(self, phrase):
    #    return phrase.key_words_in_context(self)


class AnnabellAnswerContext(AnnabellAbstractContext):
    def __init__(
        self,
        text,
    ):
        super().__init__(text, max_words_per_phrase=None)
        self.answer_words_remaining = self.words().copy()

    def get_answer_type(self):
        if len(self.text.split()) > self.max_words_per_word_group():
            return LongAnswerType()
        else:
            return ShortAnswerType()

    def phrase_class(self):
        return AnnabellAnswerPhrase

    def word_group_chunks_matching_sentence(self, sentence):
        word_group_chunks = []
        for phrase in self.phrases:
            word_group_chunks.extend(
                phrase.word_group_chunks_matching_sentence(sentence)
            )
        return word_group_chunks

    def phrases_in_context(self):
        phrases = []
        phrase = self.phrase_class()(self.text, self)
        phrases.append(phrase)
        return phrases
