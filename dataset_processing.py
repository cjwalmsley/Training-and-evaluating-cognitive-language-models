from collections import Counter
from unidecode import unidecode
import logging
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset, load_from_disk
from scipy.spatial.distance import cosine
import ollama
import pandas as pd
from config.global_config import GlobalConfig
import shutil
import re
import spacy
from spacy.cli import download

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


def load_squad_dataset(ds_filepath):
    """
    Loads the SQuAD dataset from a local file, or downloads it if it's not
    found.
    :param ds_filepath: The path to the dataset.
    :return: The SQuAD dataset.
    """
    try:
        logger.info("Loading dataset from: " + ds_filepath)
        ds = load_from_disk(ds_filepath)
    except FileNotFoundError:
        logger.warning(
            f"Dataset not found or invalid at {ds_filepath}. "
            "Attempting to download from Hugging Face."
        )
        try:
            # Remove the potentially corrupted directory before downloading
            shutil.rmtree(ds_filepath, ignore_errors=True)
            ds = load_dataset("squad")
            ds.save_to_disk(ds_filepath)
        except Exception as e:
            logger.error(f"Failed to download or save the dataset: {e}")
            raise RuntimeError(
                "Could not load or download the SQuAD dataset. "
                "Please check your network connection and disk permissions."
            ) from e

    return ds


def items_with_title(the_dataset, the_title):
    df = pd.DataFrame(the_dataset)
    return df[df.apply(lambda x: x["title"] == the_title, axis=1)]


def filter_dataset_split(the_dataset_split, title, number_of_sentences, the_id=None):

    if title != "all" and title not in titles_in_dataset_split(the_dataset_split):
        logger.critical(f"Title '{title}' not found in dataset split.")
        raise Exception(f"Title '{title}' not found in dataset split.")

    if the_id is not None:
        filtered_database_split = the_dataset_split.filter(lambda x: x["id"] == the_id)
    elif title != "all":
        filtered_database_split = the_dataset_split.filter(
            lambda x: x["title"] == title
        )

    else:
        filtered_database_split = the_dataset_split

    if number_of_sentences == "all":
        pass
    else:
        filtered_database_split = filtered_database_split.select(
            range(min(number_of_sentences, len(the_dataset_split)))
        )
    # Create the 'answer' column from the 'answers' dictionary
    # make a dataframe from the dataset split
    return filtered_database_split


def save_squad_dataset(ds, save_filepath):
    ds.save_to_disk(save_filepath)


class AnnabellCommandGenerator:

    # creates the set of commands required to train annabell for a single training sample
    def __init__(self, sample_id, declarative_sentence, question, answer, max_words=10):
        self.sample_id = sample_id
        self.declarative_sentence = declarative_sentence
        self.question = question
        self.answer = answer
        self.max_words = max_words
        self.commands = []

    @staticmethod
    def blank_line():
        return "\n"

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

    @staticmethod
    def phrases_and_answer_words(phrases, answer_words):
        # construct a dictionary that contains each phrase as the key and the list of words from the answer that are in that phrase as the value
        phrase_answer_words = {}
        for phrase in phrases:
            phrase_answer_words[phrase] = []
            for word in answer_words:
                if word in phrase:
                    phrase_answer_words[phrase].append(word)
        return phrase_answer_words

    def question_word_length(self):
        return len(self.question.split())

    def sentence_word_length(self):
        return len(self.declarative_sentence.split())

    def answer_word_length(self):
        return len(self.answer.split())

    def write_question(self):
        """#if the length of the question is greater than max_words that the model can process in a phrase, the question must be split into 2 or more phrases.
        for example if max_words = 10 ,  the question:
        ? what was the trade -ing post that precede -d New-York-City call -ed
        should be split into the following commands
        ? what was the trade -ing post that precede -d
        New-York-City call -ed
        .sctx ? what was the trade -ing post that precede -d
        .wg trade
        .wg post
        .wg precede
        .sctx New-York-City call -ed
        .wg New-York-City
        .wg call

        """
        if self.question_word_length() <= self.max_words:
            self.commands.append(self.question)
        else:
            # split the question into phrases of max_words length
            for phrase in self.phrases_in_context(self.question):
                self.commands.append(phrase)

    def write_question_commands(self):

        question_words = self.question.split()
        if len(question_words) <= self.max_words:
            self.write_question_commands_for_phrase(self.question)
        else:
            self.write_question_commands_for_context(self.question)

    def write_question_commands_for_phrase(self, phrase):
        key_words = self.remove_stopwords(phrase)
        key_words = self.remove_suffixes(key_words)
        key_words = self.remove_question_mark(key_words)
        for word in key_words.split():
            self.commands.append(f".wg {word}")

    def write_question_commands_for_context(self, context):
        # split the context into phrases of max_words length
        for phrase in self.phrases_in_context(context):
            self.commands.append(f".sctx {phrase}")
            self.write_question_commands_for_phrase(phrase)

    def phrases_in_context(self, context):
        phrases = []
        context_words = context.split()
        number_of_phrases = (len(context_words) + self.max_words - 1) // self.max_words
        for i in range(number_of_phrases):
            phrase_words = context_words[i * self.max_words : (i + 1) * self.max_words]
            phrase = " ".join(phrase_words)
            phrases.append(phrase)
        return phrases

    def write_declarative_sentence(self):

        for phrase in self.phrases_in_context(self.declarative_sentence):
            self.commands.append(phrase)

    def write_answer_commands(self):
        if self.sentence_word_length() <= self.max_words:
            self.write_short_answer_commands()
        else:
            self.write_long_answer_commands()

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
        answer_words = self.answer.split()
        phrases = self.phrases_in_context(self.declarative_sentence)
        # construct a dictionary that contains each phrase as the key and the list of words from the answer that are in that phrase as the value
        phrase_answer_words = self.phrases_and_answer_words(phrases, answer_words)
        # for each phrase in the declarative sentence, write the phrase command and the answer words
        for phrase, answer_words in phrase_answer_words.items():
            if len(answer_words) == 0:
                continue
            else:
                self.commands.append(f".ph {phrase}")
                if len(answer_words) < 4:
                    self.commands.append(f".wg {" ".join(answer_words)}")
                else:
                    self.commands.append(f".wg {" ".join(answer_words[:3])}")
                    self.commands.append(".prw")
                    self.commands.append(f".wg {" ".join(answer_words[3:])}")
                    self.commands.append(".prw")

    def create_list_of_commands(self):

        self.commands = []

        self.commands.append("#id: " + str(self.sample_id))
        self.write_declarative_sentence()
        # add a blank line to terminate the context
        self.commands.append(self.blank_line())
        self.write_question()
        self.write_question_commands()
        self.write_answer_commands()
        # add a blank line to terminate the context
        self.commands.append(self.blank_line())
        return self.commands


class DatasetPreProcessor:

    def __init__(
        self,
        dataset,
        max_words_limit=25,
        max_word_length_limit=50,
        columns_to_process=None,
    ):

        if columns_to_process is None:
            columns_to_process = [
                "declarative_statement",
                "question",
                "answer",
            ]
        self.dataset = dataset
        self.columns_to_process = columns_to_process
        self.max_words_limit = max_words_limit
        self.max_word_length_limit = max_word_length_limit

    def preprocess_data(self):
        self.convert_answers_to_answer()
        self.format_columns()
        self.join_entity_words()
        self.filter_dataset_by_limits()
        self.dataset.reset_index(drop=True, inplace=True)

    def convert_answers_to_answer(self):
        # Convert the 'answers' column from a dictionary to a single string in a new 'answer' column
        if "answers" in self.dataset.columns:
            self.dataset["answer"] = self.dataset["answers"].apply(
                lambda x: x["text"][0] if x["text"] else ""
            )
            self.dataset.drop(columns=["answers"], inplace=True)

    def format_columns(self):
        for column in self.columns_to_process:
            is_question = column == "question"
            self.dataset[column + self.formatted_column_suffix()] = self.dataset[
                column
            ].apply(lambda x: self.format_text(x, is_question=is_question))

    def format_text(self, text, is_question=False):
        # apply the formatting rules to the text
        text = self.remove_whitespace(text)
        text = self.convert_first_character_to_lower_case_if_stopword(text)
        text = self.remove_full_stop(text)
        text = self.convert_decimal_point_to_word(text)
        text = self.replace_percent(text)
        text = self.replace_dont_apostrophe(text)
        text = self.remove_accents(text)
        if is_question:
            text = self.move_question_mark_to_start(text)
        text = self.remove_special_characters(text)
        return text

    @staticmethod
    def convert_decimal_point_to_word(a_string):
        # Replace "." with 'point' if it is part of a number.
        # Regex to find numbers with a decimal point:
        # \d+\.\d+  matches numbers like 1.23
        # \d+\.     matches numbers like 1. (dot at the end after digits)
        # \.\d+     matches numbers like .5 (dot at the beginning before digits)
        # The order matters to match \d+\.\d+ before \d+\. or \.\d+ for overlapping cases.
        pattern = r"\d+\.\d+|\d+\.|\.\d+"
        return re.sub(pattern, replace_decimal_in_matched_string, a_string)

    @staticmethod
    def remove_full_stop(a_string):
        # if the last character of a sentence is "." remove it
        if a_string.strip().endswith("."):
            return a_string[:-1]
        else:
            return a_string

    @staticmethod
    def replace_percent(a_string):
        cleaned_text = re.sub(r"%", " percent", a_string)
        return cleaned_text

    @staticmethod
    def replace_dont_apostrophe(a_string):
        cleaned_text = re.sub(r"don't", "do not", a_string)
        return cleaned_text

    @staticmethod
    def remove_accents(text):
        # Convert accented characters to unaccented ones
        text_unaccented = unidecode(text)
        return text_unaccented

    # remove all special characters except question marks and hyphen form the statements
    @staticmethod
    def remove_special_characters(text):
        """
        Removes special characters from a string, keeping alphanumeric characters and spaces.
        """
        # Keep only alphanumeric characters and spaces
        cleaned_text = re.sub(r"[^A-Za-z0-9\s?-]+", "", text)
        return cleaned_text

    @staticmethod
    def remove_whitespace(text):
        # Strip whitespace from the text
        return text.strip()

    def remove_whitespace_from_dataframe(self):
        # Strip whitespace from all string columns
        for col in self.dataset.select_dtypes(include=["object"]):
            self.dataset[col] = self.dataset[col].str.strip()

    def formatted_columns_to_process(self):
        return [col + self.formatted_column_suffix() for col in self.columns_to_process]

    @staticmethod
    def formatted_column_suffix():
        return "_formatted"

    @staticmethod
    def spacy_entities_md(text):
        # Extract entities using spaCy medium model
        model_name = "en_core_web_md"
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model '{model_name}'...")
            download(model_name)
            nlp = spacy.load(model_name)

        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def entity_names_in_text(self, text):
        entities = self.spacy_entities_md(text)
        entity_names = [entity_name for entity_name, entity_type in entities]
        return entity_names

    def sentences_in_row(self, a_row):
        sentences = [a_row[column] for column in self.formatted_columns_to_process()]
        return sentences

    def entity_names_in_row(self, a_row):
        sentences = self.sentences_in_row(a_row)
        entity_names = [self.entity_names_in_text(sentence) for sentence in sentences]
        # flatten the list of lists
        flattened_entity_names = [item for sublist in entity_names for item in sublist]
        entity_names_set = set(flattened_entity_names)
        return entity_names_set

    # python
    def join_entity_names_in_row(self, a_row):
        entities_in_row = self.entity_names_in_row(a_row)
        for column in self.formatted_columns_to_process():
            a_row[column] = self.replace_entities(a_row[column], entities_in_row)
        return a_row

    @staticmethod
    def replace_entities(text, entities):
        if not entities:
            return text
        for entity_name in entities:
            # replace only whole-entity occurrences to avoid partial matches
            joined_entity = "_".join(entity_name.split())
            pattern = r"\b" + re.escape(entity_name) + r"\b"
            text = re.sub(pattern, joined_entity, text)
        return text

    def join_entity_words(self):
        # apply to every row and assign back the formatted columns so changes persist
        updated = self.dataset.apply(self.join_entity_names_in_row, axis=1)
        self.dataset[self.formatted_columns_to_process()] = updated[
            self.formatted_columns_to_process()
        ]

    def join_concurrent_capitalized_words(self):
        # for the following columns in the dataframe, "response_declarative_sentence_formatted" and "response_question_formatted", "response_answer_formatted", identify any concurrent words that begin with capital letters and join them together with a hyphen.
        for column in self.formatted_columns_to_process():
            self.dataset[column] = self.dataset[column].str.replace(
                r"(\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b)",
                lambda m: "_".join(m.group(0).split()),
                regex=True,
            )

    @staticmethod
    def convert_stopwords_to_lower_case(a_string):
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        words = a_string.split()
        # Convert stopwords to lowercase
        lower_case_stopwords = [word.lower() for word in stop_words]
        # Replace stopwords in the string with their lowercase versions
        cleaned_string = " ".join(
            [
                word if word.lower() not in lower_case_stopwords else word.lower()
                for word in words
            ]
        )
        return cleaned_string

    @staticmethod
    def convert_first_character_to_lower_case_if_stopword(a_string):
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        words = a_string.split()
        if words and words[0].lower() in stop_words:
            words[0] = words[0].lower()
        cleaned_string = " ".join(words)
        return cleaned_string

    @staticmethod
    def remove_quotes(text):
        # remove " form the text
        cleaned_text = text.replace('"', "")
        return cleaned_text

    def remove_quotes_from_file(self, filepath):
        with open(filepath, "r") as file:
            train_data = file.readlines()
            train_data_cleaned = [self.remove_quotes(line) for line in train_data]

        cleaned_filepath = filepath.replace(".tsv", "_cleaned.tsv")
        with open(cleaned_filepath, "w") as cleaned_file:
            cleaned_file.writelines(train_data_cleaned)
        logger.info(f"Cleaned data saved to {cleaned_filepath}")
        return cleaned_filepath

    @staticmethod
    def move_question_mark_to_start(question, add_if_missing=True):
        # move the ? from the end of each question to the start
        if question.strip().endswith("?"):
            edited_question = "? " + question[:-1]
        else:
            if add_if_missing:
                edited_question = "? " + question
            else:
                # raise an exception if the question does not end with a ?
                raise ValueError(
                    f"Question does not end with a question mark: {question}"
                )

        return edited_question

    def filter_dataset_by_limits(self):
        self.filter_dataset_by_word_count()
        self.filter_dataset_by_word_length()

    def filter_dataset_by_word_count(self):
        for column_name in self.formatted_columns_to_process():

            # Filter out rows where the sentence exceeds the maximum number of words
            word_count_mask = (
                self.dataset[column_name].str.split().str.len() > self.max_words_limit
            )

            indices_to_drop = self.dataset[word_count_mask].index
            self.dataset.drop(indices_to_drop, inplace=True)

    def filter_dataset_by_word_length(self):
        for column_name in self.formatted_columns_to_process():

            # Filter out rows where any word exceeds the maximum length
            word_length_mask = self.dataset[column_name].apply(
                lambda sentence: all(
                    len(word) > self.max_word_length_limit for word in sentence.split()
                )
            )

            indices_to_drop = self.dataset[word_length_mask].index
            self.dataset.drop(indices_to_drop, inplace=True)

    @staticmethod
    def statement_category_column_name():
        return "statement_category"

    @staticmethod
    def question_category_column_name():
        return "question_category"

    def question_categories(self):
        return self.dataset[self.question_category_column_name()].unique()

    def statement_categories(self):
        return self.dataset[self.statement_category_column_name()].unique()

    def select_pretraining_data(self, percentage_of_pretraining_samples):
        # pick a random sample of pretraining rows or use a pre-selected, manually generated set
        # for each category, pick an equal number of samples

        self.dataset["is_pretraining"] = False
        number_of_pretraining_samples = int(
            len(self.dataset) * percentage_of_pretraining_samples // 100
        )
        logger.info(f"Number of pretraining samples: {number_of_pretraining_samples}")
        samples_per_category = int(
            number_of_pretraining_samples
            // (len(self.question_categories()) + len(self.statement_categories()))
        )
        logger.info(f"Samples per category: {samples_per_category}")
        # sample from the question categories
        # sample from the question categories
        for category in self.question_categories():
            category_df = self.dataset[
                self.dataset[self.question_category_column_name()] == category
            ]
            samples_to_take = samples_per_category
            if len(category_df) < samples_per_category:
                logger.warning(
                    f"Warning: Not enough samples in question category '{category}'. Taking all {len(category_df)} samples."
                )
                samples_to_take = len(category_df)
            if samples_to_take > 0:
                sampled_category_df = category_df.sample(
                    n=samples_to_take, random_state=42
                )
                self.dataset.loc[sampled_category_df.index, "is_pretraining"] = True

        # sample from the sentence categories starting with those that are already selected for pretraining
        for category in self.statement_categories():
            category_df = self.dataset[
                self.dataset[self.statement_category_column_name()] == category
            ]
            already_selected_df = category_df[category_df["is_pretraining"] == True]
            already_selected_count = len(already_selected_df)
            remaining_samples = samples_per_category - already_selected_count
            if remaining_samples > 0:
                not_selected_df = category_df[category_df["is_pretraining"] == False]
                samples_to_take = remaining_samples
                if len(not_selected_df) < remaining_samples:
                    logger.warning(
                        f"Warning: Not enough samples in sentence category '{category}'. Taking all {len(not_selected_df)} available samples."
                    )
                    samples_to_take = len(not_selected_df)
                if samples_to_take > 0:
                    sampled_category_df = not_selected_df.sample(
                        n=samples_to_take, random_state=42
                    )
                    self.dataset.loc[sampled_category_df.index, "is_pretraining"] = True

        # print the counts of samples in the question and sentence categories
        logger.info("Pretraining samples by question category:")
        logger.info(
            self.dataset[self.dataset["is_pretraining"] == True][
                self.question_category_column_name()
            ].value_counts()
        )
        logger.info("Pretraining samples by sentence category:")
        logger.info(
            self.dataset[self.dataset["is_pretraining"] == True][
                self.statement_category_column_name()
            ].value_counts()
        )
        total_pretraining_count = self.dataset["is_pretraining"].sum()
        logger.info(
            f"Total number of samples selected for pretraining: {total_pretraining_count}"
        )

        return self.dataset

    def merge_categories(
        self, categorised_questions_filepath, categorised_sentences_filepath
    ):
        # add categories to the questions and declarative sentences, creating 2 new columns - question category and sentence category
        categorised_questions_df = pd.read_json(
            categorised_questions_filepath, lines=True
        )
        categorised_questions_df = categorised_questions_df.rename(
            columns={"category": self.question_category_column_name()}
        )
        self.dataset = self.dataset.merge(
            categorised_questions_df[["id", self.question_category_column_name()]],
            on="id",
            how="left",
        )
        categorised_sentences_df = pd.read_json(
            categorised_sentences_filepath, lines=True
        )
        categorised_sentences_df = categorised_sentences_df.rename(
            columns={"category": self.statement_category_column_name()}
        )
        categorised_sentences_df[self.statement_category_column_name()].value_counts()
        self.dataset = self.dataset.merge(
            categorised_sentences_df[["id", self.statement_category_column_name()]],
            on="id",
            how="left",
        )
        return self.dataset


def write_pretraining_file(the_filepath, the_df):
    with open(the_filepath, "w") as commands_file:
        for index, row in the_df.iterrows():
            commands = row["created_commands"]
            for command in commands:
                commands_file.write(command + "\n")
    logger.info(f"Wrote {the_filepath}")

    with open(the_filepath, "r") as commands_file:
        lines = commands_file.readlines()
    number_of_reward_lines = sum(1 for line in lines if line.startswith(".rw"))
    logger.info(f"Number of reward lines: {number_of_reward_lines}")
    logger.info(f"Number of commands: {len(lines)}")


def write_training_file(the_filepath, the_df):
    list_of_training_tuples = list(
        zip(
            the_df["id"],
            the_df["response_declarative_sentence_formatted"],
        )
    )

    with open(the_filepath, "w") as file:
        for each_tuple in list_of_training_tuples:
            file.write(f"#id: {each_tuple[0]}\n")
            file.write(f"{each_tuple[-1]}\n")
            # write a blank line to signal to ANNABELL the end of the context
            file.write("\n")
    logger.info(f"file written: {the_filepath}")

    with open(the_filepath, "r") as commands_file:
        lines = commands_file.readlines()
        logger.info(f"Number of commands: {len(lines)}")


def write_testing_file(the_filepath, the_df):
    list_of_testing_tuples = list(
        zip(
            the_df["id"],
            the_df["response_question_formatted"],
        )
    )

    with open(the_filepath, "w") as test_file:
        for each_tuple in list_of_testing_tuples:
            test_file.write(f"#id: {each_tuple[0]}\n")
            test_file.write(f"{each_tuple[-1]}\n.x\n")
            test_file.write("#END OF TESTING SAMPLE\n")
            # write a blank line to signal to ANNABELL the end of the context
            test_file.write("\n")
    logger.info(f"file written: {the_filepath}")

    with open(the_filepath, "r") as commands_file:
        lines = commands_file.readlines()
        logger.info(f"Number of commands: {len(lines)}")


def any_word_match(row):
    # if the row contains a non-string value return False
    if not isinstance(row["test_answer"], str) or not isinstance(
        row["response_answer_formatted"], str
    ):
        return False
    # return True if any word in test_answer is also in response_answer_formatted
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    test_words = set(row["test_answer"].split())
    response_words = set(row["response_answer_formatted"].split())
    intersecting_words = test_words.intersection(response_words)
    open_class_intersecting_words = intersecting_words - stop_words

    return open_class_intersecting_words != set()


def similarity_score(sentence_1, sentence_2):
    # compare two sentences and return a similarity score based on their embedding representation
    import ollama
    from scipy.spatial.distance import cosine, euclidean

    embedding_1 = ollama.embed(
        model=global_config.embedding_model,
        input=sentence_1,
    ).embeddings[0]
    embedding_2 = ollama.embed(
        model=global_config.embedding_model,
        input=sentence_2,
    ).embeddings[0]

    cosine_dist_12 = cosine(embedding_1, embedding_2)
    euclidean_dist_12 = euclidean(embedding_1, embedding_2)

    results_dict = {
        "cosine_distance": cosine_dist_12,
        "euclidean_distance": euclidean_dist_12,
    }
    return results_dict


def replace_decimal_in_matched_string(matched_string):
    # callable function to support the regex
    number_str = matched_string.group(0)
    return number_str.replace(".", " point ")


def truncate_to_max_words(text, max_words=20):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    else:
        return text


def filter_by_max_words(the_df, max_words=10):
    # returns a new dataframe filtered such that each question, answer and statement has less than 11 words

    filtered_df = the_df[
        the_df.apply(
            lambda row: len(row["question"].split()) <= max_words
            and len(row["answer"][0].split()) <= max_words
            and len(row["statement"].split()) <= max_words,
            axis=1,
        )
    ]
    return filtered_df


# produce a summary of a dataset by splits
def dataset_summary(a_dataset):
    for split in a_dataset.keys():
        ds_split = a_dataset[split].to_pandas()
        logger.info("summary of " + split + " split")
        logger.info(ds_split.info())
        titles = ds_split["title"]
        logger.info("number of titles: " + str(len(set(titles))))
        logger.info((set(titles)))
        bag_of_titles = Counter(titles)
        logger.info(
            "titles with most numerous examples: "
            + str((bag_of_titles.most_common(20)))
            + "\n"
        )


def titles_in_dataset_split(a_dataset_split):
    ds_split = a_dataset_split.to_pandas()
    titles = ds_split["title"]
    return set(titles)


def ids_questions_answers_from_log_file(test_log_filepath):
    with open(test_log_filepath, "r") as test_log_file:
        test_log_lines = test_log_file.readlines()

    # parse the line with hte id and extract the id number then parse the line with the question and extract the question then parse the line with the END OF TESTING SAMPLE and extract the content of the previous line

    ids_questions_answers = []
    id_number, question, answer = None, None, None
    for index, line in enumerate(test_log_lines):
        if line.startswith("#id:"):
            id_number = line.strip().split(":")[-1].strip()
        elif line.startswith("?"):
            question = line.strip()
        elif line.startswith("#END OF TESTING SAMPLE"):
            previous_line = test_log_lines[index - 1].strip()
            answer = previous_line
            if id_number and question:
                ids_questions_answers.append((id_number, question, answer))
            id_number, question, answer = None, None, None
        else:
            continue

    return ids_questions_answers


def embedding_for_sentence(a_string):
    embedding = ollama.embed(
        model=global_config.embedding_model,
        input=a_string,
    ).embeddings[0]
    return embedding


def cosine_distance(a_row):
    return cosine(
        a_row["test_answer_embedding"], a_row["response_answer_formatted_embedding"]
    )