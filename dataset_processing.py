from collections import Counter
from unidecode import unidecode
import logging
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset, load_from_disk
from scipy.spatial.distance import cosine
import ollama
import pandas as pd
from annabell_utilities import AnnabellLogfileInterpreter
from config.global_config import GlobalConfig
import shutil
import re
import spacy
from spacy.cli import download
import numpy as np
from commands import (
    AnnabellBaseCommandGenerator,
    AnnabellTestingCommandGenerator,
    AnnabellTrainingCommandGenerator,
)

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


class DatasetPreProcessor:

    def __init__(
        self,
        dataset,
        max_words_limit=global_config.maximum_number_of_words(),
        max_word_length_limit=global_config.maximum_word_length(),
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
        self.nlp = self._load_spacy_model("en_core_web_md")

    @staticmethod
    def _load_spacy_model(model_name):
        """Loads a spaCy model, downloading it if necessary."""
        try:
            return spacy.load(model_name)
        except OSError:
            logger.info(f"Downloading spaCy model '{model_name}'...")
            download(model_name)
            return spacy.load(model_name)

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
        text = self.remove_special_characters(text)
        if is_question:
            text = self.add_question_mark_to_start(text)
        text = self.lemmatize_text(text)
        return text

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        lemmatized_sentence = " ".join([token.lemma_ for token in doc])
        return lemmatized_sentence

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
        cleaned_text = re.sub(r"[^A-Za-z0-9\s-]+", "", text)
        return cleaned_text.strip()

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

    def spacy_entities_md(self, text):
        # Extract entities using the preloaded spaCy medium model
        doc = self.nlp(text)
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
    def remove_the_from_start_of_text(the_text):
        words = the_text.split("_")
        if words[0] == "the":
            return "the " + "_".join(words[1:])
        else:
            return the_text

    @staticmethod
    def replace_entities(text, entities):
        if not entities:
            return text

        # Sort entities by length (longest first) to handle nested entities correctly.
        # For example, replace "New York City" before "New York".
        sorted_entities = sorted(entities, key=len, reverse=True)

        for entity_name in sorted_entities:
            words = entity_name.split()
            replacement = ""
            # Check if the entity starts with "the" and has more than one word.
            if len(words) > 1 and words[0].lower() == "the":
                replacement = "the " + "_".join(words[1:])
            else:
                replacement = "_".join(words)

            # Use word boundaries (\b) to ensure only whole words are replaced.
            pattern = r"\b" + re.escape(entity_name) + r"\b"
            text = re.sub(pattern, replacement, text)

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
    def add_question_mark_to_start(question):
        return "? " + question

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
                lambda sentence: any(
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
        self.dataset["is_pretraining"] = False
        num_total_samples = len(self.dataset)
        num_pretraining_samples = int(
            num_total_samples * percentage_of_pretraining_samples / 100
        )

        if num_pretraining_samples == 0:
            logger.warning("Requested 0 pre-training samples. Nothing to do.")
            return None

        if num_pretraining_samples > num_total_samples:
            logger.warning(
                f"Requested {num_pretraining_samples} samples, but dataset only has {num_total_samples}. Selecting all samples."
            )
            self.dataset["is_pretraining"] = True
            return None

        logger.info(
            f"Attempting to select {num_pretraining_samples} pre-training samples."
        )

        # Consolidate all categories and their indices (handle missing columns and NaNs)
        q_col = self.question_category_column_name()
        s_col = self.statement_category_column_name()
        q_exists = q_col in self.dataset.columns
        s_exists = s_col in self.dataset.columns

        q_cats = (
            []
            if not q_exists
            else [c for c in self.dataset[q_col].dropna().unique().tolist()]
        )
        s_cats = (
            []
            if not s_exists
            else [c for c in self.dataset[s_col].dropna().unique().tolist()]
        )
        all_categories = q_cats + s_cats

        if len(all_categories) == 0:
            logger.warning(
                "No categories found for pre-training selection. Performing random sampling."
            )
            pretraining_indices = np.random.choice(
                self.dataset.index, num_pretraining_samples, replace=False
            )
            self.dataset.loc[pretraining_indices, "is_pretraining"] = True
            return None

        category_info = {}
        q_cat_col = q_col
        s_cat_col = s_col

        for category in q_cats:
            indices = self.dataset[self.dataset[q_cat_col] == category].index
            category_info[category] = {
                "indices": indices,
                "count": len(indices),
                "to_select": 0,
            }

        for category in s_cats:
            indices = self.dataset[self.dataset[s_cat_col] == category].index
            # If a statement category is also a question category, merge them.
            if category in category_info:
                category_info[category]["indices"] = category_info[category][
                    "indices"
                ].union(indices)
                category_info[category]["count"] = len(
                    category_info[category]["indices"]
                )
            else:
                category_info[category] = {
                    "indices": indices,
                    "count": len(indices),
                    "to_select": 0,
                }

        unique_categories = list(category_info.keys())

        # Initial distribution
        base_samples_per_category = (
            num_pretraining_samples // len(unique_categories)
            if unique_categories
            else 0
        )

        deficit = 0
        for cat in unique_categories:
            take = min(base_samples_per_category, category_info[cat]["count"])
            category_info[cat]["to_select"] = take
            deficit += base_samples_per_category - take

        # Account for remainder from initial division
        remainder = (
            num_pretraining_samples % len(unique_categories) if unique_categories else 0
        )
        deficit += remainder

        # Redistribute the deficit to categories with spare capacity
        if deficit > 0:
            # Sort by categories with the most remaining samples
            sorted_cats = sorted(
                unique_categories,
                key=lambda c: category_info[c]["count"] - category_info[c]["to_select"],
                reverse=True,
            )

            for cat in sorted_cats:
                if deficit == 0:
                    break
                available_extra = (
                    category_info[cat]["count"] - category_info[cat]["to_select"]
                )
                take_extra = min(deficit, available_extra)
                category_info[cat]["to_select"] += take_extra
                deficit -= take_extra

        # Final sampling
        pretraining_indices = []
        for cat in unique_categories:
            num_to_select = category_info[cat]["to_select"]
            if num_to_select > 0:
                selected = np.random.choice(
                    category_info[cat]["indices"], num_to_select, replace=False
                )
                pretraining_indices.extend(selected)

        self.dataset.loc[pretraining_indices, "is_pretraining"] = True

        # Ensure we have exactly the requested number of unique samples
        actual_pretraining_size = self.dataset["is_pretraining"].sum()
        shortfall = num_pretraining_samples - actual_pretraining_size

        if shortfall > 0:
            logger.info(
                f"Selection shortfall of {shortfall} samples (due to overlap or scarcity). Filling with random remaining samples."
            )
            df_not_selected = self.dataset[~self.dataset["is_pretraining"]]
            if len(df_not_selected) >= shortfall:
                extra_indices = np.random.choice(
                    df_not_selected.index, shortfall, replace=False
                )
                self.dataset.loc[extra_indices, "is_pretraining"] = True
            else:
                # This case implies available samples < requested, which should be caught early,
                # but for safety we take all remaining.
                self.dataset.loc[df_not_selected.index, "is_pretraining"] = True

        actual_pretraining_size = self.dataset["is_pretraining"].sum()
        logger.info(f"Selected {actual_pretraining_size} samples for pre-training.")

        if actual_pretraining_size != num_pretraining_samples:
            logger.error(
                f"CRITICAL: Sample selection failed. Requested: {num_pretraining_samples}, Got: {actual_pretraining_size}"
            )

        logger.info("Pretraining samples by question category:")
        logger.info(
            self.dataset[self.dataset["is_pretraining"] == True][
                q_cat_col
            ].value_counts()
        )
        logger.info("Pretraining samples by sentence category:")
        logger.info(
            self.dataset[self.dataset["is_pretraining"] == True][
                s_cat_col
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

    @staticmethod
    def is_pretraining_column_name():
        return "is_pretraining"

    @staticmethod
    def created_commands_error_column_name():
        return "created_commands_error"

    def pretraining_dataset(self):
        filter_mask = self.dataset[self.is_pretraining_column_name()] == True
        filter_mask = filter_mask & (
            self.dataset[self.created_commands_error_column_name()] != True
        )

        return self.dataset[filter_mask]

    def total_pretraining_samples(self):
        return self.dataset[self.is_pretraining_column_name()].sum()

    def training_dataset(self):
        return self.dataset[self.dataset[self.is_pretraining_column_name()] == False]

    def testing_dataset(self):
        return self.training_dataset()

    def pre_training_validation_dataset(self):
        return self.pretraining_dataset()

    @staticmethod
    def created_commands_column_name():
        return "created_commands"

    @staticmethod
    def declarative_statement_formatted_column_name():
        return "declarative_statement_formatted"

    @staticmethod
    def question_formatted_column_name():
        return "question_formatted"

    @staticmethod
    def answer_formatted_column_name():
        return "answer_formatted"

    @staticmethod
    def auto_save_weights_command():
        return ".auto_save_links"

    @staticmethod
    def save_weights_command():
        return ".save"

    def create_commands_for_pretraining(self):
        # if the pretraining column is true create the commands
        # add a new column to the dataframe with the created list of commands
        def generate_commands(row):
            generator = AnnabellBaseCommandGenerator(
                row[self.id_column_name()],
                row[self.declarative_statement_formatted_column_name()],
                row[self.question_formatted_column_name()],
                row[self.answer_formatted_column_name()],
                row[self.is_pretraining_column_name()],
            )
            commands = generator.create_list_of_commands()
            has_error = (
                AnnabellBaseCommandGenerator.error_generating_pretraining_command()
                in commands
            )
            return pd.Series(
                {
                    self.created_commands_column_name(): commands,
                    "created_commands_error": has_error,
                }
            )

        generated_data = self.dataset.apply(generate_commands, axis=1)
        self.dataset[self.created_commands_column_name()] = generated_data[
            self.created_commands_column_name()
        ]
        self.dataset["created_commands_error"] = generated_data[
            "created_commands_error"
        ]

    def write_pretraining_file(self, the_filepath, auto_save_weights):
        with open(the_filepath, "w") as commands_file:
            dataset_to_write = self.pretraining_dataset()
            # Only filter if the column exists
            if "created_commands_error" in dataset_to_write.columns:
                dataset_to_write = dataset_to_write[
                    dataset_to_write["created_commands_error"] != True
                ]
            dataset_to_write.reset_index(drop=True, inplace=True)
            if auto_save_weights:
                logger.info(
                    "Auto-save weights is enabled; adding save weight commands to pre-training samples."
                )
                commands_file.write(f"{self.auto_save_weights_command()}\n")
            save_weights_counter = 0
            for index, row in dataset_to_write.iterrows():
                if save_weights_counter >= global_config.save_weights_every_n_steps():
                    commands_file.write(
                        f"{self.save_weights_command()} {global_config.pre_training_weights_filename}\n"
                    )
                    save_weights_counter = 0
                save_weights_counter += 1
                commands = row["created_commands"]
                commands_file.write(
                    f"{AnnabellLogfileInterpreter.start_of_sample_string()}\n"
                )
                commands_file.write(
                    f"{AnnabellLogfileInterpreter.sample_number_count_string()} {index+1} of {len(dataset_to_write)}\n"
                )
                for command in commands:

                    if command.endswith("\n"):
                        commands_file.write(command)
                    else:
                        commands_file.write(command + "\n")

        logger.info(f"Wrote {len(dataset_to_write)} samples to {the_filepath}")

    @staticmethod
    def declarative_statement_column_name():
        return "declarative_statement"

    @staticmethod
    def question_column_name():
        return "question"

    @staticmethod
    def answer_column_name():
        return "answer"

    @staticmethod
    def id_column_name():
        return "id"

    def write_training_file(self, the_filepath):
        list_of_training_tuples = list(
            zip(
                self.training_dataset()["id"],
                self.training_dataset()[
                    self.declarative_statement_column_name()
                    + self.formatted_column_suffix()
                ],
            )
        )

        with open(the_filepath, "w") as file:
            for each_tuple in list_of_training_tuples:
                sample_id = each_tuple[0]
                declarative_statement = each_tuple[-1]
                command_generator = AnnabellTrainingCommandGenerator(
                    sample_id, declarative_statement
                )
                command_generator.create_list_of_commands()
                for command in command_generator.commands:
                    file.write(f"{command}\n")
        logger.info(f"file written: {the_filepath}")

    def write_pretraining_testing_file(self, the_filepath):
        dataset_to_write = self.pretraining_dataset()
        dataset_to_write = dataset_to_write[
            dataset_to_write[self.created_commands_error_column_name()] != True
        ]

        self.write_testing_file_with_dataset(the_filepath, dataset_to_write)

    def write_testing_file(self, the_filepath):
        self.write_testing_file_with_dataset(the_filepath, self.training_dataset())

    def write_testing_file_with_dataset(self, the_filepath, the_dataset):

        list_of_testing_tuples = list(
            zip(
                the_dataset[self.id_column_name()],
                the_dataset[
                    self.question_column_name() + self.formatted_column_suffix()
                ],
            )
        )

        with open(the_filepath, "w") as test_file:
            for each_tuple in list_of_testing_tuples:
                sample_id = each_tuple[0]
                question = each_tuple[-1]
                command_generator = AnnabellTestingCommandGenerator(sample_id, question)
                command_generator.create_list_of_commands()
                for command in command_generator.commands:
                    test_file.write(f"{command}\n")

        logger.info(f"file written: {the_filepath}")


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
        model=global_config.embedding_model(),
        input=a_string,
    ).embeddings[0]
    return embedding


def cosine_distance(a_row):
    return cosine(
        a_row["test_answer_embedding"], a_row["response_answer_formatted_embedding"]
    )