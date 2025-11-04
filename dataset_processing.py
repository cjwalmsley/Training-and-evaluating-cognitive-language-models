import re
from collections import Counter
from unidecode import unidecode
import warnings
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset, load_from_disk
from scipy.spatial.distance import cosine
import ollama
import pandas as pd


def load_squad_dataset(ds_filepath="squad_dataset"):
    # check if dataset is available on disk, if not load it
    try:
        print("Loading dataset from: " + ds_filepath)
        ds = load_from_disk(ds_filepath)
    except FileNotFoundError:
        print(
            "File not found, loading dataset from Huggingface and saving to: "
            + ds_filepath
        )
        ds = load_dataset("rajpurkar/squad")
        save_squad_dataset(ds, ds_filepath)
    return ds


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

    def phrases_and_answer_words(self, phrases, answer_words):
        # construct a dictionary that contains each phrase as the key and the list of words from the answer that are in that phrase as the value
        phrase_answer_words = {}
        for phrase in phrases:
            phrase_answer_words[phrase] = []
            for word in answer_words:
                if word in phrase:
                    phrase_answer_words[phrase].append(word)
        return phrase_answer_words

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
        dataset_filepath,
        max_words_limit=25,
        max_word_length_limit=50,
        columns_to_process=None,
    ):

        if columns_to_process is None:
            columns_to_process = [
                "response_declarative_sentence_formatted",
                "response_question_formatted",
                "response_answer_formatted",
            ]
        self.dataset_filepath = dataset_filepath
        self.dataset = pd.read_json(dataset_filepath, lines=True)
        self.columns_to_process = columns_to_process
        self.max_words_limit = max_words_limit
        self.max_word_length_limit = max_word_length_limit

    def preprocess_data(self):
        self.remove_whitespace()
        self.join_concurrent_capitalized_words()
        self.filter_dataset_by_limits()
        self.dataset.reset_index(drop=True, inplace=True)

    def remove_whitespace(self):
        # Strip whitespace from all string columns
        for col in self.dataset.select_dtypes(include=["object"]):
            self.dataset[col] = self.dataset[col].str.strip()

    def join_concurrent_capitalized_words(self):
        # for the following columns in the dataframe, "response_declarative_sentence_formatted" and "response_question_formatted", "response_answer_formatted", identify any concurrent words that begin with capital letters and join them together with a hyphen.
        for column in self.columns_to_process:
            self.dataset[column] = self.dataset[column].str.replace(
                r"(\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b)",
                lambda m: "-".join(m.group(0).split()),
                regex=True,
            )

    def filter_dataset_by_limits(self):
        for column_name in self.columns_to_process:
            # Ensure the column is of string type, handling potential non-string data
            self.dataset[column_name] = self.dataset[column_name].astype(str)

            # Filter out rows where the sentence exceeds the maximum number of words
            word_count_mask = (
                self.dataset[column_name].str.split().str.len() <= self.max_words_limit
            )
            self.dataset = self.dataset[word_count_mask]

            # Filter out rows where any word exceeds the maximum length
            word_length_mask = self.dataset[column_name].apply(
                lambda sentence: all(
                    len(word) <= self.max_word_length_limit for word in sentence.split()
                )
            )
            self.dataset = self.dataset[word_length_mask]


def merge_categories(
    the_df, categorised_questions_filepath, categorised_sentences_filepath
):
    # add categories to the questions and declarative sentences, creating 2 new columns - question category and sentence category
    categorised_questions_df = pd.read_json(categorised_questions_filepath, lines=True)
    categorised_questions_df = categorised_questions_df.rename(
        columns={"category": "question_category"}
    )
    the_df = the_df.merge(
        categorised_questions_df[["id", "question_category"]], on="id", how="left"
    )
    categorised_sentences_df = pd.read_json(categorised_sentences_filepath, lines=True)
    categorised_sentences_df = categorised_sentences_df.rename(
        columns={"category": "sentence_category"}
    )
    categorised_sentences_df["sentence_category"].value_counts()
    the_df = the_df.merge(
        categorised_sentences_df[["id", "sentence_category"]], on="id", how="left"
    )
    return the_df


def select_pretraining_data(the_df, percentage_of_pretraining_samples):
    # pick a random sample of pretraining rows or use a pre-selected, manually generated set
    # for each category, pick an equal number of samples
    question_categories = the_df["question_category"].unique()
    sentence_categories = the_df["sentence_category"].unique()
    print(f"Question categories: {question_categories}")
    print(f"Sentence categories: {sentence_categories}")

    the_df["is_pretraining"] = False
    number_of_pretraining_samples = (
        len(the_df) * percentage_of_pretraining_samples // 100
    )
    print(f"Number of pretraining samples: {number_of_pretraining_samples}")
    samples_per_category = number_of_pretraining_samples // (
        len(question_categories) + len(sentence_categories)
    )
    print(f"Samples per category: {samples_per_category}")
    # sample from the question categories
    # sample from the question categories
    for category in question_categories:
        category_df = the_df[the_df["question_category"] == category]
        samples_to_take = samples_per_category
        if len(category_df) < samples_per_category:
            print(
                f"Warning: Not enough samples in question category '{category}'. Taking all {len(category_df)} samples."
            )
            samples_to_take = len(category_df)
        if samples_to_take > 0:
            sampled_category_df = category_df.sample(n=samples_to_take, random_state=42)
            the_df.loc[sampled_category_df.index, "is_pretraining"] = True

    # sample from the sentence categories starting with those that are already selected for pretraining
    for category in sentence_categories:
        category_df = the_df[the_df["sentence_category"] == category]
        already_selected_df = category_df[category_df["is_pretraining"] == True]
        already_selected_count = len(already_selected_df)
        remaining_samples = samples_per_category - already_selected_count
        if remaining_samples > 0:
            not_selected_df = category_df[category_df["is_pretraining"] == False]
            samples_to_take = remaining_samples
            if len(not_selected_df) < remaining_samples:
                print(
                    f"Warning: Not enough samples in sentence category '{category}'. Taking all {len(not_selected_df)} available samples."
                )
                samples_to_take = len(not_selected_df)
            if samples_to_take > 0:
                sampled_category_df = not_selected_df.sample(
                    n=samples_to_take, random_state=42
                )
                the_df.loc[sampled_category_df.index, "is_pretraining"] = True

    # print the counts of samples in the question and sentence categories
    print("Pretraining samples by question category:")
    print(the_df[the_df["is_pretraining"] == True]["question_category"].value_counts())
    print("Pretraining samples by sentence category:")
    print(the_df[the_df["is_pretraining"] == True]["sentence_category"].value_counts())
    total_pretraining_count = the_df["is_pretraining"].sum()
    print(
        f"Total number of samples selected for pretraining: {total_pretraining_count}"
    )

    return the_df


def write_pretraining_file(the_filepath, the_df):
    with open(the_filepath, "w") as commands_file:
        for index, row in the_df.iterrows():
            commands = row["created_commands"]
            for command in commands:
                commands_file.write(command + "\n")
    print(f"Wrote {the_filepath}")

    with open(the_filepath, "r") as commands_file:
        lines = commands_file.readlines()
    number_of_reward_lines = sum(1 for line in lines if line.startswith(".rw"))
    print(f"Number of reward lines: {number_of_reward_lines}")
    print(f"Number of commands: {len(lines)}")
    for line in lines[:20]:
        print(line.strip())


def write_training_file(the_filepath, the_df):
    list_of_training_tuples = list(
        zip(
            the_df["id"],
            the_df["response_declarative_sentence_formatted"],
        )
    )

    with open(the_filepath, "w") as file:
        for tuple in list_of_training_tuples:
            file.write(f"#id: {tuple[0]}\n")
            file.write(f"{tuple[-1]}\n")
            # write a blank line to signal to ANNABELL the end of the context
            file.write("\n")
    print(f"file written: {the_filepath}")

    with open(the_filepath, "r") as commands_file:
        lines = commands_file.readlines()
        print(f"Number of commands: {len(lines)}")
        print("First 20 lines:")
        for line in lines[:20]:
            print(line.strip())


def write_testing_file(the_filepath, the_df):
    list_of_testing_tuples = list(
        zip(
            the_df["id"],
            the_df["response_question_formatted"],
        )
    )

    with open(the_filepath, "w") as test_file:
        for tuple in list_of_testing_tuples:
            test_file.write(f"#id: {tuple[0]}\n")
            test_file.write(f"{tuple[-1]}\n.x\n")
            test_file.write("#END OF TESTING SAMPLE\n")
            # write a blank line to signal to ANNABELL the end of the context
            test_file.write("\n")
    print(f"file written: {the_filepath}")

    with open(the_filepath, "r") as commands_file:
        lines = commands_file.readlines()
        print(f"Number of commands: {len(lines)}")
        print("First 20 lines:")
        for line in lines[:20]:
            print(line.strip())


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


def remove_quotes(text):
    # remove " form the text
    cleaned_text = text.replace('"', "")
    return cleaned_text


def remove_quotes_from_file(filepath):
    with open(filepath, "r") as file:
        train_data = file.readlines()
        train_data_cleaned = [remove_quotes(line) for line in train_data]

    cleaned_filepath = filepath.replace(".tsv", "_cleaned.tsv")
    with open(cleaned_filepath, "w") as cleaned_file:
        cleaned_file.writelines(train_data_cleaned)
    print(f"Cleaned data saved to {cleaned_filepath}")
    return cleaned_filepath


# move the ? from the end of each question to the start
def move_question_mark_to_start(question, add_if_missing=True):
    if question.strip().endswith("?"):
        edited_question = "? " + question[:-1]
    else:
        if add_if_missing:
            edited_question = "? " + question
        else:
            # raise an exception if the question does not end with a ?
            raise ValueError(f"Question does not end with a question mark: {question}")

    return edited_question


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
        model="embeddinggemma",
        input=sentence_1,
    ).embeddings[0]
    embedding_2 = ollama.embed(
        model="embeddinggemma",
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


def convert_decimal_point_to_word(a_string):
    # Replace "." with 'point' if it is part of a number.
    # Regex to find numbers with a decimal point:
    # \d+\.\d+  matches numbers like 1.23
    # \d+\.     matches numbers like 1. (dot at the end after digits)
    # \.\d+     matches numbers like .5 (dot at the beginning before digits)
    # The order matters to match \d+\.\d+ before \d+\. or \.\d+ for overlapping cases.
    pattern = r"\d+\.\d+|\d+\.|\.\d+"
    return re.sub(pattern, replace_decimal_in_matched_string, a_string)


def remove_full_stop(a_string):
    # if the last character of a sentence is "." remove it
    if a_string.strip().endswith("."):
        return a_string[:-1]
    else:
        return a_string


def replace_percent(a_string):
    cleaned_text = re.sub(r"%", " percent", a_string)
    return cleaned_text


def replace_dont_apostrophe(a_string):
    cleaned_text = re.sub(r"don't", "do not", a_string)
    return cleaned_text


def remove_accents(text):
    # Convert accented characters to unaccented ones
    text_unaccented = unidecode(text)
    return text_unaccented


# remove all special characters except question marks and hyphen form the statements
def remove_special_characters(text):
    """
    Removes special characters from a string, keeping alphanumeric characters and spaces.
    """
    # Keep only alphanumeric characters and spaces
    cleaned_text = re.sub(r"[^A-Za-z0-9\s?-]+", "", text)
    return cleaned_text


def filter_by_max_words(the_df, max_words=10):
    # returnes a new dataframe filtered such that each question, answer and statement has less than 11 words

    filtered_df = the_df[
        the_df.apply(
            lambda row: len(row["question"].split()) <= max_words
            and len(row["answer"][0].split()) <= max_words
            and len(row["statement"].split()) <= max_words,
            axis=1,
        )
    ]
    return filtered_df


def clean_text(a_series, is_question):
    # takes a dataframe series and applies reformatting
    if is_question:
        a_series = a_series.apply(move_question_mark_to_start)
    for function_name in (
        remove_full_stop,
        convert_decimal_point_to_word,
        remove_accents,
        remove_special_characters,
    ):
        a_series = a_series.apply(function_name)
    return a_series


def format_text(text, is_question=False):
    # apply the formatting rules to the text
    text = convert_first_character_to_lower_case_if_stopword(text)
    text = remove_full_stop(text)
    text = convert_decimal_point_to_word(text)
    text = replace_percent(text)
    text = replace_dont_apostrophe(text)
    text = remove_accents(text)
    if is_question:
        text = move_question_mark_to_start(text)
    text = remove_special_characters(text)
    return text


# produce a summary of a dataset by splits
def dataset_summary(a_dataset):
    for split in a_dataset.keys():
        ds_split = a_dataset[split].to_pandas()
        print("summary of " + split + " split")
        print(ds_split.info())
        titles = ds_split["title"]
        print("number of titles: " + str(len(set(titles))))
        print((set(titles)))
        bag_of_titles = Counter(titles)
        print(
            "titles with most numerous examples: "
            + str((bag_of_titles.most_common(20)))
            + "\n"
        )


def ids_questions_answers_from_log_file(test_log_filepath):
    with open(test_log_filepath, "r") as test_log_file:
        test_log_lines = test_log_file.readlines()

    # parse the line with hte id and extract the id number then parse the line with the question and extract the question then parse the line with the END OF TESTING SAMPLE and extract the content of the previous line

    ids_questions_answers = []
    for index, line in enumerate(test_log_lines):
        if line.startswith("#id:"):
            id_number = line.strip().split(":")[-1].strip()
        elif line.startswith("?"):
            question = line.strip()
        elif line.startswith("#END OF TESTING SAMPLE"):
            previous_line = test_log_lines[index - 1].strip()
            answer = previous_line
            ids_questions_answers.append((id_number, question, answer))
        else:
            continue

    return ids_questions_answers


def embedding_for_sentence(a_string):
    embedding = ollama.embed(
        model="embeddinggemma",
        input=a_string,
    ).embeddings[0]
    return embedding


def cosine_distance(a_row):
    return cosine(
        a_row["test_answer_embedding"], a_row["response_answer_formatted_embedding"]
    )