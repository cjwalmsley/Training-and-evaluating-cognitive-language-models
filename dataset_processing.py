import re
from collections import Counter
from unidecode import unidecode
import warnings
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset, load_from_disk
from scipy.spatial.distance import cosine
import ollama


def load_squad_dataset(ds_filename="squad_dataset"):
    # check if dataset is available on disk, if not load it
    try:
        print("Loading dataset from: " + ds_filename)
        ds = load_from_disk(ds_filename)
    except FileNotFoundError:
        print(
            "File not found, loading dataset from Huggingface and saving to: "
            + ds_filename
        )
        ds = load_dataset("rajpurkar/squad")
        ds.save_to_disk(ds_filename)
    return ds


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


def is_pretraining_question(the_question, the_pretraining_questions):
    result = the_question in the_pretraining_questions
    if result:
        print(f"Pretraining question found: {the_question}")
    return result


def write_training_file(a_series, a_filepath):
    # write a file that can be used to train ANNABELL
    with open(a_filepath, "w") as file:
        for statement in a_series:
            file.write(statement + "\n")
    print(f"file created: {a_filepath}")


def write_testing_file(
    a_list, a_filepath
):  # write a file that can be used to test ANNABELL
    with open(a_filepath, "w") as test_file:
        for question in a_list:
            test_file.write(f"{question}\n.x\n")
    print(f"file created: {a_filepath}")


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


def data_frame_up_to_statement_title(a_dataframe, a_statemment):
    row = a_dataframe[a_dataframe["statement"] == a_statemment]
    row_index = row.index
    print(f"Row index of statement: '{a_statemment}': {row_index}")
    target_title = row["title"].values[0]
    # find the index of the first row where the title is "Tuscon_Arizona"
    title_index = a_dataframe[a_dataframe["title"] == target_title].index
    print(
        f"Index of first row with title {target_title}: {title_index[0] if not title_index.empty else 'Not found'}"
    )
    # filter the dataset so that all the rows up to the title index are included
    filtered_train_df = a_dataframe.iloc[: title_index[0]]
    print(f"Filtered DataFrame shape: {filtered_train_df.shape}")
    return filtered_train_df


def question_and_answer_pairs_from_log_file(test_log_filepath):
    with open(test_log_filepath, "r") as test_log_file:
        test_log_lines = test_log_file.readlines()

    question_and_answer_pairs = []
    for index, line in enumerate(test_log_lines):
        if line.startswith("?") or line.startswith(".stat") and index != 0:
            new_pair = [None, None]
            question = line.strip()
            new_pair[0] = question
            question_and_answer_pairs.append(new_pair)
            new_pair_index = question_and_answer_pairs.index(new_pair)
            previous_pair_index = new_pair_index - 1
            previous_pair = question_and_answer_pairs[previous_pair_index]
            previous_answer = test_log_lines[index - 1].strip()
            if previous_answer is None:
                warnings.warn(f"Previous answer is None for question: {question}")
            previous_pair[-1] = previous_answer
            if previous_pair[-1] is None:
                warnings.warn(f"Previous answer is None for question: {question}")
        else:
            continue
    return question_and_answer_pairs


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
