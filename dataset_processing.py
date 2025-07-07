import re
from collections import Counter
from unidecode import unidecode
import tarfile
import warnings


def remove_quotes(text):
    # remove " form the text
    cleaned_text = text.replace('"', '')
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

#move the ? from the end of each question to the start
def move_question_mark_to_start(question, add_if_missing=True):

    if question.strip().endswith("?"):
        edited_question = "? " + question[:-1]
    else:
        if add_if_missing:
            edited_question = "? " + question
        else:
            #raise an exception if the question does not end with a ?
            raise ValueError(f"Question does not end with a question mark: {question}")

    return edited_question

def replace_decimal_in_matched_string(matched_string):
    #callable function to support the regex
    number_str = matched_string.group(0)
    return number_str.replace('.', ' point ')

def convert_decimal_point_to_word(a_string):
    #Replace "." with 'point' if it is part of a number.

    # Regex to find numbers with a decimal point:
    # \d+\.\d+  matches numbers like 1.23
    # \d+\.     matches numbers like 1. (dot at the end after digits)
    # \.\d+     matches numbers like .5 (dot at the beginning before digits)
    # The order matters to match \d+\.\d+ before \d+\. or \.\d+ for overlapping cases.
    pattern = r'\d+\.\d+|\d+\.|\.\d+'
    return re.sub(pattern, replace_decimal_in_matched_string, a_string)

def remove_accents(text):
    #Convert accented characters to unaccented ones
    text_unaccented = unidecode(text)
    return text_unaccented

#remove all special characters except question marks and hyphen form the statements
def remove_special_characters(text):
    """
    Removes special characters from a string, keeping alphanumeric characters and spaces.
    """
    # Keep only alphanumeric characters and spaces
    cleaned_text = re.sub(r'[^A-Za-z0-9\s?-]+', '', text)
    return cleaned_text

def filter_by_max_words(the_df, max_words=10):
    #returnes a new dataframe filtered such that each question, answer and statement has less than 11 words

    filtered_df = the_df[the_df.apply(lambda row: len(row["question"].split()) <= max_words and len(row["answer"][0].split()) <= max_words and len(row["statement"].split()) <= max_words , axis=1)]
    return filtered_df

def clean_text(a_series, is_question):
    #takes a dataframe series and applies reformatting
    if is_question:
        a_series = a_series.apply(move_question_mark_to_start)
    for function_name in (convert_decimal_point_to_word, remove_accents, remove_special_characters):
        a_series = a_series.apply(function_name)
    return a_series

def write_training_file(a_dataframe, a_filepath_minus_extension):
    #write a file that can be used to train ANNABELL
    output_filepath = a_filepath_minus_extension + ".txt"
    compressed_output_filepath = a_filepath_minus_extension + ".tar.xz"
    with open(output_filepath, "w") as output_file:
        for statement in a_dataframe["statement"]:
            output_file.write(statement + "\n")
    #check that the file looks correct
    print(f"file created: {output_filepath}")
    #compress the file to tar.xz format
    with tarfile.open(compressed_output_filepath, "w:xz") as tar:
        tar.add(output_filepath, arcname="output_filepath.txt")
    print(f"Compressed file created: {compressed_output_filepath}")

def write_testing_file(a_dataframe, a_filepath):        #write a file that can be used to test ANNABELL
    with open(a_filepath, "w") as test_file:
        for index, row in a_dataframe.iterrows():
            question = row["response_question"]
            test_file.write(f"{question}\n.x\n")

#produce a summary of a dataset by splits
def dataset_summary(a_dataset):
    for split in a_dataset.keys():
        ds_split=a_dataset[split].to_pandas()
        print("summary of " + split + " split")
        print(ds_split.info())
        titles=ds_split["title"]
        print("number of titles: " + str(len(set(titles))))
        print((set(titles)))
        bag_of_titles = Counter((titles))
        print("titles with most numerous examples: " + str((bag_of_titles.most_common(20))) +"\n")

def data_frame_up_to_statement_title(a_dataframe, a_statemment):
    row = a_dataframe[a_dataframe["statement"] == a_statemment]
    row_index = row.index
    print(f"Row index of statement: '{a_statemment}': {row_index}")
    target_title = row["title"].values[0]
    #find the index of the first row where the title is "Tuscon_Arizona"
    title_index = a_dataframe[a_dataframe["title"] == target_title].index
    print(f"Index of first row with title {target_title}: {title_index[0] if not title_index.empty else 'Not found'}")
    #filter the dataset so that all the rows up to the title index are included
    filtered_train_df = a_dataframe.iloc[:title_index[0]]
    print(f"Filtered DataFrame shape: {filtered_train_df.shape}")
    return filtered_train_df

def question_and_answer_pairs_from_log_file(test_log_filepath):
    with open(test_log_filepath, "r") as test_log_file:
        test_log_lines = test_log_file.readlines()

    question_and_answer_pairs = []
    for index, line in enumerate(test_log_lines):
        if line.startswith("?"):
            new_pair = [None, None]
            question = line.strip()
            new_pair[0] = question
            question_and_answer_pairs.append(new_pair)
            new_pair_index = question_and_answer_pairs.index(new_pair)
            previous_pair_index = new_pair_index - 1
            previous_pair = question_and_answer_pairs[previous_pair_index]
            previous_answer = test_log_lines[index -1].strip()
            if previous_answer is None:
                warnings.warn(f"Previous answer is None for question: {question}")
            previous_pair[-1] = previous_answer
            if previous_pair[-1] is None:
                warnings.warn(f"Previous answer is None for question: {question}")
        else:
            continue
    #print("length of log file questions: " + str(len(questions)))
    #print("length of log file answers: " + str(len(answers)))
    print("length of log file questions and answers: " + str(len(question_and_answer_pairs)))
    return question_and_answer_pairs
