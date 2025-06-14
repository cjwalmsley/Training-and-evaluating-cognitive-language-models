import ollama
import logger
from datasets import load_dataset, load_from_disk
import timeit
from datetime import datetime
import pandas as pd
import sys
import os
import json

def generated_text_from_prompt(the_prompt, model_string, the_options):
    generated_text = ollama.generate(model=model_string, prompt=the_prompt, options=the_options)
    return generated_text.response

def prompt_prefix_from_file():
    with open("prompts/prompt_prefix_for_squad", "r") as prefix_file:
        prompt_prefix = prefix_file.read()
    return prompt_prefix

def prepare_prompt(question, answer):

    prompt_suffix = "question: " + question + "\nanswer: " + answer

    prompt = prompt_prefix_from_file() + "\n" + prompt_suffix
    return prompt

def items_with_title(the_dataset, the_title):
    df = pd.DataFrame(the_dataset)
    return df[df.apply(lambda x: x["title"] == the_title, axis=1)]

def load_squad_dataset():

    #check if dataset is available on disk, if not load it
    ds_filename = "squad_dataset"
    try:
        ds = load_from_disk(ds_filename)
    except FileNotFoundError:
        ds = load_dataset("rajpurkar/squad")
        ds.save_to_disk(ds_filename)
    return ds

def filter_dataset_split(the_dataset_split, title, number_of_sentences, id=None):

    if id is not None:
        the_dataset = the_dataset_split.filter(lambda x: x["id"] == id)
    else:
        if title != 'all':
            the_dataset = the_dataset_split.filter(lambda x: x["title"] == title)

        if number_of_sentences == 'all':
            pass
        else:
            the_dataset = the_dataset.select(range(min(number_of_sentences, len(the_dataset_split))))

    return the_dataset

def generate_declarative_sentences(ds, number_of_sentences, the_model_string, the_options, id=None, title="all"):

    #set up output directories depending on machine and connected storage
    if os.path.exists("/Volumes/X9 Pro/datasets"):
        output_directory = "/Volumes/X9 Pro/datasets"
    elif os.path.exists("/Users/chris/datasets"):
        output_directory = "/Users/chris/datasets"
    else: output_directory = "/home/chris/datasets"

    log_writer = logger.LogWriter("declarative_statement_generation.log")

    total_elapsed = 0
    examples_generated = 0
    for ds_split_name in ("train", "validation"):
        output_filename = "declarative_sentences_" + ds_split_name + "_" + the_model_string + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".tsv"
        output_filepath = os.path.join(output_directory, output_filename)
        dataset_split = ds[ds_split_name]
        filtered_database_split = filter_dataset_split(dataset_split, title, number_of_sentences, id)
        number_of_examples = filtered_database_split.num_rows
        log_writer.log("generating: " + str(
            number_of_examples) + " examples\t" "database_split: " + ds_split_name + "\tusing  model: " + the_model_string + "\twith: options: " + str(
            the_options) + "\twith prompt_prefix: " + prompt_prefix_from_file())
        with (open(output_filepath, "w") as output_file):
            output_file.write("id\ttitle\tquestion\tanswer\tresponse_question\tresponse_answer\tstatement\n")
            for example in filtered_database_split:
                try:
                    example_id = example["id"]
                    title = example["title"]
                    question = example["question"]
                    answer = example["answers"]["text"][0]
                    prompt = prepare_prompt(question, answer)
                    start_time = timeit.default_timer()
                    response = generated_text_from_prompt(prompt, model_string, the_options)
                    response_sections = response.split("\n")
                    response_question = response_sections[0].strip()
                    response_answer = response_sections[1].strip()
                    statement = response_sections[2].strip()
                    elapsed = timeit.default_timer() - start_time
                    log_writer.log("model_string: " + model_string + "\texecution_time_in_seconds: " + str(elapsed) + "\tprompt_question: " + question +"\tprompt_answer: " + answer + "\tresponse_question: " + response_question
                   + "\tresponse_question: " + response_answer + "\tstatement: " + statement)
                    file_entry = example_id + "\t" + title + "\t" + question + "\t" + answer + "\t" + response_question + "\t" + response_answer + "\t" + statement + "\n"
                    output_file.write(file_entry)
                    total_elapsed = total_elapsed + elapsed
                    examples_generated = examples_generated + 1
                except IndexError as index_error:
                    log_writer.log("Index Error processing example with id: " + str(example_id) + "\t" + str(index_error))
                except Exception as error:
                    log_writer.log("Error processing example with id: " + str(example_id) + "\t" + str(error))
                if examples_generated == number_of_examples:
                    break
            completion_message = "generated: " + str(examples_generated) + " examples\t" + "using  model: " + model_string + "\t" + "total_execution_time_in_seconds: " + str(total_elapsed) + "\toutput_filepath:" + output_filepath + "\n"
            log_writer.log(completion_message)
            print(completion_message)

def clean_line(a_string):
    cleaned_string = a_string.replace('"', '')
    return  cleaned_string

def clean_file(a_filepath):
    with open(a_filepath, "r") as unclean_file:
        unclean_lines = unclean_file.readlines()
    clean_lines = [clean_line(unclean_line) for unclean_line in unclean_lines]
    clean_filepath = "clean_" + a_filepath
    with open(clean_filepath, "w") as clean_file:
        for line in clean_lines:
            clean_file.write(line)
    print("cleaned: " + a_filepath + " new file: " + clean_filepath)
    return clean_filepath

def load_options_from_config_file(config_filepath="options_config.json"):

    with open(config_filepath, 'r') as config_file:
        options = json.load(config_file)
    return options

if __name__ == "__main__":

    model_strings = (
        "llama3.2",
        "llama3",
        "deepseek-r1:8b",
        "gemma3:4b")

    options = load_options_from_config_file()

    if len(sys.argv) == 4:
        model_string = sys.argv[1]
        title_arg = sys.argv[2]
        num_sentences_arg = sys.argv[3]

        # Convert num_sentences_arg to int if it's a digit, otherwise keep as string (for 'all')
        if num_sentences_arg.isdigit():
            num_sentences_arg = int(num_sentences_arg)

    else:
        print("Usage: python generate_declarative_sentences.py <model_string> <title> <number_of_sentences_or_all>")
        # Default execution if no args or wrong number of args are provided
        model_string = model_strings[-1]
        title_arg = "Frédéric_Chopin"
        num_sentences_arg = 5

    dataset = load_squad_dataset()
    #id = "56cfdef3234ae51400d9bfc2"

    generate_declarative_sentences(dataset, num_sentences_arg, model_string, options, id=None, title=title_arg)