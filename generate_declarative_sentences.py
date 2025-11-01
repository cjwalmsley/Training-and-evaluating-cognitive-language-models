import logging
import ollama
import logger as my_logger
from dataset_processing import load_squad_dataset
import timeit
from datetime import datetime
import pandas as pd
import sys
import os
import json
from pydantic import BaseModel, ValidationError
from config.config import PlatformConfig

logger = logging.getLogger(__name__)
platform_config = PlatformConfig()


class DeclarativeStatement(BaseModel):
    declarative_statement: str


def generated_model_from_prompt(the_prompt, id_string):
    logger.info(f"Generating statement from prompt for id: {id_string}..")
    the_response = generate_response_with_prompt(
        the_prompt=the_prompt,
    )
    logger.debug("raw response: " + the_response)
    return DeclarativeStatement.model_validate_json(the_response)


def generate_response_with_prompt(the_prompt):

    generated_text = ollama.generate(
        model=platform_config.ollama_model(),
        prompt=the_prompt,
        format=DeclarativeStatement.model_json_schema(),
        stream=platform_config.ollama_stream(),
        think=platform_config.ollama_think(),
        options=platform_config.ollama_options_dict(),
    )
    logger.info("Generated response: " + str(generated_text.response).strip())
    logger.info("Total duration: " + str(generated_text.total_duration))
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


def filter_dataset_split(the_dataset_split, title, number_of_sentences, id=None):

    if id is not None:
        filtered_database_split = the_dataset_split.filter(lambda x: x["id"] == id)
    elif title != "all":
        filtered_database_split = the_dataset_split.filter(
            lambda x: x["title"] == title
        )
        if number_of_sentences == "all":
            pass
        else:
            filtered_database_split = filtered_database_split.select(
                range(min(number_of_sentences, len(the_dataset_split)))
            )
    else:
        filtered_database_split = the_dataset_split
    return filtered_database_split


def generate_declarative_sentences(
    ds, number_of_sentences, the_model_string, the_options, id=None, title="all"
):

    # set up output directories depending on machine and connected storage
    if os.path.exists("/Volumes/X9 Pro/datasets"):
        output_directory = "/Volumes/X9 Pro/datasets"
    elif os.path.exists("/Users/chris/datasets"):
        output_directory = "/Users/chris/datasets"
    else:
        output_directory = "/home/chris/datasets"

    log_writer = my_logger.LogWriter("declarative_statement_generation.log")

    for ds_split_name in ("train", "validation"):
        total_elapsed = 0
        examples_generated = 0
        output_filename = (
            "declarative_sentences_"
            + ds_split_name
            + "_"
            + the_model_string
            + "_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".tsv"
        )
        output_filepath = os.path.join(output_directory, output_filename)
        dataset_split = ds[ds_split_name]
        filtered_database_split = filter_dataset_split(
            dataset_split, title, number_of_sentences, id
        )
        number_of_examples = filtered_database_split.num_rows
        log_writer.log(
            "generating: " + str(number_of_examples) + " examples\t"
            "database_split: "
            + ds_split_name
            + "\tusing  model: "
            + the_model_string
            + "\twith: options: "
            + str(the_options)
            + "\twith prompt_prefix: "
            + prompt_prefix_from_file()
        )
        with open(output_filepath, "w") as output_file:
            output_file.write(
                "id\ttitle\tquestion\tanswer\tresponse_question\tresponse_answer\tstatement\n"
            )
            for example in filtered_database_split:
                try:
                    example_id = example["id"]
                    title = example["title"]
                    question = example["question"]
                    answer = example["answers"]["text"][0]
                    prompt = prepare_prompt(question, answer)
                    start_time = timeit.default_timer()
                    response = generated_model_from_prompt(prompt, example_id)
                    response_question = response.question
                    response_answer = response.answer
                    statement = response.statement
                    elapsed = timeit.default_timer() - start_time
                    log_writer.log(
                        "processed example: "
                        + str(examples_generated)
                        + "\tmodel_string: "
                        + model_string
                        + "\texecution_time_in_seconds: "
                        + str(elapsed)
                        + "\tprompt_question: "
                        + question
                        + "\tprompt_answer: "
                        + answer
                        + "\tresponse_question: "
                        + response_question
                        + "\tresponse_question: "
                        + response_answer
                        + "\tstatement: "
                        + statement
                    )
                    file_entry = (
                        example_id
                        + "\t"
                        + title
                        + "\t"
                        + question
                        + "\t"
                        + answer
                        + "\t"
                        + response_question
                        + "\t"
                        + response_answer
                        + "\t"
                        + statement
                        + "\n"
                    )
                    output_file.write(file_entry)
                    total_elapsed = total_elapsed + elapsed
                    examples_generated = examples_generated + 1
                # except IndexError as index_error:
                # log_writer.log("Index Error processing example with id: " + str(example_id) + "\t" + str(index_error))
                except ValidationError as error:
                    log_writer.log(
                        "Error processing example with id: "
                        + str(example_id)
                        + "\t"
                        + str(error)
                        + "Response: "
                        + response
                    )
                if examples_generated == number_of_examples:
                    break
            completion_message = (
                "generated: "
                + str(examples_generated)
                + " examples\t"
                + "using  model: "
                + model_string
                + "\t"
                + "total_execution_time_in_seconds: "
                + str(total_elapsed)
                + "\toutput_filepath:"
                + output_filepath
                + "\n"
            )
            log_writer.log(completion_message)
            print(completion_message)


def clean_line(a_string):
    cleaned_string = a_string.replace('"', "")
    return cleaned_string


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

    with open(config_filepath, "r") as config_file:
        options = json.load(config_file)
    return options


if __name__ == "__main__":

    print(f"🚀 Starting project: {platform_config.project_name}")
    print(f"   Using API key starting with: {platform_config.wandb_api_key()[:4]}...")

    model_strings = platform_config.ollama_models()
    options = platform_config.ollama_options_dict()

    print("\n--- Model Config ---")
    print(f"   options: {options}")
    print(f"   available models: {model_strings}")

    # You can even export the final, validated config
    print("\n--- Final Settings (as JSON) ---")
    print(platform_config.settings.model_dump_json(indent=2))

    if len(sys.argv) == 4:
        model_string = sys.argv[1]
        title_arg = sys.argv[2]
        num_sentences_arg = sys.argv[3]

        # Convert num_sentences_arg to int if it's a digit, otherwise keep as string (for 'all')
        if num_sentences_arg.isdigit():
            num_sentences_arg = int(num_sentences_arg)

    else:
        print(
            "Usage: python generate_declarative_sentences.py <model_string> <title> <number_of_sentences_or_all>"
        )
        # Default execution if no args or wrong number of args are provided
        model_string = model_strings[-1]
        title_arg = "New_York_City"
        num_sentences_arg = 5

    dataset = load_squad_dataset(platform_config.dataset_directory())

    generate_declarative_sentences(
        dataset, num_sentences_arg, model_string, options, id=None, title=title_arg
    )