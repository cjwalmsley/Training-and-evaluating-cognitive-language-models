import logging
import ollama
from dataset_processing import load_squad_dataset
import timeit
from datetime import datetime
import pandas as pd
import sys
import json
from pydantic import BaseModel, ValidationError
from config.global_config import GlobalConfig

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


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
        model=global_config.ollama_model(),
        prompt=the_prompt,
        format=DeclarativeStatement.model_json_schema(),
        stream=global_config.ollama_stream(),
        think=global_config.ollama_think(),
        options=global_config.ollama_options_dict(),
    )
    logger.info("Generated response: " + str(generated_text.response).strip())
    logger.info("Total duration: " + str(generated_text.total_duration))
    return generated_text.response


def prompt_prefix_from_file():
    with open("prompts/prompt_prefix_for_squad", "r") as prefix_file:
        prompt_prefix = prefix_file.read()
    return prompt_prefix


def process_prompt(the_base_prompt, the_line, the_id):
    the_prompt = the_base_prompt + "\n" + the_line
    generated_json = generated_model_from_prompt(the_prompt, the_id)
    return generated_json


def items_with_title(the_dataset, the_title):
    df = pd.DataFrame(the_dataset)
    return df[df.apply(lambda x: x["title"] == the_title, axis=1)]


def filter_dataset_split(the_dataset_split, title, number_of_sentences, the_id=None):

    if the_id is not None:
        filtered_database_split = the_dataset_split.filter(lambda x: x["id"] == the_id)
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
    # Create the 'answer' column from the 'answers' dictionary
    # make a dataframe from the dataset split
    return filtered_database_split


def create_prompt_input_dataframe(the_dataset):

    filtered_dataframe = pd.DataFrame(the_dataset)

    filtered_dataframe["answer"] = filtered_dataframe["answers"].apply(
        lambda x: x["text"][0] if x["text"] else ""
    )

    # Drop the original, now unneeded, columns
    filtered_dataframe = filtered_dataframe.drop(
        columns=["context", "title", "answers"]
    )

    # Add the new column for the response, which will now be at the end
    filtered_dataframe["response_declarative_sentence"] = (
        "<INSERT RESPONSE SENTENCE HERE>"
    )

    # Reorder columns to a specific, desired order
    final_columns = [
        "id",
        "question",
        "answer",
        "response_declarative_sentence",
    ]
    filtered_dataframe = filtered_dataframe[final_columns]

    filtered_dataframe.reset_index(drop=True, inplace=True)

    return filtered_dataframe


def create_prompt_input_jsonl(the_dataframe):
    # write the dataframe to a jsonl file
    the_dataframe.to_json(
        global_config.prompt_inputs_jsonl_filepath(), orient="records", lines=True
    )
    logger.info("dataframe written to: " + global_config.prompt_inputs_jsonl_filepath())
    return the_dataframe


def prompt_tuple_from_json_line(the_json_line):

    # convert the json line to a dict to get the id
    line_dict = json.loads(the_json_line)
    the_id = line_dict["id"]
    # Create a dictionary for the prompt
    prompt_dict = {
        "question": line_dict["question"],
        "answer": line_dict["answer"],
        "response_declarative_sentence": line_dict["response_declarative_sentence"],
    }
    # Convert the dictionary to a JSON string
    line_json = json.dumps(prompt_dict)
    return the_id, line_json


def timestamped_response_filepath(ds_split_name):
    suffix = ".jsonl"
    filepath = (
        global_config.responses_jsonl_filepath().removesuffix(suffix)
        + "_"
        + ds_split_name
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + suffix
    )
    return filepath


def generate_declarative_sentences(
    ds, number_of_sentences, the_model_string, splits, debug_id=None, title="all"
):

    for ds_split_name in splits:
        prompt_json_l_filepath = global_config.prompt_inputs_jsonl_filepath()
        base_prompt_filepath = global_config.base_prompt_filepath()
        examples_generated = 0
        # create output filepath using response_filepath with the model string and current datetime appended to the filename

        output_filepath = timestamped_response_filepath(ds_split_name)

        dataset_split = ds[ds_split_name]
        logger.info("Filtering dataset split: " + ds_split_name)
        filtered_database_split = filter_dataset_split(
            dataset_split, title, number_of_sentences, debug_id
        )

        prepared_dataframe = create_prompt_input_dataframe(filtered_database_split)
        create_prompt_input_jsonl(prepared_dataframe)
        number_of_examples = len(prepared_dataframe)

        logger.info(
            "generating: " + str(number_of_examples) + " examples\t"
            "database_split: " + ds_split_name + "\tusing  model: " + the_model_string
        )

        with open(base_prompt_filepath, "r") as base_prompt_file:
            base_prompt = base_prompt_file.read()
        with open(prompt_json_l_filepath, "r") as prompt_json_l_file:
            prompt_json_l = prompt_json_l_file.readlines()
        start_time = timeit.default_timer()
        for line in prompt_json_l:
            the_id, line_json = prompt_tuple_from_json_line(line)

            try:
                response_model = process_prompt(base_prompt, line_json, the_id)
                logger.info("Response:\n" + str(response_model))
                with open(output_filepath, "a") as response_file:
                    response_file.write(response_model.model_dump_json() + "\n")

            except ValidationError as error:
                logger.critical(
                    "Error processing example with id: "
                    + str(the_id)
                    + "\t"
                    + str(error)
                )
            examples_generated = examples_generated + 1
            logger.info(
                "generated: "
                + str(examples_generated)
                + " of: "
                + str(number_of_examples)
            )
            if examples_generated == number_of_examples:
                break
        end_time = timeit.default_timer()
        total_elapsed = end_time - start_time
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
        logger.info(completion_message)


def clean_line(a_string):
    cleaned_string = a_string.replace('"', "")
    return cleaned_string


if __name__ == "__main__":

    print(f"🚀 Starting project: {global_config.project_name}")
    print(f"   Using API key starting with: {global_config.wandb_api_key()[:4]}...")

    model_strings = global_config.ollama_models()
    options = global_config.ollama_options_dict()

    print("\n--- Model Config ---")
    print(f"   options: {options}")
    print(f"   available models: {model_strings}")

    # You can even export the final, validated config
    print("\n--- Final Settings (as JSON) ---")
    print(global_config.settings.model_dump_json(indent=2))

    if len(sys.argv) == 5:
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

    splits = ["train"]

    dataset = load_squad_dataset(global_config.dataset_directory())

    generate_declarative_sentences(
        dataset,
        num_sentences_arg,
        model_string,
        splits,
        debug_id=None,
        title=title_arg,
    )