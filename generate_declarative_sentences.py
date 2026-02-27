import logging
import ollama
from dataset_processing import load_squad_dataset, filter_dataset_split
import timeit
import pandas as pd
import sys
import json
from pydantic import BaseModel
from config.global_config import GlobalConfig

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class DeclarativeStatement(BaseModel):
    declarative_statement: str


class DeclarativeStatementWithID(DeclarativeStatement):
    example_id: str


def generated_model_from_prompt(the_prompt, id_string, the_model_string):
    logger.info(f"Generating statement from prompt for id: {id_string}..")
    the_response = generate_response_with_prompt(the_model_string, the_prompt)

    logger.debug("raw response: " + the_response)

    try:
        # Parse the JSON response from the model
        response_data = json.loads(the_response)

        # Add the example_id to the dictionary
        response_data["example_id"] = id_string

        # Validate the complete data against the Pydantic model
        return DeclarativeStatementWithID.model_validate(response_data)
    except json.JSONDecodeError as e:
        logger.error(
            f"JSONDecodeError for id {id_string}: {e}. Raw response: {the_response}"
        )
        return None


def generate_response_with_prompt(the_model_string, the_prompt):

    generated_text = ollama.generate(
        model=the_model_string,
        prompt=the_prompt,
        format=DeclarativeStatement.model_json_schema(),
        stream=global_config.ollama_stream(),
        think=global_config.ollama_think(),
        options=global_config.ollama_options_dict(),
    )
    logger.info("Generated response: " + str(generated_text.response).strip())
    logger.info("Total duration: " + str(generated_text.total_duration))
    return generated_text.response


def process_prompt(the_base_prompt, the_line, the_id, the_model_string):
    the_prompt = the_base_prompt + "\n" + the_line
    generated_model = generated_model_from_prompt(the_prompt, the_id, the_model_string)
    return generated_model


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
    filtered_dataframe["declarative_statement"] = "<INSERT RESPONSE SENTENCE HERE>"

    # Reorder columns to a specific, desired order
    final_columns = [
        "id",
        "question",
        "answer",
        "declarative_statement",
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
        "declarative_statement": line_dict["declarative_statement"],
    }
    # Convert the dictionary to a JSON string
    line_json = json.dumps(prompt_dict)
    return the_id, line_json


def response_filepath(ds_split_name, the_model_string):
    suffix = ".jsonl"
    filepath = (
        global_config.responses_jsonl_filepath().removesuffix(suffix)
        + "_"
        + ds_split_name
        + "_"
        + the_model_string.replace(":", "_")
        + suffix
    )
    return filepath


def generate_declarative_statements(
    number_of_sentences,
    the_model_string,
    ds_split_name="train",
    debug_id=None,
    title="all",
):

    logger.info(
        f"🚀 Starting sentence generation for experiment: {global_config.experiment_name()}"
    )
    logger.info("\n--- Model Config ---")
    logger.info(f"   options: {global_config.ollama_options_dict()}")
    logger.info(f"   model: {the_model_string}")
    logger.info("\n--- Final Settings (as JSON) ---")
    logger.info(global_config.settings.model_dump_json(indent=2))

    prompt_json_l_filepath = global_config.prompt_inputs_jsonl_filepath()
    base_prompt_filepath = "prompts/base_prompt.txt"
    examples_generated = 0

    output_filepath = response_filepath(ds_split_name, the_model_string)

    dataset_split = load_squad_dataset(global_config.dataset_directory())[ds_split_name]
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
    responses = []
    for line in prompt_json_l:
        the_id, line_json = prompt_tuple_from_json_line(line)

        response_model = process_prompt(
            base_prompt, line_json, the_id, the_model_string
        )
        if response_model and response_model.declarative_statement:
            responses.append(response_model.model_dump_json())
        else:
            logger.error(f"No valid response generated for id: {the_id}")

        examples_generated = examples_generated + 1
        logger.info(
            "generated: " + str(examples_generated) + " of: " + str(number_of_examples)
        )
        if examples_generated == number_of_examples:
            break

    with open(output_filepath, "w") as response_file:
        for response in responses:
            response_file.write(response + "\n")

    end_time = timeit.default_timer()
    total_elapsed = end_time - start_time
    completion_message = (
        "generated: "
        + str(examples_generated)
        + " examples\t"
        + "using  model: "
        + the_model_string
        + "\t"
        + "total_execution_time_in_seconds: "
        + str(total_elapsed)
        + "\toutput_filepath:"
        + output_filepath
        + "\n"
    )
    logger.info(completion_message)
    return create_dataset_with_generated_sentences(
        output_filepath, filtered_database_split
    )


def create_dataset_with_generated_sentences(
    a_sentences_jsonl_filepath, filtered_database_split
):
    # Load the generated sentences from the JSONL file
    generated_sentences_df = pd.read_json(a_sentences_jsonl_filepath, lines=True)
    # create a new dataframe form the filtered database split
    filtered_df = pd.DataFrame(filtered_database_split)
    # add the sentences to the original dataframe based on the id
    merged_df = filtered_df.merge(
        generated_sentences_df,
        left_on="id",
        right_on="example_id",
        how="left",
    )
    # Drop the now-redundant 'example_id' column
    merged_df = merged_df.drop(columns=["example_id"])
    # Save the merged dataframe to a new JSONL file
    merged_df.to_json(
        global_config.dataset_with_generated_sentences_filepath(),
        orient="records",
        lines=True,
    )
    logger.info(
        "Dataset with generated sentences saved to: "
        + global_config.dataset_with_generated_sentences_filepath()
    )
    return merged_df


def clean_line(a_string):
    cleaned_string = a_string.replace('"', "")
    return cleaned_string


if __name__ == "__main__":

    # for running from command line with args

    if len(sys.argv) == 4:
        model_string_arg = sys.argv[1]
        title_arg = sys.argv[2]
        num_sentences_arg = sys.argv[3]
        # Convert num_sentences_arg to int if it's a digit, otherwise keep as string (for 'all')
        if num_sentences_arg.isdigit():
            num_sentences_arg = int(num_sentences_arg)
        logger.info(
            f"Arguments received - Model: {model_string_arg}, Title: {title_arg}, Number of Sentences: {num_sentences_arg}"
        )

    else:
        model_string_arg = global_config.ollama_models()[-1]
        title_arg = "New_York_City"
        num_sentences_arg = 5

    generate_declarative_statements(
        num_sentences_arg,
        model_string_arg,
        title=title_arg,
    )
