import logging
import ollama
import timeit
import pandas as pd
import sys
import json
from pydantic import BaseModel, ValidationError
from config.global_config import GlobalConfig

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class Category(BaseModel):
    category_name: str


class CategoryWithId(Category):
    example_id: str


class AbstractCategoryAssigner:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    @staticmethod
    def sentence_category_name():
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def sentence_column_name():
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def no_category_string():
        return "NO CATEGORY ASSIGNED"

    def assign_category(self, row):
        sentence = row[self.sentence_column_name()]
        the_id = row["id"]
        base_prompt = self.get_base_prompt()
        line_json = json.dumps({"sentence": sentence})
        try:
            response_model = self.process_prompt(base_prompt, line_json, the_id)
            return response_model.category_name
        except Exception as e:
            logger.error(f"Error assigning category for id {the_id}: {e}")
            return None

    def generated_model_from_prompt(self, the_prompt, id_string):
        logger.info(f"selecting category from prompt for id: {id_string}..")
        the_response = self.generate_response_with_prompt(
            the_prompt=the_prompt,
        )
        logger.debug("raw response: " + the_response)

        # Parse the JSON response from the model
        response_data = json.loads(the_response)

        # Add the example_id to the dictionary
        response_data["example_id"] = id_string

        # Validate the complete data against the Pydantic model

        try:

            response_model = CategoryWithId.model_validate(response_data)
            return response_model

        except ValidationError as error:
            logger.critical(
                "Error processing example with id: "
                + str(id_string)
                + "\t"
                + str(error)
            )

    @staticmethod
    def generate_response_with_prompt(the_prompt):

        generated_text = ollama.generate(
            model=global_config.ollama_model(),
            prompt=the_prompt,
            format=Category.model_json_schema(),
            stream=global_config.ollama_stream(),
            think=global_config.ollama_think(),
            options=global_config.ollama_options_dict(),
        )
        logger.info("Generated response: " + str(generated_text.response).strip())
        logger.info("Total duration: " + str(generated_text.total_duration))
        return generated_text.response

    def process_prompt(self, the_base_prompt, the_line, the_id):
        the_prompt = the_base_prompt + "\n" + the_line
        generated_model = self.generated_model_from_prompt(the_prompt, the_id)
        return generated_model

    @staticmethod
    def create_prompt_input_jsonl(the_dataframe):
        # write the dataframe to a jsonl file
        the_dataframe.to_json(
            global_config.prompt_inputs_jsonl_filepath(), orient="records", lines=True
        )
        logger.info(
            "dataframe written to: " + global_config.prompt_inputs_jsonl_filepath()
        )
        return the_dataframe

    def prompt_tuple_from_json_line(self, the_json_line):

        # convert the json line to a dict to get the id
        line_dict = json.loads(the_json_line)
        the_id = line_dict["id"]
        line_json = json.dumps(self.prompt_dict(line_dict))
        return the_id, line_json

    @staticmethod
    def prompt_dict(line_dict):
        prompt_dict = {"sentence": line_dict["sentence"]}
        return prompt_dict

    @staticmethod
    def get_base_prompt():

        base_prompt_part1_filepath = (
            global_config.classify_sentence_prompt_part_1_filepath()
        )
        with open(base_prompt_part1_filepath, "r") as base_prompt_part1_file:
            base_prompt_part1 = base_prompt_part1_file.read()
        base_prompt_part2_filepath = (
            global_config.classify_sentence_prompt_part_2_filepath()
        )
        with open(base_prompt_part2_filepath, "r") as base_prompt_part2_file:
            base_prompt_part2 = base_prompt_part2_file.read()
        sentence_pattern_filepath = global_config.sentence_patterns_filepath()
        with open(sentence_pattern_filepath, "r") as sentence_pattern_file:
            sentence_pattern_lines = sentence_pattern_file.readlines()
        sentence_patterns = [
            json.loads(line) for line in sentence_pattern_lines if line.strip()
        ]
        pattern_names = [pattern["name"] for pattern in sentence_patterns]
        pattern_names_string = "\n".join(pattern_names)
        base_prompt = (
            base_prompt_part1 + "\n\n" + pattern_names_string + base_prompt_part2
        )
        return base_prompt

    def generate_statement_categories(
        self,
        the_model_string,
    ):

        logger.info(f"🚀 Starting project: {global_config.project_name}")
        logger.info("\n--- Model Config ---")
        logger.info(f"   options: {global_config.ollama_options_dict()}")
        logger.info(f"   model: {the_model_string}")
        logger.info("\n--- Final Settings (as JSON) ---")
        logger.info(global_config.settings.model_dump_json(indent=2))

        examples_generated = 0
        number_of_examples = len(self.dataframe)

        logger.info(
            "generating: "
            + str(number_of_examples)
            + " examples\t"
            + "\tusing  model: "
            + the_model_string
        )

        start_time = timeit.default_timer()

        self.dataframe[self.sentence_category_name()] = self.dataframe.apply(
            self.assign_category, axis=1
        )

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
        )
        logger.info(completion_message)


class QuestionCategoryAssigner(AbstractCategoryAssigner):

    @staticmethod
    def sentence_category_name():
        return "question_category"

    @staticmethod
    def sentence_column_name():
        return "question"


class QuestionNoCategoryAssigner(QuestionCategoryAssigner):

    def generate_statement_categories(self, the_model_string):
        self.dataframe[self.sentence_category_name()] = self.dataframe.apply(
            self.assign_category, axis=1
        )

    def assign_category(self, row):
        return self.no_category_string()


class StatementCategoryAssigner(AbstractCategoryAssigner):

    @staticmethod
    def sentence_category_name():
        return "statement_category"

    @staticmethod
    def sentence_column_name():
        return "declarative_statement"


class StatementNoCategoryAssigner(StatementCategoryAssigner):

    def generate_statement_categories(self, the_model_string):
        self.dataframe[self.sentence_category_name()] = self.dataframe.apply(
            self.assign_category, axis=1
        )

    def assign_category(self, row):
        return self.no_category_string()


if __name__ == "__main__":

    # for running from command line with args

    if len(sys.argv) == 3:
        model_string_arg = sys.argv[1]
        dataset_filepath_arg = sys.argv[2]
        logger.info(
            f"Arguments received - Model: {model_string_arg}, Title: {dataset_filepath_arg}"
        )

    else:
        model_string_arg = global_config.ollama_models()[-1]
        dataset_filepath_arg = global_config.dataset_with_generated_sentences_filepath()

    # assign the dataset to the result of reading the jsonl file
    dataset = pd.read_json(dataset_filepath_arg, lines=True)

    QuestionCategoryAssigner(dataset).generate_statement_categories(
        model_string_arg,
    )
    StatementCategoryAssigner(dataset).generate_statement_categories(
        model_string_arg,
    )

    dataset_with_categories_filepath = (
        global_config.dataset_with_sentence_categories_filepath()
    )
    dataset.to_json(dataset_with_categories_filepath, orient="records", lines=True)
    logger.info(
        "Dataset with categories written to: " + dataset_with_categories_filepath
    )

    logger.info("Processing complete.")
