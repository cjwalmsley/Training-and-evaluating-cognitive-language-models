import ollama
import logger
from datasets import load_dataset
import timeit
from datetime import datetime
import pandas as pd
import sys

def generated_text_from_prompt(the_prompt, model_string):
    generated_text = ollama.generate(model=model_string, prompt=the_prompt)
    return generated_text.response

def prepare_prompt(question, answer):

    prompt = """you are a writing assistant whose job is to write a declarative statement.
The statement must have no punctuation and no special characters and proper nouns must begin with Capital letters.
for example: given this question and answer: The Basilica of the Sacred Heart at Notre Dame is beside to which structure? / the Main Building.
your output would be: the Basilica of the Sacred heart at Notre Dame is beside the Main Building
write a declarative statement from the following question and answer:\n""" + question + " / " + answer
    return prompt

def items_with_title(the_dataset, the_title):
    df = pd.DataFrame(the_dataset)
    return df[df.apply(lambda x: x["title"] == the_title, axis=1)]

def generate_declarative_sentences(the_model_string, title='all', number_of_sentences = 5):
    # generate an output file of n examples
    ds = load_dataset("rajpurkar/squad")

    output_filename = "declarative_statement_generation_output_" + the_model_string + "_" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ".tsv"

    if title != 'all':
        ds = ds.filter(lambda x: x["title"] == title)

    if number_of_sentences == 'all':
        number_of_examples = len(ds["train"])
    else:
       number_of_examples = min(len(ds["train"]), number_of_sentences)

    log_writer = logger.LogWriter("declarative_statement_generation.log")
    log_writer.log("generating: " + str(number_of_examples) + "examples\t" + "using  model: " + the_model_string)

    total_elapsed = 0
    examples_generated = 0
    with (open(output_filename, "w") as output_file):
        output_file.write("id\ttitle\tquestion\tanswer\tstatement\n")
        for example in ds["train"]:
            example_id = example["id"]
            title = example["title"]
            question = example["question"]
            answer = example["answers"]["text"][0]
            prompt = prepare_prompt(question, answer)
            start_time = timeit.default_timer()
            statement = generated_text_from_prompt(prompt, model_string)
            elapsed = timeit.default_timer() - start_time
            log_writer.log("model_string: " + model_string + "\texecution_time_in_seconds: " + str(elapsed) + "\tprompt: " + prompt + "\tstatement: " + statement)
            file_entry = example_id + "\t" + title + "\t" + question + "\t" + answer + "\t" + statement + "\n"
            output_file.write(file_entry)
            total_elapsed = total_elapsed + elapsed
            examples_generated = examples_generated + 1
            if examples_generated == number_of_examples:
                break
    log_writer.log("generated: " + str(
        examples_generated) + " examples\t" + "using  model: " + model_string + "\t" + "total_execution_time_in_seconds: " + str(total_elapsed) + "\n")

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






if __name__ == "__main__":

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
        model_string = "llama3.2"
        #model_string="llama3"
        #model_string = "gemma3:4b"
        title_arg = "Frédéric_Chopin"
        num_sentences_arg = 5  # Default to 5 if not specified

    generate_declarative_sentences(model_string, title_arg , num_sentences_arg )