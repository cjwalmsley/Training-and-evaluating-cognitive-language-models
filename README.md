# Training and Evaluating Cognitive Language Models

## Project Aim

To explore the capabilities of a cognitive language model with datasets containing specialist topics. The model is trained with declarative sentences containing subject knowledge and then tested by presenting questions on the material then comparing the answers to ground truth answers.

## Models and datasets

The initial model used is ANNABELL is a cognitive language model that has previously been demonstrated to learn language skills like a 4-year-old child (1).
The dataset used for training and evaluation is the Stanford Question Answering Dataset (SQuAD) (2).

1. Golosio B, Cangelosi A, Gamotina O, Masala GL. A Cognitive Neural Architecture Able to Learn and Communicate through Natural Language. PLOS ONE [Internet]. 2015 Nov 11 [cited 2025 Feb 22];10(11):e0140866. Available from: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140866

2. Rajpurkar P, Zhang J, Lopyrev K, Liang P. SQuAD: 100,000+ Questions for Machine Comprehension of Text [Internet]. arXiv; 2016 [cited 2025 May 5]. Available from: http://arxiv.org/abs/1606.05250

## Overview

The workflow to train and evaluate the model has the following steps:

1.  **Training Data Generation**:
      * A large language models (LLM) is used to generate declarative sentences from the pre-existing Question and Answers  in the SQuAD database.
      * The data is further refined by applying formatting rules to allow ANNABELL to process it.
      * The training samples are put into categories based on their grammatical structure using a LLM.
2.  **Pre-training**: Training ANNABELL with a subset of the training data samples.
     * Samples are taken form each grammatical category and training commands are automatically created for the samples using rules.
3.  **Training**: Present the model with the declarative statements in the training set tht were not used in pretraining.
4.  **Evaluation**: Test the model's performance by presenting the trained model with questions and compare the results to the ground truth answers.

## Key items in the Directory Structure

-   `generate_declarative_sentences.py`: Notebook for generating declarative sentences with a prompt per row, using a LLM run locally with Ollama.
-   `training/prepare_declarative_sentences_prompt.ipynb`: Notebook for generating declarative sentences in a single prompt using a LLM running in the cloud.
-   `prompts`: Prompts for using the LLMs to generate declarative sentences and categorisation of samples.
-   `pre_training/nyc_pretraining_training_testing.ipynb`: Notebook for creating pretraining, training and testing data.
-   `dataset_processing.py`: Python code for processing datasets.
-   `docker/`: Contains Dockerfile and Compose file for building and running the containerized environment.
-   `docker/scripts/`: Shell scripts to run training and evaluation inside the Docker container.
-   `pre_training/categorisation/categorise_statements_and_questions.ipynb`: Notebook used to categorise the statements and questions across all samples.
-   `testing/test_annabell.ipynb`: Notebook for evaluating the model.
-   `prompts/`: A collection of prompts used to generate data from other LLMs.

## Getting Started

### Prerequisites

-   Docker
-   An environment capable of running shell scripts (e.g., WSL, Git Bash).

### Installation & Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/cjwalmsley/Training-and-evaluating-cognitive-language-models.git
    cd Training-and-evaluating-cognitive-language-models
    ```

2.  Build and run the Docker container:
    ```bash
    cd docker
    docker-compose up --build
    ```

## Usage

The main training and evaluation pipelines are managed through shell scripts located in the `docker/scripts/` directory. These scripts are intended to be run from within the Docker container.

### Example: Pre-training Annabell

```bash
docker compose run --remove-orphans --entrypoint ./pre_train_annabell_squad_nyc.sh app data/pre-training/<LOGFILE_NAME> data/pre-training/<PRETRAINING COMMANDS FILENAME> data/pre-training/<WEIGHTS FILENAME>>