from dataset_processing import (
    embedding_for_sentence,
    ids_questions_answers_from_log_file,
)
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords
from config.global_config import GlobalConfig
import logging

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class AnnabellTestResultsEvaluator:

    def __init__(self, testing_context):
        self.testing_context = testing_context
        self.ids_questions_answers = None
        self.prepared_dataframe = self.testing_context.prepared_dataframe().copy()

    def run(
        self,
    ):
        self.setup()
        self.run_processing()
        self.teardown()

    def setup(self):
        self.ids_questions_answers = ids_questions_answers_from_log_file(
            self.testing_context.testing_log_file()
        )
        logger.info(
            "length of log file questions and answers: "
            + str(len(self.ids_questions_answers))
        )

    def default_blank_or_none_answers(self):
        for index, each_tuple in enumerate(self.ids_questions_answers):
            if each_tuple[-1] == "" or each_tuple[-1] is None:
                self.ids_questions_answers[index] = (
                    each_tuple[0],
                    each_tuple[1],
                    "NO ANSWER GIVEN",
                )

    def total_number_of_test_samples(self):
        with open(self.testing_context.test_input_filepath(), "r") as test_input_file:
            test_input_lines = test_input_file.readlines()
        total_number_of_test_samples = len(
            [id_line for id_line in test_input_lines if id_line.startswith("#id:")]
        )
        logger.info(
            f"total number of test samples in input file: {total_number_of_test_samples}"
        )

        return total_number_of_test_samples

    def add_answers_column_to_dataframe(self):
        questions_not_found = []
        for (
            the_id,
            question,
            answer,
        ) in self.ids_questions_answers:
            if the_id in self.prepared_dataframe["id"].values:
                self.prepared_dataframe.loc[
                    self.prepared_dataframe["id"] == the_id, "test_answer"
                ] = answer
            else:
                questions_not_found.append(question)
        logger.info(
            f"number of test samples not found in training data: {len(questions_not_found)}"
        )
        logger.info(
            "test samples not found in training data: "
            + str(questions_not_found[:5])
            + " ..."
        )

    def remove_samples_with_no_answers(self):
        # drop any rows that do not have a test answer
        self.prepared_dataframe.dropna(subset=["test_answer"], inplace=True)
        self.prepared_dataframe.reset_index(inplace=True)

    def add_cosine_distance(self):

        self.prepared_dataframe["test_answer_cosine_distance"] = (
            self.prepared_dataframe.apply(self.cosine_distance, axis=1)
        )

    def write_test_answer_summary(self):
        # Get the counts for each unique value in the 'test_answer' column
        test_answer_summary = (
            self.prepared_dataframe["test_answer"].value_counts().reset_index()
        )
        # Rename the columns for clarity
        test_answer_summary.columns = ["test_answer", "count"]
        # Sort the results by count in descending order
        test_answer_summary.sort_values(by="count", ascending=False, inplace=True)

        # write the results dataframe to a tsv file
        test_answer_summary.to_csv(
            self.testing_context.test_answer_summary_filepath(), sep="\t", index=False
        )

    def count_of_long_test_answers(self):
        num_long_answers = (
            self.prepared_dataframe["test_answer"]
            .apply(lambda x: len(x.split()) > 20 if pd.notnull(x) else False)
            .sum()
        )

        return num_long_answers

    def number_of_correct_answers(self):

        self.prepared_dataframe["test_answer_correct"] = (
            self.prepared_dataframe["test_answer"]
            == self.prepared_dataframe["answer_formatted"]
        )
        number_correct = self.prepared_dataframe["test_answer_correct"].sum()
        return number_correct

    def percentage_of_correct_answers(self):
        percentage_correct = self.prepared_dataframe["test_answer_correct"].mean() * 100
        return percentage_correct

    def add_any_answer_word_match_column(self):
        self.prepared_dataframe["test_answer_any_matching_word"] = (
            self.prepared_dataframe.apply(
                self.answers_with_any_word_match_to_ground_truth, axis=1
            )
        )

    def count_of_answers_with_any_word_match(self):
        return self.prepared_dataframe["test_answer_any_matching_word"].sum()

    def percentage_of_answers_with_any_word_match(self):
        percentage_any_word_matches = (
            self.prepared_dataframe["test_answer_any_matching_word"].mean() * 100
        )
        return percentage_any_word_matches

    def count_of_answers_below_cosine_distance_threshold(self):

        threshold = global_config.cosine_distance_threshold()
        return (
            self.prepared_dataframe["test_answer_cosine_distance"] < threshold
        ).sum()

    def percentage_of_answers_below_cosine_distance_threshold(self):
        return (
            self.count_of_answers_below_cosine_distance_threshold()
            / len(self.prepared_dataframe)
            * 100
        )

    def count_of_answers_with_any_word_match_below_cosine_distance_threshold(self):
        threshold = global_config.cosine_distance_threshold()
        return (
            (self.prepared_dataframe["test_answer_any_matching_word"])
            & (self.prepared_dataframe["test_answer_cosine_distance"] < threshold)
        ).sum()

    def correct_matches(self):
        return self.prepared_dataframe[self.prepared_dataframe["test_answer_correct"]]

    def any_matches(self):
        return self.prepared_dataframe[
            self.prepared_dataframe["test_answer_any_matching_word"]
        ]

    def incorrect_matches(self):
        return self.prepared_dataframe[~self.prepared_dataframe["test_answer_correct"]]

    def percentage_of_answers_with_any_word_match_below_cosine_distance_threshold(self):
        return (
            self.count_of_answers_with_any_word_match_below_cosine_distance_threshold()
            / len(self.prepared_dataframe)
            * 100
        )

    def close_cosine_distance_df(self):
        threshold = global_config.cosine_distance_threshold()
        return self.prepared_dataframe[
            self.prepared_dataframe["test_answer_cosine_distance"] < threshold
        ]

    def close_cosine_distance_correct_df(self):
        threshold = global_config.cosine_distance_threshold()
        return self.prepared_dataframe[
            (self.prepared_dataframe["test_answer_cosine_distance"] < threshold)
            & (self.prepared_dataframe["test_answer_correct"])
        ]

    def any_matches_below_cosine_distance_threshold(self):
        threshold = global_config.cosine_distance_threshold()
        return self.prepared_dataframe[
            (self.prepared_dataframe["test_answer_cosine_distance"] < threshold)
            & (self.prepared_dataframe["test_answer_correct"])
        ]

    def run_processing(self):

        self.add_answers_column_to_dataframe()
        self.remove_samples_with_no_answers()
        self.generate_embeddings()
        self.add_cosine_distance()
        self.write_test_answer_summary()

        # count the number of results where the test answer is > 20 words
        logger.info(
            f"number of test answers longer than 20 words: {str(self.count_of_long_test_answers())}"
        )
        logger.info(
            f"number correct = {str(self.number_of_correct_answers())} out of {str(len(self.prepared_dataframe))}"
        )
        logger.info(
            f"percentage correct = {str(self.percentage_of_correct_answers())} %"
        )

        self.add_any_answer_word_match_column()
        logger.info(
            f"number any word matches = {str(self.count_of_answers_with_any_word_match())} out of {str(len(self.prepared_dataframe))}"
        )
        logger.info(
            f"percentage any word matches = {str(self.percentage_of_answers_with_any_word_match())} %"
        )

        logger.info(
            f"number of rows with cosine distance less than {str(global_config.cosine_distance_threshold())}: {str(self.count_of_answers_below_cosine_distance_threshold())}"
        )
        logger.info(
            "percentage of total: "
            + str(self.percentage_of_answers_below_cosine_distance_threshold)
            + " %"
        )

        logger.info(
            f"number of rows with cosine distance less than {str(global_config.cosine_distance_threshold())} and any matching answer correct: {str(self.count_of_answers_with_any_word_match_below_cosine_distance_threshold())}"
        )
        print(
            "percentage of total: "
            + str(
                self.percentage_of_answers_with_any_word_match_below_cosine_distance_threshold()
            )
            + " %"
        )

        # write the results to a file and export the results dataframe to a tsv file
        self.write_results_to_file()

    def write_results_to_file(self):
        detailed_results_filepath = (
            self.testing_context.test_detailed_results_filepath()
        )
        self.prepared_dataframe.to_csv(detailed_results_filepath, sep="\t", index=False)

        results_summary_filepath = self.testing_context.test_summary_results_filepath()

        with open(results_summary_filepath, "w") as results_file:
            # write the number of samples tested
            results_file.write(
                f"total number of samples\t{str(self.total_number_of_test_samples())}\n"
            )
            results_file.write(
                f"number_of_test_answers\t{str(len(self.prepared_dataframe))}\n"
            )
            results_file.write(
                f"total_number_of_pretraining_samples\t{str(self.testing_context.total_number_of_pretraining_samples)}\n"
            )
            results_file.write(
                f"percentage_correct\t{str(self.percentage_of_correct_answers())}\n"
            )
            results_file.write(
                f"percentage_any_word_matches\t{self.percentage_of_answers_with_any_word_match()}\n"
            )
            results_file.write(
                f"percentage_close_cosine_distance\t{str(self.percentage_of_answers_below_cosine_distance_threshold())}\n"
            )
            results_file.write(
                f"percentage_close_cosine_distance_and_any_word_match\t{str(self.percentage_of_answers_with_any_word_match_below_cosine_distance_threshold())}\n"
            )
            results_file.write(
                f"number of test answers longer than 20 words (removed)\t{str(self.count_of_long_test_answers())}\n"
            )
            # write the rows that had exact word matches to the file
            results_file.write("\nRows with exact matches:\n")
            results_file.write(
                self.correct_matches()[
                    ["question", "answer", "test_answer"]
                ].to_markdown(index=False)
            )
            # write the rows in any_matches to the file
            results_file.write("\nRows with any word matches:\n")
            results_file.write(
                self.any_matches()[["question", "answer", "test_answer"]].to_markdown(
                    index=False
                )
            )
            # write the rows that had a close cosine distance to the file
            results_file.write(
                f"\nRows with cosine distance less than {str(global_config.cosine_distance_threshold())}:\n"
            )
            results_file.write(
                self.close_cosine_distance_df()[
                    [
                        "question",
                        "answer",
                        "test_answer",
                        "test_answer_cosine_distance",
                    ]
                ].to_markdown(index=False)
            )
            # write the rows that had a close cosine distance and any word match to the file
            results_file.write(
                f"\nRows with cosine distance less than {str(global_config.cosine_distance_threshold())} and any word match:\n"
            )
            results_file.write(
                self.any_matches_below_cosine_distance_threshold()[
                    [
                        "question",
                        "answer",
                        "test_answer",
                        "test_answer_cosine_distance",
                    ]
                ].to_markdown(index=False)
            )
            # write the rows that had any matches and with a close cosine distance to the file
            results_file.write(
                f"\nRows with cosine distance less than {str(global_config.cosine_distance_threshold())} and exact match:\n"
            )
            results_file.write(
                self.correct_matches()[
                    [
                        "question",
                        "answer",
                        "test_answer",
                        "test_answer_cosine_distance",
                    ]
                ].to_markdown(index=False)
            )
        print(
            f"results written to {detailed_results_filepath} and {results_summary_filepath}"
        )

    def generate_embeddings(self):
        # generate embeddings for the test answer and the answer_formatted columns and compare them using cosine distance
        tqdm.pandas(desc="Generating test answer embeddings")
        self.prepared_dataframe["test_answer_embedding"] = self.prepared_dataframe[
            "test_answer"
        ].progress_apply(
            lambda x: (
                embedding_for_sentence(x) if pd.notnull(x) and (len(x) > 0) else None
            )
        )
        tqdm.pandas(desc="Generating response answer embeddings")
        self.prepared_dataframe["answer_formatted_embedding"] = self.prepared_dataframe[
            "answer_formatted"
        ].progress_apply(lambda x: embedding_for_sentence(x) if pd.notnull(x) else None)

    @staticmethod
    def cosine_distance(a_row):
        return cosine(
            a_row["test_answer_embedding"], a_row["answer_formatted_embedding"]
        )

    @staticmethod
    def answers_with_any_word_match_to_ground_truth(row):
        # if the row contains a non-string value return False
        if not isinstance(row["test_answer"], str) or not isinstance(
            row["answer_formatted"], str
        ):
            return False
        # return True if any word in test_answer is also in answer_formatted
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        test_words = set(row["test_answer"].split())
        response_words = set(row["answer_formatted"].split())
        intersecting_words = test_words.intersection(response_words)
        open_class_intersecting_words = intersecting_words - stop_words

        return open_class_intersecting_words != set()

    def write_annabell_files_to_gdrive(self):
        pass

    def teardown(self):
        self.write_annabell_files_to_gdrive()


class AnnabellTestContext:
    def __init__(self, dataset_processor):
        self.dataset_processor = dataset_processor

    def testing_log_file(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def test_input_filepath(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def prepared_dataframe(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def test_answer_summary_filepath(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def test_detailed_results_filepath(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def test_summary_results_filepath(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def total_number_of_pretraining_samples(self):
        raise NotImplementedError("Subclasses should implement this method.")


class AnnabellPreTrainingTestContext(AnnabellTestContext):

    def testing_log_file(self):
        return global_config.pretraining_validation_testing_log_filepath()

    def test_input_filepath(self):
        return global_config.pre_training_filepath()

    def prepared_dataframe(self):
        return self.dataset_processor.pretraining_dataset()

    def test_answer_summary_filepath(self):
        return global_config.test_pre_training_validation_answer_summary_filepath()

    def test_detailed_results_filepath(self):
        return global_config.test_pre_training_validation_detailed_results_filepath()

    def test_summary_results_filepath(self):
        return global_config.test_pre_training_validation_summary_results_filepath()

    def total_number_of_pretraining_samples(self):
        return 0


class AnnabellTrainingTestContext(AnnabellTestContext):

    def testing_log_file(self):
        return global_config.testing_log_filepath()

    def test_input_filepath(self):
        return global_config.training_filepath()

    def prepared_dataframe(self):
        return self.dataset_processor.training_dataset()

    def test_answer_summary_filepath(self):
        return global_config.test_answer_summary_filepath()

    def test_detailed_results_filepath(self):
        return global_config.test_detailed_results_filepath()

    def test_summary_results_filepath(self):
        return global_config.test_summary_results_filepath()

    def total_number_of_pretraining_samples(self):
        return self.dataset_processor.total_number_of_pretraining_samples()