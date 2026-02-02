import unittest
import pandas as pd
import os
from dataset_processing import DatasetPreProcessor
from annabell_utilities import AnnabellLogfileInterpreter


class TestPretrainingFileGeneration(unittest.TestCase):

    def setUp(self):
        self.sample_id = "test_01"
        self.declarative_sentence = "the sky is blue with patches of grey"
        self.question = "? what color is the sky"
        self.short_answer = "blue"
        self.output_file = "test_pretraining_commands_output.txt"
        self.sample_commands = [
            "#id: test_01",
            "the sky is blue with patches of grey",
            "\n",
            "? what color is the sky",
            ".pg sky",
            ".ggp",
            ".ph the sky is blue with patches of grey",
            ".drop_goal",
            ".wg blue",
            ".rw",
            "\n",
        ]

        self.df = pd.DataFrame(
            {
                "is_pretraining": [True],
                "created_commands_error": [False],
                "created_commands": [self.sample_commands],
            }
        )
        self.processor = DatasetPreProcessor(self.df)

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_write_pretraining_file_formatting(self):
        """
        Tests that write_pretraining_file correctly handles commands that may or may not
        already include newlines, preventing double blank lines.
        """
        self.processor.write_pretraining_file(self.output_file, False)
        with open(self.output_file, "r") as f:
            content = f.read()

        self.assertIn("the sky is blue with patches of grey\n\n? what color", content)
        self.assertNotIn(
            "the sky is blue with patches of grey\n\n\n? what color", content
        )
        self.assertIn(".rw\n\n", content)
        self.assertNotIn(".rw\n\n\n", content)

    def test_write_pretraining_file_sample_count(self):

        self.processor.write_pretraining_file(self.output_file, False)
        with open(self.output_file, "r") as f:
            content = f.read()

        self.assertIn(
            f"{AnnabellLogfileInterpreter.sample_number_count_string()} 1 of 1", content
        )

    def test_write_pretraining_file_start_of_sample(self):

        self.processor.write_pretraining_file(self.output_file, False)
        with open(self.output_file, "r") as f:
            content = f.read()

        self.assertIn(f"{AnnabellLogfileInterpreter.start_of_sample_string()}", content)

    def test_write_pretraining_file_with_auto_saving_of_weights(self):
        self.processor.write_pretraining_file(self.output_file, True)
        with open(self.output_file, "r") as f:
            content = f.read()

        self.assertIn(f"{DatasetPreProcessor.auto_save_weights_command()}", content)


if __name__ == "__main__":
    unittest.main()