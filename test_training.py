import unittest
import pandas as pd
import os
from dataset_processing import DatasetPreProcessor


class TestPretrainingFileGeneration(unittest.TestCase):

    def setUp(self):
        self.sample_id = "test_01"
        self.declarative_sentence = "the sky is blue with patches of grey"
        self.question = "? what color is the sky"
        self.short_answer = "blue"
        self.output_file = "test_pretraining_commands_output.txt"

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_write_pretraining_file_formatting(self):
        """
        Tests that write_pretraining_file correctly handles commands that may or may not
        already include newlines, preventing double blank lines.
        """
        commands = [
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

        df = pd.DataFrame(
            {
                "is_pretraining": [True],
                "created_commands_error": [False],
                "created_commands": [commands],
            }
        )

        processor = DatasetPreProcessor(df)

        # 4. Write to file
        processor.write_pretraining_file(self.output_file)

        # 5. Verify file content
        with open(self.output_file, "r") as f:
            content = f.read()

        self.assertIn("the sky is blue with patches of grey\n\n? what color", content)
        self.assertNotIn(
            "the sky is blue with patches of grey\n\n\n? what color", content
        )
        self.assertIn(".rw\n\n", content)
        self.assertNotIn(".rw\n\n\n", content)


if __name__ == "__main__":
    unittest.main()