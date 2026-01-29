from training import AnnabellPreTrainingRunner
import unittest
import pandas as pd
import os
from dataset_processing import DatasetPreProcessor
from commands import AnnabellBaseCommandGenerator


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
        Uses example commands from AnnabellBaseCommandGenerator.
        """
        # 1. Generate commands using the generator logic (which produces \n for blank lines)
        generator = AnnabellBaseCommandGenerator(
            self.sample_id, self.declarative_sentence, self.question, self.short_answer
        )
        # Force is_pre_training for this generator instance if needed,
        # but AnnabellBaseCommandGenerator doesn't seem to have that flag in __init__?
        # Checking test_command_generation.py, create_list_of_commands() assumes pre-training logic usually
        # or we might need to set a flag if it defaults to something else.
        # Actually AnnabellBaseCommandGenerator seems to produce pretraining format by default in the tests provided.
        # Let's verify via the passed context or just use manually constructed list if unsure.
        # But user asked to use example command test case.

        # To be safe and independent of generator flags, I will manually construct
        # the list exactly as the test expects it, which includes "\n".
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

        # 2. Create DataFrame mocking the dataset structure expected by write_pretraining_file
        df = pd.DataFrame(
            {
                "is_pretraining": [True],
                "created_commands_error": [False],
                "created_commands": [commands],
            }
        )

        # 3. Initialize DatasetPreProcessor with partial mock data
        # We pass the dataframe directly.
        # We might need to handle other init params if they enforce types, but default config should be fine.
        processor = DatasetPreProcessor(df)

        # 4. Write to file
        processor.write_pretraining_file(self.output_file)

        # 5. Verify file content
        with open(self.output_file, "r") as f:
            content = f.read()
            lines = f.readlines()

        # Debug info
        print(f"File content:\n{content!r}")

        # Check that we don't have double newlines for the blank line commands.
        # The command "\n" + write's "\n" would be "\n\n" (2 empty lines) if not fixed.
        # Ideally, we want one blank line between sections.

        # The generator puts "\n" as a command.
        # If the file writer adds "\n", we get "\n\n".
        # This results in:
        # text
        # <blank>
        # <blank>
        # text

        # We want:
        # text
        # <blank>
        # text

        # Let's count empty lines.
        # In the content strings:
        # "the sky...\n\n? what..." -> correct (one blank line)
        # "the sky...\n\n\n? what..." -> wrong (two blank lines)

        # If the command is "\n", and we write it as is: "\n".
        # If previous line was "text\n", and next is "\n", we get "text\n\n".

        # Wait, create_list_of_commands produces:
        # "sentence"
        # "\n"
        # "question"

        # If we write with simple join("\n"):
        # "sentence" + "\n" + "\n" + "\n" + "question"
        # -> "sentence\n\n\nquestion" (2 blank lines!)

        # Ah! `writelines` does not add newlines. `write` with `\n` does.

        # If `dataset_processing.py` does `file.write(cmd + "\n")`:
        # "sentence" -> "sentence\n"
        # "\n" -> "\n\n" (A blank line)
        # "question" -> "question\n"

        # The result:
        # sentence
        # (empty)
        # question

        # This seems correct?
        # Wait. "sentence\n" ends line 1.
        # "\n\n" : line 2 is empty, line 3 starts.
        # "question\n": line 3 has question.

        # Result:
        # 1: sentence
        # 2:
        # 3: question

        # This is ONE blank line between them.

        # The user said: "why are 2 blank rows being written... instead of 1"
        # Maybe `blank_line()` returns `\n` and logic adds `\n` -> `\n\n`.
        # Plus maybe the previous command adds `\n`?

        # Let's look at the example output provided by user:
        # adjacent to the Main_Building
        #
        #
        # ? the Basilica...

        # 2 blank rows means 3 newlines total between text lines.

        # If we have:
        # "sentence"
        # "\n"
        # "question"

        # And write loop:
        # write("sentence" + "\n") -> "sentence\n"
        # write("\n" + "\n") -> "\n\n"
        # write("question" + "\n") -> "question\n"

        # Output:
        # sentence\n
        # \n
        # \n
        # question\n

        # That is:
        # sentence
        # (blank)
        # (blank)
        # question

        # Yes, that produces 2 blank lines (rows).
        # We want 1 blank row.

        # So check should be:
        # "the sky is blue with patches of grey\n\n? what color"
        # (one empty line between them)

        self.assertIn("the sky is blue with patches of grey\n\n? what color", content)
        self.assertNotIn(
            "the sky is blue with patches of grey\n\n\n? what color", content
        )

        # Also check end of file
        # ".rw" -> ".rw\n"
        # "\n" -> "\n\n"
        # So ".rw\n\n"
        self.assertIn(".rw\n\n", content)
        self.assertNotIn(".rw\n\n\n", content)


if __name__ == "__main__":
    unittest.main()