class AnnabellLogfileInterpreter:
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path
        self.entries = []

    def __repr__(self):
        # Displays the phrase text and how many word groups it has
        return f"<AnnabellLogEntry: '{self.logfile_path}' | Number of Entries: {len(self.entries)}>"

    @staticmethod
    def end_of_declaration_string():
        return "#END OF DECLARATION"

    @staticmethod
    def end_of_question_string():
        return "#END OF QUESTION"

    @staticmethod
    def end_of_commands_string():
        return "#END OF COMMANDS"

    @staticmethod
    def end_of_time_string():
        return "#END OF TIME"

    def parse_entries(self):
        # read the file and split the contents by the delimiter "#id: "
        with open(self.logfile_path, "r") as file:
            content = file.read()
        raw_entries = content.split("#id: ")[1:]  # Skip the first empty split
        self.entries.extend([self.create_entry(raw_entry) for raw_entry in raw_entries])

    def create_entry(self, raw_entry):

        lines = raw_entry.strip().split("\n")

        entry_id = lines[0].strip()

        index_of_declaration_end = lines.index(self.end_of_declaration_string())
        declaration_lines = lines[1:index_of_declaration_end]
        index_of_question_end = lines.index(self.end_of_question_string())
        question_lines = lines[index_of_declaration_end + 2 : index_of_question_end]
        index_of_commands_end = lines.index(self.end_of_commands_string())
        command_lines = lines[index_of_question_end + 1 : index_of_commands_end - 1]
        index_of_time_end = lines.index(self.end_of_time_string())
        time_lines = lines[index_of_commands_end + 1 : index_of_time_end]

        return AnnabellLogEntry(
            self, entry_id, declaration_lines, question_lines, command_lines, time_lines
        )

    def previous_entry(self, reference_entry):
        reference_entry_index = self.entries.index(reference_entry)
        if reference_entry_index > 0:
            return self.entries[reference_entry_index - 1]
        else:
            return None

    def total_elapsed_time_computed(self):
        return sum(entry.time() for entry in self.entries)

    def total_elapsed_time_recorded(self):
        if self.entries:
            result = self.entries[-1].elapsed_time()
        else:
            result = 0.0
        return result


class AnnabellLogEntry:
    def __init__(
        self,
        interpreter,
        entry_id,
        declaration_lines,
        question_lines,
        command_lines,
        time_lines,
    ):
        self.interpreter = interpreter
        self.entry_id = entry_id
        self.declaration_lines = declaration_lines
        self.question_lines = question_lines
        self.command_lines = command_lines
        self.time_lines = time_lines

    def __repr__(self):
        # Displays the phrase text and how many word groups it has
        return f"<AnnabellLogEntry: '{self.entry_id}' | Declaration: {self.declaration_lines[0]}>"

    def elapsed_time(self):
        time_line = self.time_lines[1]
        return float(time_line.split(":")[1].strip())

    def time(self):

        previous_entry = self.interpreter.previous_entry(self)
        if previous_entry:
            previous_entry_elapsed_time = previous_entry.elapsed_time()
        else:
            previous_entry_elapsed_time = 0.0

        return self.elapsed_time() - previous_entry_elapsed_time