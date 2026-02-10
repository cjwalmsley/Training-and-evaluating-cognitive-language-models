import matplotlib.pyplot as plt


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

    @staticmethod
    def end_of_stats_string():
        return "#END OF STATS"

    @staticmethod
    def start_of_sample_string():
        return "#START OF SAMPLE"

    @staticmethod
    def id_string():
        return "#id:"

    @staticmethod
    def sample_number_count_string():
        return "#sample:"

    def parse_entries(self):
        # read the file and split the contents by the start_of_sample_string delimiter
        with open(self.logfile_path, "r") as file:
            content = file.read()
        raw_entries = content.split(self.start_of_sample_string())[
            1:
        ]  # Skip the first empty split
        self.entries.extend([self.create_entry(raw_entry) for raw_entry in raw_entries])

    def create_entry(self, raw_entry):

        lines = raw_entry.strip().split("\n")

        sample_number_line = [
            line for line in lines if line.startswith(self.sample_number_count_string())
        ][0].strip()
        entry_id_line = [line for line in lines if line.startswith(self.id_string())][
            0
        ].strip()
        index_of_id_line = lines.index(entry_id_line)
        index_of_declaration_end = lines.index(self.end_of_declaration_string())
        declaration_lines = lines[index_of_id_line + 1 : index_of_declaration_end]
        index_of_question_end = lines.index(self.end_of_question_string())
        question_lines = lines[index_of_declaration_end + 2 : index_of_question_end]
        index_of_commands_end = lines.index(self.end_of_commands_string())
        command_lines = lines[index_of_question_end + 1 : index_of_commands_end - 1]
        index_of_time_end = lines.index(self.end_of_time_string())
        time_lines = lines[index_of_commands_end + 1 : index_of_time_end]
        #       index_of_stats_end = lines.index(self.end_of_stats_string())
        index_of_stats_end = len(
            lines
        )  # If stats are not present, use the end of the lines
        stat_lines = lines[index_of_time_end + 1 : index_of_stats_end]

        return AnnabellLogEntry(
            self,
            sample_number_line,
            entry_id_line,
            declaration_lines,
            question_lines,
            command_lines,
            time_lines,
            stat_lines,
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

    def plot_entry_time_vs_sample_number(self):
        entry_times = [entry.time() for entry in self.entries]
        sample_numbers = [entry.sample_number() for entry in self.entries]
        plt.plot(sample_numbers, entry_times)
        plt.xlabel("Index")
        plt.ylabel("Entry Time (seconds)")
        plt.title("Entry Time vs Index")
        return plt

    def plot_stat_measures_vs_index(self):
        metrics = [
            "learned_words",
            "learned_phrases",
            "learned_associations",
            "iw_input_links",
            "elActfSt_neurons",
            "elActfSt_input_links",
            "elActfSt_virtual_input_links",
            "elActfSt_output_links",
            "remPh_output_links",
            "remPh_virtual_output_links",
            "remPhfWG_neurons",
            "remPhfWG_input_links",
            "remPhfWG_virtual_input_links",
            "remPhfWG_output_links",
            "remPhfWG_virtual_output_links",
        ]
        entries_with_stats = [entry for entry in self.entries if entry.stat_lines]
        sample_numbers = [entry.sample_number() for entry in entries_with_stats]

        plots = []
        for metric in metrics:
            stat_values = [getattr(entry, metric)() for entry in entries_with_stats]
            plt.figure()
            plt.plot(sample_numbers, stat_values)
            plt.xlabel("Index")
            plt.ylabel(metric)
            plt.title(f"{metric} vs Index")
            plots.append(plt)
        return plots


class AnnabellLogEntry:
    def __init__(
        self,
        interpreter,
        sample_number_line,
        entry_id_line,
        declaration_lines,
        question_lines,
        command_lines,
        time_lines,
        stat_lines,
    ):
        self.interpreter = interpreter
        self.sample_number_line = sample_number_line
        self.entry_id_line = entry_id_line
        self.declaration_lines = declaration_lines
        self.question_lines = question_lines
        self.command_lines = command_lines
        self.time_lines = time_lines
        self.stat_lines = stat_lines

    def __repr__(self):
        # Displays the phrase text and how many word groups it has
        return f"<AnnabellLogEntry: '{self.entry_id_line}' | Declaration: {self.declaration_lines[0]}>"

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

    def id(self):
        return self.entry_id_line.replace(self.interpreter.id_string(), "").strip()

    def sample_number(self):
        # "#sample: 1 of 3"
        return int(self.sample_number_line.split(":")[1].split("of")[0].strip())

    def learned_words(self):
        stat_line = [
            line for line in self.stat_lines if line.startswith("Learned Words:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def learned_phrases(self):
        stat_line = [
            line for line in self.stat_lines if line.startswith("Learned Phrases:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def learned_associations(self):
        stat_line = [
            line for line in self.stat_lines if line.startswith("Learned associations")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def iw_input_links(self):
        stat_line = [
            line for line in self.stat_lines if line.startswith("IW input links:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def elActfSt_neurons(self):
        stat_line = [
            line for line in self.stat_lines if line.startswith("ElActfSt neurons:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def elActfSt_input_links(self):
        stat_line = [
            line for line in self.stat_lines if line.startswith("ElActfSt input links:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def elActfSt_virtual_input_links(self):
        stat_line = [
            line
            for line in self.stat_lines
            if line.startswith("ElActfSt virtual input links:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def elActfSt_output_links(self):
        stat_line = [
            line
            for line in self.stat_lines
            if line.startswith("ElActfSt output links:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def remPh_output_links(self):
        stat_line = [
            line for line in self.stat_lines if line.startswith("RemPh output links:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def remPh_virtual_output_links(self):
        stat_line = [
            line
            for line in self.stat_lines
            if line.startswith("RemPh virtual output links:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def remPhfWG_neurons(self):
        stat_line = [
            line for line in self.stat_lines if line.startswith("RemPhfWG neurons:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def remPhfWG_input_links(self):
        stat_line = [
            line for line in self.stat_lines if line.startswith("RemPhfWG input links:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def remPhfWG_virtual_input_links(self):
        stat_line = [
            line
            for line in self.stat_lines
            if line.startswith("RemPhfWG virtual input links:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def remPhfWG_output_links(self):
        stat_line = [
            line
            for line in self.stat_lines
            if line.startswith("RemPhfWG output links:")
        ][0]
        return int(stat_line.split(":")[1].strip())

    def remPhfWG_virtual_output_links(self):
        stat_line = [
            line
            for line in self.stat_lines
            if line.startswith("RemPhfWG virtual output links:")
        ][0]
        return int(stat_line.split(":")[1].strip())
