import unittest
import os
from annabell_utilities import AnnabellLogfileInterpreter


class TestAnnabellLogfileInterpreter(unittest.TestCase):

    def setUp(self):
        self.log_filepath = "test_annabell_pretraining_log.txt"
        self.log_filepath_stat = "test_annabell_pretraining_log_with_stat.txt"
        with open(self.log_filepath, "w") as f:
            f.write(self.sample_logfile_content())
        with open(self.log_filepath_stat, "w") as f:
            f.write(self.sample_logfile_content_with_stat())
        self.interpreter = AnnabellLogfileInterpreter(self.log_filepath)
        self.interpreter.parse_entries()
        self.intepreter_stat = AnnabellLogfileInterpreter(self.log_filepath_stat)
        self.intepreter_stat.parse_entries()

    def tearDown(self):
        if os.path.exists(self.log_filepath):
            os.remove(self.log_filepath)
            if os.path.exists(self.log_filepath_stat):
                os.remove(self.log_filepath_stat)

    @staticmethod
    def sample_logfile_content_with_stat():
        return """#START OF SAMPLE
#sample: 10 of 270
#id: 5733ae924776f41900661014
Notre_Dame admit 3577 incoming student during the fall_semester of
2015
#END OF DECLARATION

 >>> End context
? how many incoming student do Notre_Dame admit in
fall_2015
#END OF QUESTION
.sctx ? how many incoming student do Notre_Dame admit in
? how many incoming student do Notre_Dame admit in
.wg incoming student
.push_goal
.wg Notre_Dame admit
.ggp
.ph Notre_Dame admit 3577 incoming student during the fall_semester of
.drop_goal
.wg 3577
.rw

 >>> End context
#END OF COMMANDS
.time
Elapsed time: 94.552508
StActMem->act_time: 0.056034
StActMem->as_time: 28.277105
ElActfSt->act_time: 0.932212
ElActfSt->as_time: 1.562486
RemPh->act_time: 36.495014
RemPh->as_time: 0.728138
RemPhfWG->act_time: 0.012018
RemPhfWG->as_time: 0.806025
ElActfSt neurons: 3040
ElActfSt links: 0
#END OF TIME
.stat
Learned Words: 92
Learned Phrases: 115
Learned associations between word groups and phrases: 568
IW input links: 175000
ElActfSt neurons: 3040
ElActfSt input links: 1358880
ElActfSt virtual input links: 11022840000
ElActfSt output links: 2400000
RemPh output links: 646
RemPh virtual output links: 11000000000
RemPhfWG neurons: 568
RemPhfWG input links: 5680
RemPhfWG virtual input links: 1001000000
RemPhfWG output links: 568
RemPhfWG virtual output links: 10000000000"""

    @staticmethod
    def sample_logfile_content():
        return f"""{AnnabellLogfileInterpreter.start_of_sample_string()}
#sample: 1 of 3
#id: 5733be284776f41900661180
the Basilica of the Sacred Heart at Notre_Dame be
adjacent to the Main_Building
{AnnabellLogfileInterpreter.end_of_declaration_string()}

? the Basilica of the sacred heart at Notre_Dame
be beside to which structure
{AnnabellLogfileInterpreter.end_of_question_string()}
.sctx ? the Basilica of the sacred heart at Notre_Dame
.pg Basilica
.wg Notre_Dame
.ggp
.ph the Basilica of the Sacred Heart at Notre_Dame be
.drop_goal
.sctx adjacent to the Main_Building
.wg the Main_Building
.rw

{AnnabellLogfileInterpreter.end_of_commands_string()}
.time
Elapsed time: 1.000000
StActMem->act_time: 0.000000
StActMem->as_time: 0.000000
ElActfSt->act_time: 0.000000
ElActfSt->as_time: 0.000000
RemPh->act_time: 0.000000
RemPh->as_time: 0.000000
RemPhfWG->act_time: 0.000000
RemPhfWG->as_time: 0.000000
ElActfSt neurons: 0
ElActfSt links: 0
{AnnabellLogfileInterpreter.end_of_time_string()}
{AnnabellLogfileInterpreter.start_of_sample_string()}
#sample: 2 of 3
#id: 5733be284776f4190066117e
atop the Main_Building of Notre - Dame a golden
statue of the Virgin_Mary be prominently display
{AnnabellLogfileInterpreter.end_of_declaration_string()}

? what sit on top of the Main_Building at
Notre_Dame
{AnnabellLogfileInterpreter.end_of_question_string()}
.sctx ? what sit on top of the Main_Building at
.pg Main_Building
.ggp
.ph atop the Main_Building of Notre - Dame a golden
.drop_goal
.wg a golden
.prw
.sctx statue of the Virgin_Mary be prominently display
.wg statue of the Virgin_Mary
.rw

{AnnabellLogfileInterpreter.end_of_commands_string()}
.time
Elapsed time: 4.000000
StActMem->act_time: 0.000000
StActMem->as_time: 0.000000
ElActfSt->act_time: 0.000000
ElActfSt->as_time: 0.000000
RemPh->act_time: 0.000000
RemPh->as_time: 0.000000
RemPhfWG->act_time: 0.000000
RemPhfWG->as_time: 0.000000
ElActfSt neurons: 0
ElActfSt links: 0
{AnnabellLogfileInterpreter.end_of_time_string()}
{AnnabellLogfileInterpreter.start_of_sample_string()}
#sample: 3 of 3
#id: 5733bf84d058e614000b61be
the Scholastic_Magazine of Notre_Dame begin publish in September_1876
{AnnabellLogfileInterpreter.end_of_declaration_string()}

? when do the Scholastic_Magazine of Notre dame begin
publish
{AnnabellLogfileInterpreter.end_of_question_string()}
.sctx ? when do the Scholastic_Magazine of Notre dame begin
.pg Scholastic_Magazine
.pg begin
.sctx publish
.wg publish
.ggp
.ph the Scholastic_Magazine of Notre_Dame begin publish in September_1876
.drop_goal
.drop_goal
.wg September_1876
.rw

{AnnabellLogfileInterpreter.end_of_commands_string()}
.time
Elapsed time: 10.000000
StActMem->act_time: 0.000000
StActMem->as_time: 0.000000
ElActfSt->act_time: 0.000000
ElActfSt->as_time: 0.000000
RemPh->act_time: 0.000000
RemPh->as_time: 0.000000
RemPhfWG->act_time: 0.000000
RemPhfWG->as_time: 0.000000
ElActfSt neurons: 0
ElActfSt links: 0
{AnnabellLogfileInterpreter.end_of_time_string()}
"""

    def test_timing_extraction(self):
        timings = [entry.time() for entry in self.interpreter.entries]
        self.assertEqual(3, len(timings))
        self.assertEqual(1.0, timings[0])
        self.assertEqual(3.0, timings[1])
        self.assertEqual(6.0, timings[2])
        elapsed_timings = [entry.elapsed_time() for entry in self.interpreter.entries]
        self.assertEqual(1.0, elapsed_timings[0], 1.0)
        self.assertEqual(4.0, elapsed_timings[1])
        self.assertEqual(10.0, elapsed_timings[2])
        self.assertEqual(10.0, self.interpreter.total_elapsed_time_recorded())
        self.assertEqual(
            self.interpreter.total_elapsed_time_recorded(),
            self.interpreter.total_elapsed_time_computed(),
        )

    def test_id_extraction(self):
        ids = [entry.id() for entry in self.interpreter.entries]
        self.assertEqual(3, len(ids))
        self.assertEqual("5733be284776f41900661180", ids[0])
        self.assertEqual("5733be284776f4190066117e", ids[1])
        self.assertEqual("5733bf84d058e614000b61be", ids[2])

    def test_sample_numbers(self):
        sample_numbers = [entry.sample_number() for entry in self.interpreter.entries]
        self.assertEqual(3, len(sample_numbers))
        self.assertEqual(1, sample_numbers[0])
        self.assertEqual(2, sample_numbers[1])
        self.assertEqual(3, sample_numbers[2])

    def test_stat_extraction(self):
        entry = self.intepreter_stat.entries[0]
        self.assertEqual(92, entry.learned_words())
        self.assertEqual(115, entry.learned_phrases())
        self.assertEqual(568, entry.learned_associations())
        self.assertEqual(175000, entry.iw_input_links())
        self.assertEqual(3040, entry.elActfSt_neurons())
        self.assertEqual(1358880, entry.elActfSt_input_links())
        self.assertEqual(11022840000, entry.elActfSt_virtual_input_links())
        self.assertEqual(2400000, entry.elActfSt_output_links())
        self.assertEqual(646, entry.remPh_output_links())
        self.assertEqual(11000000000, entry.remPh_virtual_output_links())
        self.assertEqual(568, entry.remPhfWG_neurons())
        self.assertEqual(5680, entry.remPhfWG_input_links())
        self.assertEqual(1001000000, entry.remPhfWG_virtual_input_links())
        self.assertEqual(568, entry.remPhfWG_output_links())
        self.assertEqual(10000000000, entry.remPhfWG_virtual_output_links())


if __name__ == "__main__":
    unittest.main()
