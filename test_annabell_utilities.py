import unittest
import os
from annabell_utilities import AnnabellLogfileInterpreter


class TestAnnabellLogfileInterpreter(unittest.TestCase):

    def setUp(self):
        self.log_filepath = "test_annabell_pretraining_log.txt"
        with open(self.log_filepath, "w") as f:
            f.write(self.sample_logfile_content())
        self.interpreter = AnnabellLogfileInterpreter(self.log_filepath)
        self.interpreter.parse_entries()

    def tearDown(self):
        if os.path.exists(self.log_filepath):
            os.remove(self.log_filepath)

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


if __name__ == "__main__":
    unittest.main()