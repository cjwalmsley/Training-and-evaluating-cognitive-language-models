from config.global_config import GlobalConfig
import logging

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class AnnabellTestResultsEvaluator:

    def __init__(self, testing_context):
        self.testing_context = testing_context

    def run(
        self,
    ):
        self.setup()
        self.run_processing()
        self.teardown()

    def setup(self):
        pass

    def run_processing(self):
        pass

    def write_annabell_files_to_gdrive(self):
        pass

    def teardown(self):
        self.write_annabell_files_to_gdrive()


class AnnabellTestContext:
    def __init__(self, dataset_processor):
        self.dataset_processor = dataset_processor


class AnnabellPreTrainingTestContext(AnnabellTestContext):
    pass


class AnnabellTrainingTestContext(AnnabellTestContext):
    pass