from learner import MyLearner           # Not used for now
from hp_optimizer import MyHPOptimizer
from data_ingestor import MyDataIngestor2


class Model():

    def __init__(self, metadata):
        # `training_info` starts with the metadata and can be updated to store 
        # any intermediate training information
        self.training_info = {}
        self.training_info['metadata'] = metadata

        # Instantiate an HPOptimizer
        self.hp_optimizer = MyHPOptimizer()

        # Instantiate a DataIngestor
        self.data_ingestor = MyDataIngestor2(info=self.training_info)

        # Get learner using the HPOptimizer. 
        # The learner can absorb `training_info` as its own attribute
        self.learner = self.hp_optimizer.fit(self.training_info)

        self.done_training = False

    def train(self, dataset, remaining_time_budget=None):
        dataset_uw = self.data_ingestor.ingest(dataset, mode='train')    # uw for universal workflow
        self.predictor = self.learner.learn(dataset_uw)
        # self.done_training = True

    def test(self, dataset, remaining_time_budget=None):
        dataset_uw = self.data_ingestor.ingest(dataset, mode='test')
        predictions = self.predictor.predict(dataset_uw)
        self.done_training = True
        return predictions