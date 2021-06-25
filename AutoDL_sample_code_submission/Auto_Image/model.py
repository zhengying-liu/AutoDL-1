from learner import MyLearner
from hp_optimizer import MyHPOptimizer
from data_ingestor import MyDataIngestor2 as MyDataIngestor


class Model():

    def __init__(self, metadata):
        # `train_info` starts with the metadata and can be updated to store 
        # any intermediate training information
        train_info = {}
        train_info['metadata'] = metadata

        # Instantiate an HPOptimizer
        hp_optimizer = MyHPOptimizer()

        # Instantiate a DataIngestor
        data_ingestor = MyDataIngestor(info=train_info)

        # Get learner using the HPOptimizer. 
        # The learner can absorb `training_info` as its own attribute
        self.learner = MyLearner(
            hp_optimizer=hp_optimizer,
            data_ingestor=data_ingestor,
            train_info=train_info,
        ) 

        self.done_training = False

    def train(self, dataset, remaining_time_budget=None):
        dataset_uw = self.learner.data_ingestor.ingest(dataset, mode='train')    # uw for universal workflow
        self.predictor = self.learner.learn(dataset_uw)

    def test(self, dataset, remaining_time_budget=None):
        dataset_uw = self.learner.data_ingestor.ingest(dataset, mode='test')
        predictions = self.predictor.predict(dataset_uw)
        self.done_training = True
        return predictions