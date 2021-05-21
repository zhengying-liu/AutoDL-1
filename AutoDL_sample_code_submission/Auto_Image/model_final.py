from learner import MyLearner           # Not used for now
from hp_optimizer import MyHPOptimizer
from data_ingestor import MyDataIngestor


class Model():

    def __init__(self, metadata):
        # `training_info` starts with the metadata and can be updated to store 
        # any intermediate training information
        self.training_info = {}
        self.training_info['metadata'] = metadata

        # Instantiate an HPOptimizer
        self.hp_optimizer = MyHPOptimizer()

        # Instantiate a DataIngestor
        self.data_ingestor = MyDataIngestor()

        # Get learner using the HPOptimizer. 
        # The learner can absorb `training_info` as its own attribute
        self.learner = self.hp_optimizer.fit(self.training_info)

    def train(self, dataset):
        dataset_uw = self.data_ingestor(dataset)    # uw for universal workflow
        self.predictor = self.learner.learn(dataset_uw)

    def test(self, dataset):
        dataset_uw = self.data_ingestor(dataset)
        predictions = self.predictor.predict(dataset_uw)
        return predictions