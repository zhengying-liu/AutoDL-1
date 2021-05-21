from automl_workflow.api import Learner, Predictor

from data_loader import MyDataLoader


class MyPredictor(Predictor):

    def predict(self, dataset):
        data_loader = MyDataLoader()
        pt_dataloader = data_loader(dataset, train=False)


class MyLearner(Learner):

    # TODO: define components such as BackboneModel, DataIngestor, DataLoader,
    #   Optimizer, HPOptimizer. Potentially in a separate file for each 
    #   component.
    def __init__(self):
        pass

    
    def learn(self, train_set) -> Predictor:
        # TODO: Update training info, if necessary
        pass

        predictor = MyPredictor()
        return predictor
        