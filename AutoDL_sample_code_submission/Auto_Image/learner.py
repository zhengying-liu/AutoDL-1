from automl_workflow.api import Learner, Predictor

from data_loader import MyDataLoader
import numpy as np

class MyPredictor(Predictor):

    def predict(self, dataset):
        data_loader = MyDataLoader()
        pt_dataloader = data_loader(dataset, train=False)

        
        for x, y in pt_dataloader:
            # y : [1, 10]
            n_examples = len(pt_dataloader)
            n_classes = y.shape[-1]
            break
        
        return np.zeros((n_examples, n_classes))


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
        