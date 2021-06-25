# This script should be shared by different strategies/participants

from automl_workflow.api import Learner, Predictor, Dataset

from data_loader import MyDataLoader
import numpy as np

from model_old import Model


class MyPredictor(Predictor):

    def predict(self, dataset: Dataset):
        data_loader = MyDataLoader()
        pt_dataloader = data_loader(dataset, train=False)

        
        for x, y in pt_dataloader:
            # y : [1, 10]
            n_examples = len(pt_dataloader)
            n_classes = y.shape[-1]
            break
        
        return np.zeros((n_examples, n_classes))


class MyLearner(Learner):

    def __init__(self, 
        backbone_model=None,
        data_loader=None,
        data_ingestor=None,
        data_augmentor=None,
        ensembler=None,
        optimizer=None,
        loss_func=None,
        hp_optimizer=None,
        train_info=None,
        ):
        self.backbone_model = backbone_model
        self.data_loader = data_loader
        self.data_ingestor = data_ingestor
        self.data_augmentor = data_augmentor
        self.ensembler = ensembler
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.hp_optimizer = hp_optimizer
        self.train_info = train_info

    
    def learn(self, train_set) -> Predictor:
        # All hyper-parameters are set for a specific learner (e.g. we can call 
        # sklearn.some_model.fit() directly
        classic_learner = self.hp_optimizer.fit(dataset)

        # In this sense, we cannot do something like
        # while True:
        #     classic_learner = specific_learner.hp_optimizer.fit(dataset)

        # Now we use the specific learner to learn
        predictor = classic_learner.learn(dataset)

        return predictor
        