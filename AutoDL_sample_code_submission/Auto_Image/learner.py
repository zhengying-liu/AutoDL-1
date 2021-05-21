from automl_workflow.api import Learner, Predictor


class MyLearner(Learner):

    # TODO: define components such as BackboneModel, DataIngestor, DataLoader,
    #   Optimizer, HPOptimizer. Potentially in a separate file for each 
    #   component.
    def __init__(self, metadata):
        self.training_info = {}     # Initilize training info using `metadata`

    
    def learn(self, train_set) -> Predictor:
        # Update training info, if necessary
        pass
        self.backbone_model = self.hp_optimizer.fit(self.training_info)
        