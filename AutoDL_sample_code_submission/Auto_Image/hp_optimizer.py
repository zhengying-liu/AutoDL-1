from automl_workflow.api import HPOptimizer, Learner

from learner import MyLearner
from architectures.resnet import ResNet9, ResNet18

# from skeleton.projects.logic import LogicModel
from model_old import Model


class MyHPOptimizer(HPOptimizer):

    def __init__(self):
        pass

    def fit(self, training_info: dict) -> Learner:
        metadata = training_info['metadata']
        logic_model = Model(metadata)

        in_channels = logic_model.info['dataset']['shape'][-1]
        num_class = logic_model.info['dataset']['num_class']

        if logic_model.info['loop']['epoch'] == 2 and logic_model.info['need_switch']:
            backbone_model = ResNet9(in_channels, num_class)
        else:
            backbone_model = ResNet18(in_channels, num_class)

        # Create a learner and set its backbone model
        learner = MyLearner()
        learner.backbone_model = backbone_model
        
        # Absorb `training_info` to keep track of training
        if hasattr(learner, 'training_info'):
            for key in training_info:
                learner.training_info[key] = training_info[key]
        else:
            learner.training_info = training_info

        return learner