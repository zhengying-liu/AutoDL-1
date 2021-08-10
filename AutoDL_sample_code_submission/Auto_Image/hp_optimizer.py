from automl_workflow.api import HPOptimizer, Learner, ClassicLearner

from learner import MyLearner
from architectures.resnet import ResNet9, ResNet18

# from skeleton.projects.logic import LogicModel
from model_old import Model


class MyHPOptimizer(HPOptimizer):

    def __init__(self):
        pass

    def fit(self, train_info: dict) -> ClassicLearner:
        metadata = train_info['metadata']
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
        
        # Absorb `train_info` to keep track of training
        if hasattr(learner, 'train_info'):
            for key in train_info:
                learner.train_info[key] = train_info[key]
        else:
            learner.train_info = train_info

        return learner