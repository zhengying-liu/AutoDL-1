from automl_workflow.api import HPOptimizer, Learner

from learner import MyLearner
from architectures.resnet import ResNet9, ResNet18


class MyHPOptimizer(HPOptimizer):

    def fit(self, training_info: dict) -> Learner:
        in_channels = training_info['dataset']['shape'][-1]
        num_class = training_info['dataset']['num_class']

        if training_info['loop']['epoch'] == 2 and training_info['need_switch']:
            backbone_model = ResNet9(in_channels, num_class)
        else:
            backbone_model = ResNet18(in_channels, num_class)

        # Create a learner and set its backbone model
        learner = MyLearner()
        learner.backbone_model = backbone_model
        
        # Absorb `training_info` to keep track of training
        if learner.hasattr('training_info'):
            for key in training_info:
                learner.training_info[key] = training_info[key]
        else:
            learner.training_info = training_info

        return learner