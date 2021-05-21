from automl_workflow.api import HPOptimizer

from architectures.resnet import ResNet9, ResNet18


class MyHPOptimizer(HPOptimizer):

    def fit(self, training_info):
        in_channels = training_info['dataset']['shape'][-1]
        num_class = training_info['dataset']['num_class']

        if training_info['loop']['epoch'] == 2 and training_info['need_switch']:
            return ResNet9(in_channels, num_class)
        else:
            return ResNet18(in_channels, num_class)