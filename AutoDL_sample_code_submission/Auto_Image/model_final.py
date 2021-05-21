from learner import MyLearner


class Model():

    def __init__(self, metadata):
        self.learner = MyLearner()

    def train(self, dataset):
        self.predictor = self.learner.learn(dataset)

    def test(self, dataset):
        predictions = self.predictor.predict(dataset)
        return predictions