import numpy
from classifieur import Classifier

class BayesNaifClassifier(Classifier):

    def __init__(self, **kwargs):
        print(kwargs)

    def train(self, train, train_labels):
        print("train")

    def predict(self, exemple, label):
        print("predict")

    def test(self, test, test_labels):
        print("test")


