from Analysis.Analysis import Analysis


def main():
    print("Start analysis of MNIST dataset...")
    an = Analysis()
    print("Is the CNN accuracy higher then the accuracy of the random forest: {0}".format(an.checkVSBaseline()))
    print("Is the CNN accuracy higher then 0.95 (95%): {0}".format(an.checkHypothesis(0.95)))
    print("Save confusion matrix: {0}".format(an.saveConfusionMatrix()))
    print("Analysis done!")
