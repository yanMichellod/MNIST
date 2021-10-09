import argparse
import sys


def main(args=None):

    if not args:
        args = sys.argv[1:]

    example_doc = """\
examples:
    1. Runs the full analysis with the whole MNIST dataset:
       $ mnist --full=True
    2. Only runs for a subset of MNIST dataset (f.e. for quick test):
       $ mnist.py
       or
       $ mnist --full=False
    """

    parser = argparse.ArgumentParser(
        usage="python %(prog)s [options]",
        description="Performs CNN on MNIST dataset and checks the accuracy",
        epilog=example_doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-f",
        "--full",
        choices=["True", "False"],
        nargs="*",
        default=["False"],
        help="Decides if analysis have to take place at whole MNIST dataset. "
        "Options are %(default)s (default: %(default)s)",
    )

    args = parser.parse_args(args)

    full = args.full[0] == "True"

    print("Start analysis of MNIST dataset (full={0})...".format(full))
    from Analysis.Analysis import Analysis

    an = Analysis(full=full)
    print("*************************************************")
    print("Analyse results")
    print("")
    print("Accuracy RF: %.3f" % (an.getRFAccuracy() * 100.0))
    print("Accuracy CNN: %.3f" % (an.getCNNAccuracy() * 100.0))
    print(
        "Is the CNN accuracy higher then the accuracy of the random forest: {0}".format(
            an.checkVSBaseline()
        )
    )
    print(
        "Is the CNN accuracy higher then 0.95 (95%): {0}".format(
            an.checkHypothesis(0.95)
        )
    )
    print("Save confusion matrix: {0}".format(an.saveConfusionMatrix()))
    print("Analysis done!")
    print("*************************************************")
    return True
