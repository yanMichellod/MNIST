def main():

    import argparse

    example_doc = """\
examples:
    1. Runs the full analysis with the whole MNIST dataset:
       $ python mnist.py
       or
       $ python mnist.py --full=True
    2. Only runs for a subset of MNIST dataset (f.e. for quick test):
       $ python mnist.py --full=False
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
        nargs='*',
        default=["True", "False"],
        help="Decides if analysis have to take place at whole MNIST dataset. "
             "Options are %(default)s (default: %(default)s)",
        )

    args = parser.parse_args()

    from Analysis.Analysis import Analysis
    full = True
    if args.full[0] == "False":
        full = False
    print("Start analysis of MNIST dataset (full={0})...".format(full))
    an = Analysis(full=full)
    print("Is the CNN accuracy higher then the accuracy of the random forest: {0}".format(an.checkVSBaseline()))
    print("Is the CNN accuracy higher then 0.95 (95%): {0}".format(an.checkHypothesis(0.95)))
    print("Save confusion matrix: {0}".format(an.saveConfusionMatrix()))
    print("Analysis done!")


if __name__ == "__main__":
    main()
