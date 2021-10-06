import os

def test_main():
    os.system("python main/mnist.py --full=False")
    assert os.path.exists('main/cfm.png')
