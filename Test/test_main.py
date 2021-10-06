import subprocess
    
def test_main():
    proc = subprocess.Popen("python main/mnist.py --full=False",
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    out,err = proc.communicate()
    assert proc.returncode == 0
    assert len(out) > 0
