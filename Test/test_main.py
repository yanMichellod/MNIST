import subprocess
    
def test_main():
    """Run the cli command to test the whole analysis in test mode (full=False)
    """
    proc = subprocess.Popen(["python3", "main/mnist.py"],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    out,err = proc.communicate()
    assert proc.returncode == 0
    assert len(out) > 0
