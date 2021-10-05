from setuptools import find_packages, setup


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="MNIST_Classification",
    version="1.0.0",
    description="MNIST classification using Random Forest and neuronal network",
    url="https://github.com/yanMichellod/MNIST",
    license="MIT",
    author="Ralf Jandl & Yan Michellod",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={"console_scripts": ["mnist = main.mnist:main"]},
)
