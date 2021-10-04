.. MNIST documentation master file, created by
   sphinx-quickstart on Mon Oct  4 11:18:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MNIST's documentation!
=================================

This project concerns classification of MNIST Database. 

Two different algorithms are used to achieve the goal, a random forest (RF) and a convolutionnal neuronal netword (CNN)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

	modules
	
How to deploy the project
=========================

There are two possibilities to deploy the project.

1. Git clone
	* Git clone : git clone https://github.com/yanMichellod/MNIST.git
	* Create environement in the MNIST folder : conda env create -f envs/MNIST.yml
	* Activate the environement : conda activate MNIST
	* Run python file : python Analysis/analysis.py


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
