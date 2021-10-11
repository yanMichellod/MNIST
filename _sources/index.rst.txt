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
	
Hypothese
=========

It is not possible to achieve a classification accuracy for MNIST test set higher then 95%.

The hypothese can be rejected as the accuracy of the Random Forest is 97% and the CNN one is bigger than 98%!
	
How to deploy the project
=========================

There are two steps before using the projet.

1. Install the package : pip install git+https://github.com/yanMichellod/MNIST#egg=MNIST_Classification
2. Execute it : mnist (Have a look on help => mnist --help)


How to uninstall the package
============================

pip uninstall MNIST_Classification

We hope you enjoy to use your package :)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
