
Adversarial Deep Network Embedding for Cross-Network Node Classification (ACDNE)
====
This repository contains the author's implementation in Tensorflow for the paper "Adversarial Deep Network Embedding for Cross-Network Node Classifications".

Environment Requirement
===
The code has been tested running under Python 3.6.2. The required packages are as follows:

•	python == 3.6.2

•	tensorflow == 1.13.1

•	numpy == 1.16.2

•	scipy == 1.2.1

•	sklearn == 0.21.1


Datasets
===
input/ contains the 5 datasets used in our paper.

Each ".mat" file stores a network dataset, where

the variable "network" represents an adjacency matrix, 

the variable "attrb" represents a node attribute matrix,

the variable "group" represents a node label matrix. 

Code
===
"ACDNE_model.py" is the implementation of the ACDNE model.

"ACDNE_test_Blog.py" is an example case of the cross-network node classification task from Blog1 to Blog2 networks.

"ACDNE_test_citation.py" is an example case of the cross-network node classification task from citationv1 to dblpv7 networks.

Plese cite our paper as:
===
Xiao Shen, Quanyu Dai, Fu-lai Chung, Wei Lu, and Kup-Sze Choi. Adversarial Deep Network Embedding for Cross-Network Node Classification. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 2991-2999, 2020.

Pytorch Implementation of ACDNE can be found at:
===

