# ACDNE
Adversarial Deep Network Embedding for Cross-Network Node Classification
====
This repository contains the author's implementation in Tensorflow for the paper "Adversarial Deep Network Embedding for Cross-Network Node Classifications".


Datasets
===
input/ contains the 5 datasets used in our paper.

Each ".mat" file stores a network dataset, where 
the variable "network" represents an adjacency matrix, 
the variable "attrb" represents a node attribute matrix,  
the variable "group" represents a node label matrix. 

Codes
===
"ACDNE_model.py" is the implementation of the ACDNE model.

"ACDNE_test_Blog.py" is an example case of the cross-network node classification task from Blog1 to Blog2 networks.

"ACDNE_test_citation.py" is an example case of the cross-network node classification task from citationv1 to dblpv7 networks.

Plese cite our paper as:
===
Xiao Shen, Quanyu Dai, Fu-lai Chung, Wei Lu, and Kup-Sze Choi. Adversarial Deep Network Embedding for Cross-Network Node Classification. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 2991-2999, 2020.
