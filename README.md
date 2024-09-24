# HGDM
The code will coming soon!
>Heterogeneous Diffusion Graph Learning
![model](./HDL.jpg)
## Abstract
In recent years, research on graph-structured data modeling based
on Graph Neural Networks (GNNs) has achieved significant breakthroughs.
However, traditional GNNs are primarily designed for
homogenous graphs or specific bipartite graphs, limiting their ability
to accommodate the rich heterogeneity in structural information.
While recent studies have suggested capturing dependencies
between different types of heterogeneous interactions, few have
explored two critical challenges: 1) The presence of noisy data in
heterogeneous structures, which can contaminate the learning of
embeddings and adversely affect the performance of graph learning
task; 2) The sparse supervisory signals encountered in specific
predictive tasks, such as recommendation tasks. To address these
challenges, we propose a novel framework, termed Heterogeneous
Graph Diffusion Learning (HDL). Our framework incorporates a
simple yet effective latent space diffusion mechanism aimed at eliminating
noise in the representation space of compressed and dense
multi-type heterogeneous graph relations. We conducted extensive
experiments on real-world data to evaluate the efficacy of our
proposed framework. The results demonstrate that HDL exhibits
superiority in both recommendation tasks and node classification
tasks.
## Environment
- python=3.8
- torch=1.12.1
- numpy=1.23.1
- scipy=1.9.1
- dgl=1.0.2+cu113
