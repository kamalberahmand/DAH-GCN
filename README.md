# DAH-GCN

Implementation of "Dual Adaptive Higher-Order Graph Convolutional Network for Attributed Graph Clustering" published in IEEE TKDE 2025.

## Overview

This repository contains the PyTorch implementation of DAH-GCN, a novel approach for attributed graph clustering that processes structural and attribute information through dual views with adaptive higher-order propagation.

## Key Features

- Dual-view architecture for structure and attribute processing
- Adaptive higher-order graph propagation
- End-to-end clustering optimization
- Support for multiple benchmark datasets

## Requirements
torch>=1.12.0
torch-geometric>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0

## Installation
git clone https://github.com/YOUR_USERNAME/DAH-GCN.git
cd DAH-GCN
pip install -r requirements.txt

## Quick Start
python train.py --dataset cora --epochs 200

## Results

Performance on benchmark datasets:

| Dataset  | NMI   | ARI   | ACC   |
|----------|-------|-------|-------|
| Cora     | 0.716 | 0.691 | 0.832 |
| Citeseer | 0.441 | 0.471 | 0.696 |
| PubMed   | 0.451 | 0.432 | 0.778 |

## Citation

If you find this code useful, please cite:
@article{berahmand2025dahgcn,
title={DAH-GCN: Dual Adaptive Higher-Order Graph Convolutional Network for Attributed Graph Clustering},
author={Berahmand, Kamal and Mohammadi, Mehrnoush and Khosravi, Hassan and Jalili, Mahdi},
journal={IEEE Transactions on Knowledge and Data Engineering},
year={2025}
}

## Contact

Kamal Berahmand - kamal.berahmand@rmit.edu.au

## License

MIT License
