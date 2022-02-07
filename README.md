# Distributed practice in League of Legends
This repository contains the analysis code and results for our paper Mind the Gap: Distributed practice enhances performance in a MOBA game. Our analysis tests the effects of clustering versus spacing play instances on performance in League of Legends. We iterate on previous work that has investigated the effects of distributed practice - a known effect in the psychology of learning - using observational game telemetry data.

## Usage
This code repository allows interested readers (we hope these exist!) to reproduce our results. Clone this repository on your machine. Then, run `main.ipynb` to reproduce all figures and statistics found in the paper main body. Running `appendix_1.ipynb` will produce all auxiliary results concerning normalised MMR trends, found in the appendices. Running `appendix_2.ipynb` will produce auxiliary results concerning secondary clustering techniques, also found in the paper appendices.

## Dependencies
All dependencies are listed in the top level file `requirements.txt`. 

We recommend installing specified package versions to avoid conflicts. We also recommend installing packages in a virtualenv (unless a system-wide installation of dependencies is desired).
