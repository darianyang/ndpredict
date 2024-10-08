## NDPredict
NDPredict: ASN (N) Deamidation Predictor

Use this program to take a PDB file with ASN residues, and predict the probability that the ASN site will be deamidated.

To install:
* git clone this repo
``` Bash
git clone https://github.com/darianyang/ndpredict.git
```
* cd into the repo
``` Bash
cd ndpredict
```
* pip install directly
``` Python
pip install .
```
* eventually I may publish this on PyPI for easier installation

Requirements:
* numpy
* pandas
* matplotlib
* sklearn
* mdtraj
* biopython

Example Usage
--
For arg info:
``` Python
ndpredict --help
```
<p align="left">
    <img src="https://github.com/darianyang/ndpredict/blob/main/figures/help.png?raw=true" alt="ndpredict help" width="400">
</p>


To use on an example PDB file (note that first the PDB must be cleaned and ready for input into MDTraj).
``` Python
ndpredict -i pdb/1gb1_leap.pdb
```
The output will be a plot of each ASN residue of the input PDB (X axis) and the corresponding deamidation probability (Y axis):
<p align="left">
    <img src="https://github.com/darianyang/ndpredict/blob/main/figures/1gb1.png?raw=true" alt="1gb1 ndpredict" width="400">
</p>

Note that there is still some work to be done to make the feature calculation robust enough be able to handle any new PDB input. So not every PDB input will work at the current time.

References & Motivation
--
* using reference data and methodology inspiration from: Jia L, Sun Y (2017) Protein asparagine deamidation prediction based on structures with machine learning methods. PLoS ONE 12(7): e0181347

* reference paper uses Discovery Studio to calculate features, which is not open source or reproducible. Here, I am working on an fully automated feature calculation pipeline from an input PDB file using pure Python and I refined a few of their metrics.

## Copyright

Copyright (c) 2024, Darian Yang
