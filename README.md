# EEG-Based Stroke Severity Classification using DWT-MSAF-Mamba

Implementation code for the paper "EEG-Based Level of Stroke Classification
Using Multi-Scale Attention Fusion and Mamba" (Fifin Agustiana, Esmeralda
Contessa Djamal, Wina Witanti — Universitas Jenderal Achmad Yani).

## Overview
This repository contains the training and evaluation pipeline for a
stroke severity classification model that integrates Discrete Wavelet
Transform (DWT), Multi-Scale Attention Fusion (MSAF), and Mamba for
EEG-based three-class classification (No Stroke, Minor Stroke,
Moderate Stroke).

## Requirements
Install dependencies with:

pip install -r requirements.txt

## Dataset
This code expects EEG data as 5-second segments (640 samples x 14
channels) stored as CSV files, organized as:

DATASET_SEGMEN5/
├── NoStroke/
├── MinorStroke/
└── ModerateStroke/

The dataset used in this study is not publicly redistributed due to
patient data privacy and ethical approval terms. It is available from
the corresponding author upon reasonable request — see the Data and
Code Availability Statement in the manuscript.

## Usage
Update the `DATA_PATH` variable in `train.py`, then run:

python train.py

## Citation
If you use this code, please cite the associated paper (citation
details to be added upon publication).