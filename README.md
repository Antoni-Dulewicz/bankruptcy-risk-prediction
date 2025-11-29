# Ensemble-Based Bankruptcy Risk Prediction with Gradient Boosting (LightGBM)

This repository contains a predictive modeling project for assessing corporate bankruptcy risk using financial reports. The goal is to develop and evaluate models that can reliably predict bankruptcy, particularly in highly imbalanced datasets.

## Overview

- **Objective:** Design a model to predict corporate bankruptcy based on financial indicators.
- **Approach:** Comparative study of ensemble classifiers:
  - **Bagging**
  - **Random Forests**
  - **Gradient Boosting (LightGBM)**
- **Focus:** Examine performance under severe class imbalance and identify architectures resilient to minority-class scarcity.

## Data Handling

- **Missing Values:** Handled using mean imputation.  
- **Class Imbalance:** Addressed with stratified sampling to ensure proportional representation in training and test sets.

## Key Features

- Implementation of baseline classifiers for benchmarking.
- Ensemble methods with a focus on variance reduction and bias control.
- LightGBM implementation for gradient boosting with hyperparameter tuning.
- Evaluation using AUROC and other relevant metrics for imbalanced classification tasks.

## Usage

1. Load and preprocess the dataset.
2. Train baseline models (e.g., decision trees) for reference.
3. Train ensemble models including Random Forests and LightGBM.
4. Evaluate and compare model performance using AUROC, precision, recall, and F1-score.
5. Optionally perform hyperparameter tuning to improve boosting performance.
