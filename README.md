# Risk_Score_Prediction for Human Due Diligence

PyTorch and PyTorch Lightning Implementations of Risk Score Prediction for Human Due Diligence.

## Abstract
Human due diligence is a crucial assessment that should be provided to client companies before they invest in another company or hire individuals. This assessment serves several purposes, among which are the following (and many more):
- Uncovering capability gaps
- Analyzing points of friction
- Finding out differences in decision making

## Dependencies
- PyTorch
- PyTorch Lightning
- Torchvision
- Scikit-learn

## Models
This repository contains 3 separate models for risk score prediction:
- PyTorch Lightning implementation of a MLP
- PyTorch implementation of a MLP + Ensemble Learning 
- PyTorch Lightning implementation of a MLP + Ensemble Learning

## Dataset
The training dataset is in .csv format.
- 2,000 training data
- Each data contains:
  - Name
  - Risk score (1 to 5)
  - 23 features (Degrees, Work Experience, Political Activities, Social Media Activities, etc.)

## Experiment
- Perfomred data preprocessing (e.g. converted Yes/No boolean features to integer boolean representation).
- Perfomred feature selection by removing features with an absolute correlation with the corresponding risk score below a threshold (0.001).
- Performed 80%/10%/10% Train Validation Test Split.
- Conducted risk score prediction using 3 different prediction models.
- Applied Gaussian noise as a data augmentation technique.
- Tuned hyperparameters (batch size, learning rate, epochs, hidden dim & number of models for ensemble learning) - heuristic.

## Experimental Results
- Achieved 85 ~ 88% prediction accuracy on test data.
