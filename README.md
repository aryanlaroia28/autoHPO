
# Automated Hyperparameter Optimization (HPO) System

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Bayesian Optimization](#bayesian-optimization)
  - [Particle Swarm Optimization (PSO)](#particle-swarm-optimization-pso)
  - [Tree-structured Parzen Estimator (TPE)](#tree-structured-parzen-estimator-tpe)
- [Evaluation](#evaluation)
- [Results](#results)

## Introduction
This project focuses on developing an automated hyperparameter optimization (HPO) system using AutoML techniques. The goal is to efficiently identify the best hyperparameter configuration for a given machine learning model and dataset, enhancing the performance of ML models without manual intervention.

## Problem Statement
The quality of performance of a Machine Learning model heavily depends on its hyperparameter settings. Given a dataset and a task, the choice of the machine learning (ML) model and its hyperparameters is typically performed manually, so we need to automate the process.

### Specifics
- Integrated with various machine learning models and coded to handle different data types.
- Employed efficient AutoML techniques like Bayesian optimization, particle swarm optimization, and TPE (Tree-Parzen Estimator) for HPO.
- ROC AUC, cross-validation and comparison of learning rate distribution curves with respect to hyperopt to check viability.

### Resources
- [Kaggle Notebook on Automated Model Tuning](https://www.kaggle.com/code/willkoehrsen/automated-model-tuning/notebook)
- [AutoML HPO Overview](https://www.automl.org/hpo-overview)

## Project Structure
The project consists of several Jupyter notebooks and related documents:
- `HPO(Bayesian).ipynb`: Implementation of HPO using Bayesian Optimization.
- `HPO(PSO).ipynb`: Implementation of HPO using Particle Swarm Optimization.
- `HPO(TPE).ipynb`: Implementation of HPO using Tree-structured Parzen Estimator.
- `requirements.txt`: Pip installations required for the project

## Installation
To run the notebooks and reproduce the results, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/aryanlaroia28/autoHPO.git
   cd auto-HPO
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Jupyter Notebook if not already installed:
   ```bash
   pip install jupyter
   ```

## Usage
To run the Jupyter notebooks, navigate to the project directory and start Jupyter Notebook:
```bash
jupyter notebook
```

Open the desired notebook (`HPO(Bayesian).ipynb`, `HPO(PSO).ipynb`, or `HPO(TPE).ipynb`) and run the cells sequentially to execute the code.

## Methodology
### Bayesian Optimization
Bayesian Optimization is a probabilistic model-based optimization technique. It builds a surrogate probabilistic model of the objective function and uses it to select the most promising hyperparameters to evaluate in the true objective function.

**Steps:**
1. **Surrogate Model**: Build a probabilistic model using Gaussian Process of the objective function.
2. **Acquisition Function**: Use the model to select the next set of hyperparameters to evaluate by optimizing an acquisition function.
3. **Evaluation**: Evaluate the selected hyperparameters on the actual objective function.
4. **Update**: Update the surrogate model with the new evaluation results.

The notebook `HPO(Bayesian).ipynb` implements this technique.

### Particle Swarm Optimization (PSO)
PSO is a computational method inspired by social behavior of birds flocking or fish schooling. It optimizes a problem by iteratively trying to improve a candidate solution with respect to a given measure of quality.

**Steps:**
1. **Initialization**: Initialize a swarm of particles with random positions and velocities.
2. **Evaluation**: Evaluate the fitness of each particle.
3. **Update Velocities and Positions**: Update velocities and positions based on personal and global best positions.
4. **Iteration**: Repeat the evaluation and update steps until convergence.

The notebook `HPO(PSO).ipynb` provides an implementation of PSO for hyperparameter optimization.

### Tree-structured Parzen Estimator (TPE)
TPE is an approach to sequential model-based optimization. It models the distribution of good and bad hyperparameters and uses this model to select new hyperparameters.

**Steps:**
1. **Modeling**: Model the objective function using two densities: one for good hyperparameters and one for bad.
2. **Sampling**: Sample new hyperparameters by maximizing the ratio of the two densities.
3. **Evaluation**: Evaluate the new hyperparameters on the actual objective function.
4. **Update**: Update the densities with the new evaluation results.

The implementation is found in the notebook `HPO(TPE).ipynb`.

## Evaluation
The evaluation involves:
- **Cross-validation**: Evaluated the model performance using 5-fold cross validation, comparing results with Hyperopt and default hyperparamater settings.
- **ROC-AUC Scores**: ROC-AUC score of optimised model, hyperopt and default models were compared.
- **Comparison of learning rate distribution curves**: Compare the learning rate distribution curves of Hyperopt Models and my optimised models.

This comprehensive analysis ensures a thorough understanding of the effectiveness of each HPO technique.

## Results
- **Bayesian Optimization**:
<img width="1240" alt="Screenshot 2024-06-14 at 9 27 35 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/78a8f1e1-243b-480a-bb0f-8f9dabb9baf4">
<img width="1231" alt="Screenshot 2024-06-14 at 9 28 12 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/6ddecb11-3a97-4f3b-9102-3d6755fcbc4e">
<img width="691" alt="Screenshot 2024-06-14 at 9 28 42 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/96c077dd-695c-4be9-90dd-adeff31b64aa">
<img width="596" alt="Screenshot 2024-06-14 at 9 29 04 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/322b84ef-2ac8-413f-83bb-7947aa79d411">


- **TPE Optimization**:
<img width="1232" alt="Screenshot 2024-06-14 at 9 30 38 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/181023f4-ab90-44d3-9f54-a74d5923f684">
<img width="1227" alt="Screenshot 2024-06-14 at 9 31 00 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/57c377c2-fa35-46b9-b798-90e44530d140">
<img width="607" alt="Screenshot 2024-06-14 at 9 31 21 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/d77e24d4-7eed-4fd6-8cc2-963415fa6c32">
<img width="684" alt="Screenshot 2024-06-14 at 9 31 41 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/68099b56-e0b3-4fa1-bb74-26c17774ce28">


- **PSO Optimization**:
<img width="1197" alt="Screenshot 2024-06-14 at 9 32 56 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/bab6cbf5-e396-432a-8242-5495e7935af1">
<img width="1277" alt="Screenshot 2024-06-14 at 9 33 18 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/a7a2ebfe-0dd3-46cb-bea8-085b02c41471">
<img width="578" alt="Screenshot 2024-06-14 at 9 33 34 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/2967d5c0-1166-4eeb-aa87-62b731bce0e6">
<img width="687" alt="Screenshot 2024-06-14 at 9 33 57 AM" src="https://github.com/aryanlaroia28/autoHPO/assets/166947111/cec7cb94-9c32-40b3-ac05-3a37cc330abc">

## Conclusion:
Hence we can observe that our optimization algorithms provide competitive results with standard libraries like HyperOpt.
