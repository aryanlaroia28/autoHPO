
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
<img width="1280" alt="Screenshot 2024-06-13 at 8 43 55 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/848743a5-b889-441c-ac24-91d71c9d1745">
<img width="1246" alt="Screenshot 2024-06-13 at 8 44 44 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/343a6f34-fa83-4ea2-9b95-6eaa9050eb54">
<img width="846" alt="Screenshot 2024-06-13 at AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/63140cec-29af-442a-af7c-ed401904fbd7">
<img width="846" alt="Screenshot 2024-06-13 at AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/90b6b964-cefd-43fe-bcdd-e288a98051f0">

- **TPE Optimization**:
<img width="1284" alt="Screenshot 2024-06-13 at 8 47 15 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/c12a4c09-fb76-40c7-82a1-23ac673802a8">
<img width="1283" alt="Screenshot 2024-06-13 at 8 47 40 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/dde0447b-d951-4352-bcce-9c4232eb0496">
<img width="846" alt="Screenshot 2024-06-13 at 8 44 44 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/e9878c53-d3b3-4c7d-a4e9-5b180c60c74a">
<img width="846" alt="Screenshot 2024-06-13 at 8 44 44 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/f6f36c47-874f-4f45-8fcc-b7a258e0ba01">


- **PSO Optimization**:
<img width="1294" alt="Screenshot 2024-06-13 at 8 48 47 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/ef7de1ff-c098-48b4-aa59-ea19b8e782c9">
<img width="1276" alt="Screenshot 2024-06-13 at 8 49 13 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/8678e1be-16c1-407b-b9f5-7e6640f39ce8">
<img width="846" alt="Screenshot 2024-06-13 at 8 44 44 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/5606acf6-2f16-4218-b995-d6b51c0009ee">
<img width="846" alt="Screenshot 2024-06-13 at 8 44 44 AM" src="https://github.com/aryanlaroia28/auto-HPO/assets/166947111/7475ff2a-3846-41a2-8a9f-bd5bbae8a5ec">

## Conclusion:
Hence we can observe that our optimization algorithms provide competitive results with standard libraries like HyperOpt.
