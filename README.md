# Machine Learning
This repo holds all programming assignments completed for my Machine Learning course (Fall 2022).

# Assignment Descriptions

## A1 --- Probability, MLE+MAP Estimations and the EM Algorithm
Includes probability proofs, PMF derivations, MLE, MAP and Bayes parameter estimation calculations, EM algorithm derivation and implementation and a high dimensional hypercube proof.

- **Code**
	- [`a1_Prob_MLE+MAP_EM/a1_EM-Algorithm.py`](https://github.com/luke-davidson/MachineLearning/blob/main/a1_Prob_MLE%2BMAP_EM/a1_EM-Algorithm.py)
		- Implementation of the EM algorithm.
	- [`a1_Prob_MLE+MAP_EM/a1_Hypercube.py`](https://github.com/luke-davidson/MachineLearning/blob/main/a1_Prob_MLE%2BMAP_EM/a1_Hypercube.py)
		- Generating points in a high dimensional hypercube and hypersphere and proving the performed derivation in the report.
- **Report:** [`a1_Prob_MLE+MAP_EM/a1_report.pdf`](https://github.com/luke-davidson/MachineLearning/blob/main/a1_Prob_MLE%2BMAP_EM/a1_report.pdf)

## A2 --- Bayes Theorem Implementation + Gradient Descent
Implementation of the perceptron algorithm, Naive Bayes classifier, basis functions, optimal decision surface derivation, linear regression gradient descent derivations. 

- **Code** 
	- [`a2_Bayes_GradDescent/a2_NaiveBayesClassifier.py`](https://github.com/luke-davidson/MachineLearning/blob/main/a2_Bayes_GradDescent/a2_NaiveBayesClassifier.py)
		- Implementation of a Naive Bayes Classifier with binary classifications. Calculates class priors, posteriors and final class assignments for both low and high dimensional data points.
	- [`a2_Bayes_GradDescent/a2_GradDescent.py`](https://github.com/luke-davidson/MachineLearning/blob/main/a2_Bayes_GradDescent/a2_GradDescent.py)
		- Implementation of gradient descent derivations for minimizing sum of squared errors and sum of squared distances.
	- [`a2_Bayes_GradDescent/a2_HyperplaneAccuracy.py`](https://github.com/luke-davidson/MachineLearning/blob/main/a2_Bayes_GradDescent/a2_HyperplaneAccuracy.py)
		- Implementation of the perceptron algorithm as it relates to hyperplane accuracy.
- **Report:** [`a2_Bayes_GradDescent/a2_report.pdf`](https://github.com/luke-davidson/MachineLearning/blob/main/a2_Bayes_GradDescent/a2_report.pdf)

## A3 --- Project Proposal
Copy of my semester project proposal. See [TimeSeriesMotionClassification](https://github.com/luke-davidson/TimeSeriesMotionClassification) for whole project.

- **Report:** [`a3_ProjectProposal/a3_ProjectProposal.pdf`](https://github.com/luke-davidson/MachineLearning/blob/main/a3_ProjectProposal/a3_ProjectProposal.pdf)

## A4 --- Neural Networks + Performance Evaluation
Implementation of differently sized Neural Networks, matrix factorization, the Alternating Least Squares algorithm and representational bias in neural network applications. 

- **Code:** [`a4_NeuralNetworks_ROC/a4_NeuralNetworks.py`](https://github.com/luke-davidson/MachineLearning/blob/main/a4_NeuralNetworks_ROC/a4_NeuralNetworks.py)
	- Test data is generated based on decision regions (defined in self.bounds) and is assigned a class based on probabilities (ex. 98% will be correctly labeled, 2% will be incorrectly labeled). Neural networks of various sizes are then created, trained and tested on the generated data. Performance of differently sized neural nets is then evaluated.
- **Report:** [`a4_NeuralNetworks_ROC/a4_report.pdf`](https://github.com/luke-davidson/MachineLearning/blob/main/a4_NeuralNetworks_ROC/a4_report.pdf)