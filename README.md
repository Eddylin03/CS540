# CS540 Introduction to Artificial Intelligence

This repository contains my coursework for CS540: Introduction to Artificial Intelligence at the University of Wisconsin-Madison, Fall 2024.

## Course Information

* **Course Description:**  Gain principles of knowledge-based search techniques; automatic deduction, knowledge representation using predicate logic, machine learning, probabilistic reasoning. Develop applications in tasks such as problem solving, data mining, game playing, natural language understanding, and robotics.
* **Credits:** 3
* **Prerequisites:** (COMP SCI 300 or 367) and (MATH 211, 217, 221, or 275) or graduate/professional standing or declared in the Capstone Certificate in Computer Sciences for Professionals.
* **Sections:**
    * Section 1: MW 4:00 - 5:15, 145 Birge Hall
    * Section 2: TR 2:30 - 3:45, 19 Ingraham Hall
    * Section 3: TR 1:00 - 2:15, 5206 Sewell Social Sciences
* **Textbook (Optional):** Artificial Intelligence: A Modern Approach (4th edition) by Stuart Russell and Peter Norvig. Pearson, 2020. ISBN 978-0134610993.

## Course Objectives

* Understand and apply foundational tools in Machine Learning and Artificial Intelligence: Linear algebra, Probability, Logic, and elements of Statistics.
* Understand core techniques in Natural Language Processing (NLP), including bag-of-words, tf-idf, n-Gram Models, and Smoothing.
* Understand the basics of Machine Learning, including supervised and unsupervised learning.
* Distinguish between regression and classification, and understand basic algorithms: Linear Regression, k-Nearest Neighbors, and Naive Bayes.
* Understand the basics of Neural Networks: Network Architecture, Training, Backpropagation, Stochastic Gradient Descent.
* Learn aspects of Deep Learning, including network architectures, convolution, training techniques.
* Understand the fundamentals of Game Theory.
* Understand how to formulate and solve several types of Search problems.
* Understand basic elements of Reinforcement Learning.
* Consider the applications and ethics of Artificial Intelligence and Machine Learning in real-world settings.


## Projects

* **Letter Frequency Analysis:** https://github.com/Eddylin03/CS540/blob/main/hw2/hw2.py - This script analyzes the frequency of letters in a given text file and uses Bayesian reasoning to determine if the text is more likely to be in English or Spanish. 
* **Facial Recognition with Eigenfaces:** https://github.com/Eddylin03/CS540/blob/main/hw3/Demonstration.ipynb - Implemented a facial recognition system using Principal Component Analysis (PCA) to identify individuals from a dataset of facial images.
* **Hierarchical Clustering of Socioeconomic Data:** https://github.com/Eddylin03/CS540/blob/main/hw4/hw4.py - Applied hierarchical agglomerative clustering (HAC) to analyze socioeconomic indicators from various countries, visualizing country clusters based on socioeconomic profiles.
* **Linear Regression on Lake Mendota Ice:** https://github.com/Eddylin03/CS540/blob/main/hw5/hw5.py - Implemented a linear regression model to analyze historical ice coverage days on Lake Mendota. Tasks included dataset curation, data normalization, closed-form solution calculation, gradient descent optimization, and prediction for future ice coverage based on trends.
* **Neural Network with PyTorch:** https://github.com/Eddylin03/CS540/blob/main/hw6/intro_pytorch.py - Developed a basic neural network in PyTorch to classify images from the FashionMNIST dataset. Implemented data loading, model building, training, evaluation, and individual image prediction to familiarize with PyTorch's capabilities.
* **LeNet on MiniPlaces Dataset:** https://github.com/Eddylin03/CS540/blob/main/hw7/demo.ipynb - Implemented the LeNet-5 convolutional neural network architecture to perform scene recognition on the MiniPlaces dataset. Built a custom CNN model with two convolutional layers and three fully connected layers, explored various hyperparameter configurations including batch size and learning rate, and analyzed model performance across different training settings.

* **A* Search for N-Tile Puzzle:** https://github.com/Eddylin03/CS540/blob/main/hw8/Demo.py - Developed an A* search implementation to solve variants of the 8-tile sliding puzzle with multiple empty spaces. Implemented Manhattan distance heuristic for state evaluation, dynamic successor state generation, and efficient priority queue-based pathfinding. The solution handles various puzzle configurations (6-tile, 7-tile, etc.) and includes comprehensive state validation.
* **[...and so on...]** 


## Technologies Used 

* Python
* NumPy - For numerical operations, particularly useful for matrix manipulations in linear regression and gradient descent.
* Matplotlib - Used for visualizing data plots, loss plots, and other figures in assignments.
* PyTorch - Utilized for building, training, and evaluating neural networks in deep learning assignments.
* Pandas - Employed for data handling and cleaning, especially in preprocessing datasets.
* Torchvision - Part of PyTorch, used to load and transform image data (like FashionMNIST) for neural network training.
* heapq - Python's built-in heap queue algorithm implementation, used for priority queue in A* search puzzle solving.
* tqdm - Progress bar library for monitoring training progress in neural network implementations.
* shadcn/ui - React component library utilized for building modern user interfaces in web-based demonstrations.
* torch.nn - PyTorch's neural network modules, essential for implementing LeNet architecture.
* torch.optim - PyTorch's optimization algorithms for training neural networks with different learning rates.


## Notes

* This repository will be updated throughout the semester as I complete assignments and projects.
* Feel free to reach out if you have any questions!
