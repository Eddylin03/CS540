import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename):
    data = pd.read_csv(filename)
    return data

def normalize_data(years):
    m = years.min()
    M = years.max()
    x_normalized = (years - m) / (M - m)
    X_normalized = np.column_stack((x_normalized, np.ones(len(x_normalized))))
    return X_normalized, m, M

def compute_closed_form_solution(X, Y):
    XtX = X.T.dot(X)
    XtX_inv = np.linalg.inv(XtX)
    XtY = X.T.dot(Y)
    weights = XtX_inv.dot(XtY)
    return weights

def gradient_descent(X, Y, learning_rate, iterations):
    n_samples = X.shape[0]
    weights = np.zeros(2)  # Initialize weights [w, b] to zeros
    losses = []
    weights_history = []

    for t in range(iterations):
        predictions = X.dot(weights)
        errors = predictions - Y
        gradient = (1 / n_samples) * X.T.dot(errors)
        weights -= learning_rate * gradient
        loss = (1 / (2 * n_samples)) * np.sum(errors ** 2)
        losses.append(loss)

        if (t + 1) % 10 == 0 and t < iterations - 1:
            weights_history.append(weights.copy())

    return weights, losses, weights_history

def plot_data(years, days):
    plt.figure()
    plt.plot(years, days, 'o')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig('data_plot.jpg')
    plt.close()

def plot_loss(losses):
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.jpg')
    plt.close()

def predict(year, weights, m, M):
    x_normalized = (year - m) / (M - m)
    y_hat = weights[0] * x_normalized + weights[1]
    return y_hat

if __name__ == "__main__":
    # Parse command-line arguments
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])

    # Load data
    data = load_data(filename)
    years = data['year'].values
    days = data['days'].values

    # Question 2: Plot data
    plot_data(years, days)

    # Question 3: Data normalization
    X_normalized, m, M = normalize_data(years)
    print("Q3:")
    print(X_normalized)

    # Question 4: Closed-form solution
    weights_closed_form = compute_closed_form_solution(X_normalized, days)
    print("Q4:")
    print(weights_closed_form)

    # Question 5: Gradient descent
    weights_gd, losses, weights_history = gradient_descent(X_normalized, days, learning_rate, iterations)

    # Q5a: Print weights every 10 iterations
    print("Q5a:")
    print(np.array([0., 0.]))  # Initial weights

    for weights in weights_history:
        print(weights)


    # Q5b and Q5c: Print learning rate and iterations
    print("Q5b:", 1)
    print("Q5c:", 160)

    # Q5d: Plot loss over time
    plot_loss(losses)


    # Question 6: Prediction for 2023
    y_hat_2023 = predict(2023, weights_closed_form, m, M)
    print("Q6: " + str(y_hat_2023))

    # Question 7: Model interpretation
    w = weights_closed_form[0]
    if w > 0:
        symbol = ">"
    elif w < 0:
        symbol = "<"
    else:
        symbol = "="
    print("Q7a: " + symbol)

    explanation = ("A positive weight indicates that the number of ice days is increasing over time. "
                   "A negative weight indicates that the number of ice days is decreasing over time. "
                   "A zero weight indicates no trend over time.")
    print("Q7b: " + explanation)

    # Question 8: Model limitations
    x_star_normalized = -weights_closed_form[1] / weights_closed_form[0]
    x_star = x_star_normalized * (M - m) + m
    print("Q8a: " + str(x_star))
    
    analysis = ("The prediction of 1812.873 is not compelling because it suggests the lake stopped freezing before our recorded data begins, which contradicts the actual observations. Limitations include the extremely small dataset size, the assumption of a linear relationship over time, and the model's inability to account for natural climate variability and external environmental factors.")
    print("Q8b: " + analysis)
