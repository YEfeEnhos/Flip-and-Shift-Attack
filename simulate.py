import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from part2_attack import flip_and_shift  

def trim_defense(X_train, y_train, poisoned_indices, epsilon, sizesubset, true_slope, true_intercept, X_test, y_test, max_iter=3, tol=1e-4):
    n = len(y_train)
    subset_size = int(n - sizesubset * epsilon)
    
    # Initialize a random subset of indices
    indices = np.random.choice(n, subset_size, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]

    # Fit the initial model
    model = LinearRegression()
    model.fit(X_subset, y_subset)
    prev_loss = mean_squared_error(y_train, model.predict(X_train))
    
    # Collect initial intercept for the final plot title
    initial_intercept = model.intercept_
    
    # Collect iteration plots for the first three iterations
    fig, axs = plt.subplots(4, 2, figsize=(20, 30))

    for i in range(max_iter):
        # Compute the loss for each point in the training set
        predictions = model.predict(X_train)
        losses = (predictions - y_train) ** 2

        # Select the subset of points with the smallest losses
        selected_indices = np.argsort(losses)[:subset_size]
        X_subset = X_train[selected_indices]
        y_subset = y_train[selected_indices]

        # Refit the model on the new subset
        model.fit(X_subset, y_subset)
        current_loss = mean_squared_error(y_train, model.predict(X_train))
        current_intercept = model.intercept_

        # Calculate and print the percentage of cleared points that are poisoned
        cleared_indices = np.setdiff1d(np.arange(n), selected_indices)
        cleared_poisoned_count = np.sum([i in poisoned_indices for i in cleared_indices])
        cleared_fraction_poisoned = cleared_poisoned_count / len(cleared_indices) * 100
        print(f"Iteration {i + 1}: Percentage of cleared points that are poisoned = {cleared_fraction_poisoned:.2f}%")

        # Plot the data and the regression lines before and after ignoring certain points
        ax_left = axs[i, 0]
        ax_right = axs[i, 1]

        ax_left.scatter(X_train, y_train, color='blue', label='All Data')
        ax_left.scatter(X_subset, y_subset, color='green', label='Cleaned Data')
        ax_left.plot(X_train, model.predict(X_train), color='orange', linestyle='--', label='Regression Line After Cleaning')
        
        full_model = LinearRegression()
        full_model.fit(X_train, y_train)
        ax_left.plot(X_train, full_model.predict(X_train), color='red', linestyle='--', label='Original Regression Line')

        # Plot the true line
        ax_left.plot(X_train, true_slope * X_train + true_intercept, color='black', label='True Line')

        ax_left.set_xlabel('X')
        ax_left.set_ylabel('y')
        ax_left.legend()
        ax_left.set_title(f'Iteration {i + 1}')
        ax_left.text(0.5, -.085, f"PIPP = {cleared_fraction_poisoned:.2f}%, y-intercept = {current_intercept:.2f}", 
                     size=12, ha="center", transform=ax_left.transAxes)

        # Plot the zoomed version
        ax_right.scatter(X_train, y_train, color='blue', label='All Data')
        ax_right.scatter(X_subset, y_subset, color='green', label='Cleaned Data')
        ax_right.plot(X_train, model.predict(X_train), color='orange', linestyle='--', label='Regression Line After Cleaning')
        
        full_model = LinearRegression()
        full_model.fit(X_train, y_train)
        ax_right.plot(X_train, full_model.predict(X_train), color='red', linestyle='--', label='Original Regression Line')

        # Plot the true line
        ax_right.plot(X_train, true_slope * X_train + true_intercept, color='black', label='True Line')

        ax_right.set_xlim(9.7, 10)
        ax_right.set_ylim(25, 26.4)
        ax_right.set_xlabel('X')
        ax_right.set_ylabel('y')
        ax_right.legend()
        ax_right.set_title(f'Iteration {i + 1} (Zoomed)')

        # Check for convergence
        if np.abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    # Final plot with all data points
    ax_left = axs[3, 0]
    ax_right = axs[3, 1]

    ax_left.scatter(X_train, y_train, color='blue', label='Original Data That has Flipped')
    ax_left.scatter(X_train[:sizesubset], y_poisoned, color='red', label='Flipped Data')
    ax_left.scatter(X_subset, y_subset, color='green', label='Cleaned Data')
    ax_left.scatter(X_test, y_test, color='purple', marker='x', label='Test Data')
    ax_left.plot(X_train, true_slope * X_train + true_intercept, color='black', label='True Line')

    # Plot the regression line from the cleaned data
    x_range = np.linspace(0, 10, 100)
    ax_left.plot(x_range, model.coef_[0] * x_range + model.intercept_, color='orange', linestyle='--', label='Cleaned Regression Line')

    ax_left.set_xlabel('X')
    ax_left.set_ylabel('y')
    ax_left.legend()
    ax_left.set_title('Final')
    ax_left.text(0.5, -0.085, f"PIPP = {cleared_fraction_poisoned:.2f}%, y-intercept = {current_intercept:.2f}, Initial y-intercept = 1", 
                     size=12, ha="center", transform=ax_left.transAxes)

    # Plot the zoomed version
    ax_right.scatter(X_train, y_train, color='blue', label='Original Data That has Flipped')
    ax_right.scatter(X_train[:sizesubset], y_poisoned, color='red', label='Flipped Data')
    ax_right.scatter(X_subset, y_subset, color='green', label='Cleaned Data')
    ax_right.scatter(X_test, y_test, color='purple', marker='x', label='Test Data')
    ax_right.plot(X_train, true_slope * X_train + true_intercept, color='black', label='True Line')

    # Plot the regression line from the cleaned data
    ax_right.plot(x_range, model.coef_[0] * x_range + model.intercept_, color='orange', linestyle='--', label='Cleaned Regression Line')

    ax_right.set_xlim(9.7, 10)
    ax_right.set_ylim(25, 26.4)
    ax_right.set_xlabel('X')
    ax_right.set_ylabel('y')
    ax_right.legend()
    ax_right.set_title('Final (Zoomed)')

    plt.tight_layout()
    # plt.savefig("trim_defense_iterations.png")
    plt.show()

    # Return the cleaned dataset
    return X_train[selected_indices], y_train[selected_indices]

# Example usage
if __name__ == "__main__":
    sizesubset = 100
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.rand(1200, 1) * 10  # 100 points in the range [0, 10]
    true_slope = 2.5
    true_intercept = 1.0
    y = true_slope * X.flatten() + true_intercept + np.random.randn(1200) * 2  # Linear relation with noise

    # Create a substitute dataset (e.g., first 50 points)
    X_sub = X[:sizesubset]
    y_sub = y[:sizesubset]

    # Parameters for the attack
    epsilon = 0.5
    gamma_min = min(y_sub)
    gamma_max = max(y_sub)
    
    # Perform the flip attack on the substitute dataset
    X_poisoned, y_poisoned, poisoned_indices = flip_and_shift(X_sub, y_sub, epsilon, gamma_min, gamma_max, n=1200)

    # Combine the poisoned subset with the rest of the dataset
    X_train = np.vstack((X_poisoned, X[sizesubset:]))
    y_train = np.concatenate((y_poisoned, y[sizesubset:]))

    # Get indices of the poisoned points
    poisoned_indices = np.arange(len(X_sub))

    # Generate test data
    X_test = np.linspace(0, 10, 20).reshape(-1, 1)
    y_test = true_slope * X_test.flatten() + true_intercept + np.random.randn(20) * 2

    # Perform the Trim defense
    X_clean, y_clean = trim_defense(X_train, y_train, poisoned_indices, epsilon, sizesubset, true_slope, true_intercept, X_test, y_test)

    # Fit a linear regression model to the cleaned data
    model = LinearRegression()
    model.fit(X_clean, y_clean)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Output the model coefficients
    print("Model coefficients:", model.coef_)
    print("Model intercept:", model.intercept_)

    # Calculate and print the accuracy (mean squared error) on the test data
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error on Test Data:", mse)

    # Plot the original data, poisoned data, cleaned data, test data, and regression lines
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:sizesubset], y[:sizesubset], color='blue', label='Original Data That has Flipped')
    plt.scatter(X[:sizesubset], y_poisoned, color='red', label='Flipped Data')
    plt.scatter(X_clean, y_clean, color='green', label='Cleaned Data')
    plt.scatter(X_test, y_test, color='purple', marker='x', label='Test Data')
    plt.plot(X[:sizesubset], true_slope * X[:sizesubset] + true_intercept, color='black', label='True Line')

    # Plot the regression line from the cleaned data
    x_range = np.linspace(0, 10, 100)
    plt.plot(x_range, model.coef_[0] * x_range + model.intercept_, color='orange', linestyle='--', label='Cleaned Regression Line')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Final')
    # plt.savefig("trim_defense_summary.png")
    plt.show()
