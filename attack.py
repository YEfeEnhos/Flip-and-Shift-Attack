import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def flip_and_shift(X_sub, y_sub, epsilon, gamma_min, gamma_max, n, mode='over', shift=3):
    
    model = LinearRegression()
    model.fit(X_sub, y_sub)
    
    m = len(y_sub)
    n_poison = int(np.ceil(epsilon * m))
    n_flip = int(np.ceil((shift*n-shift*epsilon*m)/(m*epsilon*gamma_max-m*shift)))
    n_line = n_poison - n_flip  # Number of points to lie on the line
    
    # Adjust shift based on mode
    if mode == 'under':
        shift = -abs(shift)
    else:
        shift = abs(shift)

    # Step 1: Identify the range of x-values
    x_min = np.min(X_sub)

    x_max = np.max(X_sub)
    interval_length = (x_max - x_min) / n_flip

    # Step 2: Determine indices to flip based on intervals
    flip_indices = []
    for i in range(n_flip):
        x_value = x_min + (i + 0.5) * interval_length  # Select the middle of each interval
        closest_index = np.argmin(np.abs(X_sub.flatten() - x_value))
        flip_indices.append(closest_index)
    
    flip_indices = np.array(flip_indices)

    # Step 3: Flip values
    y_poisoned = np.copy(y_sub)
    for i in range(len(flip_indices)):
        if mode == 'over':
            if i > 0:
    
                y_poisoned[flip_indices[i]] =  (y_poisoned[flip_indices[i-1]] + (X_sub[flip_indices[i]]-X_sub[flip_indices[i-1]])*model.coef_)

            else: 
                y_poisoned[flip_indices[i]] =  (gamma_max + (X_sub[flip_indices[i]]-x_min)*model.coef_)

        else:
            y_poisoned[i] = gamma_min


    remaining_indices = np.setdiff1d(np.arange(m), flip_indices)
    remaining_indices_sorted = remaining_indices[np.argsort(X_sub[remaining_indices].flatten())]
    
    interval = len(remaining_indices_sorted) // n_line
    selected_indices = remaining_indices_sorted[::interval][:n_line]

    for i in range(len(selected_indices)):
        if i > 0:

            y_poisoned[selected_indices[i]] = y_poisoned[selected_indices[i-1]] + (X_sub[selected_indices[i]]-X_sub[selected_indices[i-1]])*model.coef_
        else: 

            y_poisoned[selected_indices[i]] = shift + (X_sub[selected_indices[i]])*model.coef_ + model.intercept_


    return X_sub, y_poisoned, flip_indices
