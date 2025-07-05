import numpy as np

def medianf(X, W):
    """
    Applies a 1D median filter to the input array X.

    Args:
        X (numpy.ndarray): The input array. If 1D, the filter is applied to it directly.
                           If 2D, the filter is applied to each row independently.
        W (int): The size of the median filter window.

    Returns:
        numpy.ndarray: The array after applying the median filter.
    """
    if X.ndim == 1:
        X_reshaped = X.reshape(1, -1)  # Reshape 1D array to a single row 2D array
        num_cols = X_reshaped.shape[1]
        num_rows = X_reshaped.shape[0]
        R = np.zeros_like(X_reshaped, dtype=np.float64)
        w2 = int(np.floor(W / 2))
        xx = np.pad(X_reshaped, ((0, 0), (w2, w2)), mode='constant')

        for c in range(num_cols):
            window = xx[:, c : c + W]
            median_values = np.median(window, axis=1)
            R[:, c] = median_values

        return R.flatten()  # Flatten the result back to 1D
    elif X.ndim == 2:
        num_cols = X.shape[1]
        num_rows = X.shape[0]
        R = np.zeros_like(X, dtype=np.float64)
        w2 = int(np.floor(W / 2))
        xx = np.pad(X, ((0, 0), (w2, w2)), mode='constant')

        for c in range(num_cols):
            window = xx[:, c : c + W]
            median_values = np.median(window, axis=1)
            R[:, c] = median_values
        return R
    else:
        raise ValueError("Input array must be 1D or 2D.")