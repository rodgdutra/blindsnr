import numpy as np
from scipy.signal import convolve, medfilt
import matplotlib.pyplot as plt
from .utils.audio_utils import read_audio


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

def locate_extremum(h, from_idx, to_idx, extremum_type):
    """
    Locates an extremum (peak or trough) in a histogram.

    Args:
        h (numpy.ndarray): The histogram.
        from_idx (int): The starting index for the search.
        to_idx (int): The ending index for the search.
        extremum_type (int): 0 for peak, 1 for trough.

    Returns:
        int: The index of the located extremum, or to_idx if not found.
    """
    PEAK = 0
    TROUGH = 1
    PEAK_WIDTH = 3

    for i in range(from_idx + PEAK_WIDTH, to_idx - PEAK_WIDTH):
        if h[i] == 0:  # not interested in extrema at 0
            continue

        is_extremum = True  # assume it's an extremum to begin with
        pre_swing = 0
        post_swing = 0
        swing_loc = i - PEAK_WIDTH

        # check the preceding samples
        for j in range(i - PEAK_WIDTH, i):
            if extremum_type == PEAK:
                if h[j] > h[j + 1]:
                    is_extremum = False
                    break
                if h[j] != h[j + 1]:
                    pre_swing = 1
            else:  # extremum_type == TROUGH
                if h[j] < h[j + 1]:
                    is_extremum = False
                    break
                if h[j] != h[j + 1]:
                    pre_swing = 1

        if not is_extremum:
            continue

        # check the subsequent samples
        for j in range(i, i + PEAK_WIDTH):
            if extremum_type == PEAK:
                if h[j] < h[j + 1]:
                    is_extremum = False
                    break
                if h[j] != h[j + 1]:
                    post_swing = 1
            else:  # extremum_type == TROUGH
                if h[j] > h[j + 1]:
                    is_extremum = False
                    break
                if h[j] != h[j + 1]:
                    post_swing = 1

        # check to make sure it isn't a step function
        # this kind of check is necessary if the peak is wider than the window
        if (pre_swing + post_swing <= 1) and is_extremum:
            k = i
            while k > from_idx:
                diff1 = h[k - 1] - h[k]
                if diff1 != 0:
                    break
                k -= 1
            swing_loc = k

            k = i
            while k < to_idx - 1:
                diff2 = h[k] - h[k + 1]
                if diff2 != 0:
                    break
                k += 1
            next_swing_loc = k

            if (extremum_type == PEAK and (diff1 > 0 or diff2 < 0)):
                continue  # no dice
            if (extremum_type == TROUGH and (diff1 < 0 or diff2 > 0)):
                continue  # ditto

            # otherwise, the peak is at the mid-point of this plateau
            return round((swing_loc + next_swing_loc) / 2)

        if is_extremum:
            return i

    return to_idx

def pick_center(h, bin_index, low=-28.125, high=96.875, bins=500):
    """
    Calculates the center value of a histogram bin.

    Args:
        h (numpy.ndarray): The histogram (not directly used here, but kept for consistency).
        bin_index (int): The index of the bin.
        low (float): The lower bound of the histogram range.
        high (float): The upper bound of the histogram range.
        bins (int): The number of bins in the histogram.

    Returns:
        float: The center value of the bin.
    """
    step = (high - low) / bins
    center = low + step * (bin_index + 0.5)
    print(center)
    return center

def percentile_hist(h, bins, percent, low=-28.125, high=96.875):
    """
    Calculates the value at a given percentile of a histogram.

    Args:
        h (numpy.ndarray): The histogram.
        bins (int): The number of bins.
        percent (float): The percentile (between 0 and 1).
        low (float): The lower bound of the histogram range.
        high (float): The upper bound of the histogram range.

    Returns:
        float: The value at the given percentile.
    """
    cumulative_sum = np.cumsum(h)
    bin_index = np.where(cumulative_sum >= percent * cumulative_sum[-1])[0][0]
    print("max hist: ", max(h))
    print("bin_index: ", bin_index)
    print("cumulative_sum: ", cumulative_sum[-1])
    
    return pick_center(h, bin_index, low, high, bins)

@read_audio
def nist_stnr_m(noisy_signal, sample_rate=16000, doplot=0, verbose=0):
    """
    Calculate NIST STNR actually in Python, attempting to duplicate
    stnr -c algorithm.
    
    The original implemantation can be found at https://labrosa.ee.columbia.edu/projects/snreval/

    Args:
        D (numpy.ndarray): The input audio signal.
        SR (int): The sampling rate.
        doplot (int, optional): Flag to enable plotting. Defaults to 0.

    Returns:
        float: The calculated STNR value.
    """
    # constants from snr.h
    MILI_SEC = 20.0
    PEAK_LEVEL = 0.95

    BINS = 500
    SMOOTH_BINS = 15
    CODEC_SMOOTH_BINS = 15
    LOW = -28.125
    HIGH = 96.875

    BLOCKSIZE = 2048

    # from hist.h
    PEAK = 0
    TROUGH = 1
    PEAK_WIDTH = 3

    # algorithm is to form a power histogram over 20ms windows

    frame_width = int(sample_rate / 1000 * MILI_SEC)
    frame_adv = int(frame_width / 2)

    print(max(noisy_signal))

    # calculate power, assuming short samples
    D2 = (noisy_signal * 16384) ** 2
    print(max(D2))

    nhops = int(len(D2) / frame_adv)
    D2_reshaped = D2[:nhops * frame_adv].reshape(nhops, frame_adv).T
    # Power in each half window
    P2 = np.mean(D2_reshaped, axis=0)
    # Power in overlapping windows
    Pdb = 10 * np.log10(np.convolve(P2, np.ones(2), 'valid'))

    # Histogram
    hvals = np.linspace(LOW, HIGH, BINS + 1)
    power_hist, _ = np.histogram(Pdb, bins=hvals)
    if verbose:
        print(" power hist max: ", max(power_hist))
        print(" power shape ", power_hist.shape)
        
    

    # stnr -c algorithm
    unspiked_hist = medianf(power_hist, 3)
    codec_smooth_kernel = np.ones(CODEC_SMOOTH_BINS * 2 + 1) / (CODEC_SMOOTH_BINS * 2 + 1)
    presmooth_hist = convolve(unspiked_hist, codec_smooth_kernel, mode='same')
    smooth_kernel = np.ones(SMOOTH_BINS * 2 + 1) / (SMOOTH_BINS * 2 + 1)
    smoothed_hist = convolve(presmooth_hist, smooth_kernel, mode='same')

    if doplot:
        plt.figure(figsize=(10, 8))
        plt.subplot(411)
        plt.plot(hvals[:-1], power_hist)
        plt.title('power_hist')
        plt.subplot(412)
        plt.plot(hvals[:-1], unspiked_hist)
        plt.title('unspiked_hist')
        plt.subplot(413)
        plt.plot(hvals[:-1], presmooth_hist)
        plt.title('presmooth_hist')
        plt.subplot(414)
        plt.plot(hvals[:-1], smoothed_hist)
        plt.title('smoothed_hist')
        plt.tight_layout()
        plt.show()

    # assume to begin with that we don't find any extrema */
    first_peak = BINS
    first_trough = BINS
    second_peak = BINS
    second_trough = BINS

    max_val = np.max(smoothed_hist)

    # now look for the extrema, sequentially */

    # find the noise peak; it should be reasonably big */
    starting_point = 0
    first_peak = locate_extremum(smoothed_hist, starting_point, BINS, PEAK)
    while (10 * smoothed_hist[first_peak]) < max_val and starting_point < BINS:
        starting_point = first_peak + 1
        first_peak = locate_extremum(smoothed_hist, starting_point, BINS, PEAK)
        if first_peak == BINS and starting_point >= BINS - 1: # Avoid infinite loop
            break

    # now find the rest */
    if first_peak < BINS:
        first_trough = locate_extremum(smoothed_hist, first_peak + 1, BINS, TROUGH)
        if first_trough < BINS:
            second_peak = locate_extremum(smoothed_hist, first_trough + 1, BINS, PEAK)
            if second_peak < BINS:
                second_trough = locate_extremum(smoothed_hist, second_peak + 1, BINS, TROUGH)

    if verbose:
        print(
            'peak=%d (%5.2f) trough=%d (%5.2f) peak=%d (%5.2f) trough=%d (%5.2f)' % (
                first_peak,
                pick_center(smoothed_hist, first_peak, LOW, HIGH, BINS) if first_peak <= BINS else np.nan,
                first_trough,
                pick_center(smoothed_hist, first_trough, LOW, HIGH, BINS) if first_trough <= BINS else np.nan,
                second_peak,
                pick_center(smoothed_hist, second_peak, LOW, HIGH, BINS) if second_peak <= BINS else np.nan,
                second_trough,
                pick_center(smoothed_hist, second_trough, LOW, HIGH, BINS) if second_trough <= BINS else np.nan
            )
        )

    if first_peak == BINS:
        if verbose:
            print("I can't find the first peak of the power distribution. Is this a null file?")
        return np.nan

    noise_lvl = pick_center(smoothed_hist, first_peak, LOW, HIGH, BINS)

    if first_trough == BINS:
        if verbose:
            print("Can't find first trough. I'll do my best from here...")

        temp_unspiked_hist = np.copy(unspiked_hist)

        cross_lvl = -np.inf
        speech_lvl = percentile_hist(temp_unspiked_hist, BINS, PEAK_LEVEL, LOW, HIGH)
        if verbose:
            print("speech_lvl ",speech_lvl)
            print("noise_lvl", noise_lvl)
        
        S = speech_lvl - noise_lvl
        return S

    temp_unspiked_hist = np.copy(unspiked_hist)
    for i in range(first_trough):
        temp_unspiked_hist[i] = 0

    if second_peak == BINS:
        if verbose:
            print("Can't find second peak.")

        cross_lvl = -np.inf
        speech_lvl = percentile_hist(temp_unspiked_hist, BINS, PEAK_LEVEL, LOW, HIGH)
        S = speech_lvl - noise_lvl
        return S

    # check for bogus hump */
    if 60 * (smoothed_hist[second_peak] - smoothed_hist[first_trough]) < smoothed_hist[first_peak]:
        cross_lvl = -np.inf
    else:
        cross_lvl = pick_center(smoothed_hist, second_peak, LOW, HIGH, BINS)

    if second_trough == BINS:
        second_lim = second_peak
    else:
        second_lim = second_trough

    for i in range(second_lim):
        temp_unspiked_hist[i] = 0

    speech_lvl = percentile_hist(temp_unspiked_hist, BINS, PEAK_LEVEL, LOW, HIGH)

    S = speech_lvl - noise_lvl
    return S