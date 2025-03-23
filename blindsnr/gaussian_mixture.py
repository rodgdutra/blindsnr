import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import soundfile as sf
from .utils import read_audio

def compute_log_power(signal, frame_length=128, hop_length=64):
    """
    Compute the log-power of a signal by framing it into overlapping windows.
    """
    # Pad the signal to ensure all frames have equal length
    padded_signal = np.pad(signal, (0, frame_length - len(signal) % frame_length), mode='constant')
    
    # Create overlapping frames
    num_frames = 1 + (len(padded_signal) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        padded_signal,
        shape=(num_frames, frame_length),
        strides=(padded_signal.strides[0] * hop_length, padded_signal.strides[0])
    )
    
    # Compute power and log-power
    power = np.mean(frames**2, axis=1)
    log_power = 10 * np.log10(power + 1e-10)  # Add small value to avoid log(0)
    return log_power

def fit_gmm_to_log_power(log_power, n_components=2):
    """
    Fit a Gaussian Mixture Model (GMM) to the log-power data.
    """
    # Reshape log_power to fit GMM input format
    log_power = log_power.reshape(-1, 1)
    
    # Fit GMM with 2 components
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(log_power)
    
    # Extract parameters
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    return means, variances, weights

def estimate_snr(means, variances, weights):
    """
    Estimate SNR based on the GMM parameters.
    Assume the component with the lower mean corresponds to noise.
    """
    # Identify noise and speech components
    noise_idx = np.argmin(means)
    speech_idx = 1 - noise_idx
    
    # Extract noise and speech parameters
    noise_mean, noise_var = means[noise_idx], variances[noise_idx]
    speech_mean, speech_var = means[speech_idx], variances[speech_idx]
    
    # Compute SNR
    snr = 10 * np.log10((speech_mean - noise_mean)**2 / (noise_var + speech_var))
    return snr

@read_audio
def gaussian_mixture_snr(noisy_signal, sample_rate=16000, frame_length=128, hop_length=64, plot=False):
    """
    Perform Gaussian Mixture Model (GMM) based SNR estimation of a noisy signal.
    """
    # Step 1: Compute log-power
    log_power = compute_log_power(noisy_signal)

    # Step 2: Fit GMM to log-power
    means, variances, weights = fit_gmm_to_log_power(log_power)

    # Step 3: Estimate SNR
    estimated_snr = estimate_snr(means, variances, weights)

    if plot:
        # Plot the log-power distribution and GMM fit
        x = np.linspace(np.min(log_power), np.max(log_power), 1000).reshape(-1, 1)
        gmm_pdf = np.sum([weights[i] * norm.pdf(x, means[i], np.sqrt(variances[i])) for i in range(len(means))], axis=0)
        
        plt.hist(log_power, bins=50, density=True, alpha=0.5, label="Log-Power Histogram")
        plt.plot(x, gmm_pdf, label="Fitted GMM", color="red")
        plt.title("GMM Fit to Log-Power Distribution")
        plt.xlabel("Log-Power")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
    
    return estimated_snr