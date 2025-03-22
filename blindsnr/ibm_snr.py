import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gammatone, stft, freqz
from scipy.fft import fft, fftfreq
from gammatone import gtgram
import librosa
import numpy as np
from scipy.signal import stft
from gammatone import gtgram
from .utils import calculate_true_snr
import soundfile as sf

def erb_to_hz(erb):
    """Convert ERB scale to Hz"""
    return (1000/ (2.302585093 * 4.37)) * np.exp(erb * (2.302585093 * 4.37)/1000) - 14675/4937

def compute_gammatone_erb(signal, sr, window_time=0.025, hop_time=0.01, channels=128, f_min=50, f_max=8000):
    """Compute gammatone filterbank cochleagram with ERB spacing."""
    # Compute Gammatone-based spectrogram
    gtg = gtgram.gtgram(signal, sr, window_time, hop_time, channels, f_min)
    cochleagram = np.abs(gtg) ** 2  # Energy calculation
    return cochleagram

def compute_gammatone_erb_scipy(signal, sr, window_time=0.025, hop_time=0.01, channels=128, f_min=50, f_max=8000):
    """Compute gammatone filterbank cochleagram with ERB spacing using scipy"""
    # Generate ERB-spaced center frequencies
    erb_min = 21.4 * np.log10(f_min/4.37 + 1)  # Convert Hz to ERB
    erb_max = 21.4 * np.log10(f_max/4.37 + 1)
    erb_centers = np.linspace(erb_min, erb_max, channels)
    f_centers = erb_to_hz(erb_centers)
    
    # Create time vector for filter response calculation
    n = 8192  # FFT size for frequency response calculation
    freqs = fftfreq(n, 1/sr)
    
    # Calculate aggregate frequency response
    total_response = np.zeros_like(freqs, dtype=float)
    filters = []
    
    for fc in f_centers:
        # Create gammatone filter
        b, a = gammatone(fc, 'fir', numtaps=1024, fs=sr)
        w, h = freqz(b, a, worN=n, fs=sr)
        
        # Get magnitude response
        mag = np.abs(h)
        freq_idx = np.abs(w - freqs).argmin(axis=0)
        mag_response = mag[freq_idx]
        
        # Accumulate total response
        total_response += mag_response**2
        filters.append((b, a, mag_response))
    
    # Calculate normalization factor
    freq_bins = np.linspace(0, sr/2, len(freqs)//2)
    norm_freqs = freq_bins[(freq_bins >= f_min) & (freq_bins <= f_max)]
    norm_response = np.sqrt(np.mean(total_response[(freqs >= f_min) & (freqs <= f_max)]))
    
    # Process signal with filterbank
    cochleagram = np.zeros((channels, len(signal)))
    for i, (b, a, _) in enumerate(filters):
        filtered = np.convolve(signal, b, mode='same')
        cochleagram[i, :] = filtered**2  # Energy calculation
    
    # Frame the signal
    frame_length = int(window_time * sr)
    hop_length = int(hop_time * sr)
    num_frames = 1 + (cochleagram.shape[1] - frame_length) // hop_length
    
    framed_energy = np.zeros((channels, num_frames))
    for t in range(num_frames):
        start = t * hop_length
        end = start + frame_length
        framed_energy[:, t] = np.mean(cochleagram[:, start:end], axis=1)
    
    # Apply normalization
    return framed_energy / (norm_response**2 + 1e-10)


def normalize_filterbank(cochleagram, signal):
    """Normalize filterbank energy to match time-domain energy."""
    time_domain_energy = np.sum(np.square(signal))
    filterbank_energy = np.sum(cochleagram)
    scale_factor = time_domain_energy / filterbank_energy
    return cochleagram * scale_factor


def compute_true_ibm(clean_signal, noise_signal, sr, frame_length=256, hop_length=128, num_filters=128):
    """
    Compute the true Ideal Binary Mask (IBM) for a noisy speech signal.
    
    Parameters:
        clean_signal (np.ndarray): Clean speech signal (1D array).
        noise_signal (np.ndarray): Noise signal (1D array).
        sr (int): Sampling rate of the signals.
        frame_length (int): Length of each frame in samples.
        hop_length (int): Hop size between frames in samples.
        num_filters (int): Number of frequency bands in the filterbank.
    
    Returns:
        ibm (np.ndarray): True Ideal Binary Mask of shape (num_filters, num_frames).
    """

    clean_coch = compute_gammatone_erb(clean_signal, sr)
    noise_coch = compute_gammatone_erb(noise_signal, sr)
    
    # Ensure both cochleagrams have the same shape
    min_frames = min(clean_coch.shape[1], noise_coch.shape[1])
    clean_coch = clean_coch[:, :min_frames]
    noise_coch = noise_coch[:, :min_frames]
    
    # Step 2: Compute the true IBM
    ibm = (clean_coch > noise_coch).astype(float)
    
    return ibm

def estimate_ibm_simple(cochleagram, noise_profile, snr_threshold_db=0):
    """Estimate Ideal Binary Mask using simple SNR thresholding."""
    # Calculate SNR in dB for each T-F unit
    signal_power = cochleagram - noise_profile[:, np.newaxis]  # Shape: (num_channels, num_frames)
    
    # Avoid division by zero by checking where noise_profile > 0
    snr_linear = np.where(
        noise_profile[:, np.newaxis] > 0,  # Condition, shape: (num_channels, 1)
        signal_power / noise_profile[:, np.newaxis],  # True case, shape: (num_channels, num_frames)
        0  # False case, scalar (broadcasted to shape of signal_power)
    )
    
    # Convert linear SNR to dB
    snr_db = 10 * np.log10(snr_linear + 1e-10)  # Add small constant to avoid log(0)
    
    # Create binary mask
    ibm = (snr_db > snr_threshold_db).astype(float)  # Shape: (num_channels, num_frames)
    
    return ibm

def estimate_noise_profile(cochleagram, percentile=10):
    """Estimate noise profile using lower percentile energy."""
    return np.percentile(cochleagram, percentile, axis=1)

def calculate_filtered_snr(cochleagram, ibm):
    """Calculate filtered SNR from cochleagram and IBM."""
    speech_energy = np.sum(cochleagram * ibm)
    noise_energy = np.sum(cochleagram * (1 - ibm))
    return 10 * np.log10(speech_energy / noise_energy) if noise_energy > 0 else np.nan

def snr_transform(filtered_snr, noise_profile):
    """Apply SNR transformation to convert filtered SNR to broadband SNR."""
    low_freq_mask = (np.arange(len(noise_profile)) < len(noise_profile)//4)
    low_freq_noise = np.sum(noise_profile[low_freq_mask])
    total_noise = np.sum(noise_profile)
    correction = 10 * np.log10(1 + (low_freq_noise / total_noise)) if total_noise > 0 else 0
    return filtered_snr + correction

def snr_ibm_estimator_generic(noisy_signal, estimate_ibm_func, sr=16000):
    """Estimate SNR using a generic  IBM approach. This function is designed to be flexible a
    nd can use different IBM estimation methods."""
    # Step 1: Compute Gammatone cochleagram
    cochleagram = compute_gammatone_erb(noisy_signal, sr)
    
    # Step 2: Normalize the filterbank coefficients
    normalized_coch = normalize_filterbank(cochleagram, noisy_signal)

    # Step 3: Estimate noise profile
    noise_profile = estimate_noise_profile(normalized_coch)
    print("Noise profile: ", noise_profile.shape)
    
    # Step 4: Estimate IBM
    ibm_estimated = estimate_ibm_func(normalized_coch, noise_profile)
    
    # Step 4: Calculate filtered SNR
    filtered_estimated_snr = calculate_filtered_snr(normalized_coch, ibm_estimated)    
    
    # Step 5: Apply SNR transformation
    broadband_estimated_snr = snr_transform(filtered_estimated_snr, noise_profile)
    
    # Return the estimated SNR
    return broadband_estimated_snr

def simple_ibm_snr_estimator(noisy_signal, sr=16000):
    """
    Simple IBM SNR estimator using the simple IBM estimation
    
    Args:
        noisy_signal: numpy array of the noisy signal
        sr: sampling rate of the signal
        
    Returns:
        broadband_estimated_snr: estimated broadband SNR in dB
    """
    if type(noisy_signal) == str:
        noisy_signal = sf.read(noisy_signal)[0]
    broadband_estimated_snr = snr_ibm_estimator_generic(noisy_signal, estimate_ibm_simple, sr)
    return broadband_estimated_snr

def true_ibm_snr_estimator(signal, noise, sr=16000):
    """
    Calculate the SNR using the true Ideal Binary Mask (IBM), using the clean signal and noise
    one is able to generate the true IBM mask, and then use it to calculate the SNR. 
    This is a reference method for comparison to other IBM based methods.
    
    Args:
        signal: numpy array of the clean signal
        noise: numpy array of the noise signal
        sr: sampling rate of the signal and noise
        
    Returns:
        broadband_estimated_snr: estimated broadband SNR in dB
    """
    # Step 1: Compose noisy signal
    noisy_signal = signal + noise
    
    # Step 2: Compute Gammatone cochleagram
    cochleagram = compute_gammatone_erb(noisy_signal, sr)
    
    # Step 3: Normalize the filterbank coefficients
    normalized_coch = normalize_filterbank(cochleagram)

    # Step 4: Estimate noise profile
    noise_profile = estimate_noise_profile(normalized_coch)
    
    # Step 5: Estimate IBM
    ibm = compute_true_ibm(signal, noise, sr)
    
    # Step 6: Calculate filtered SNR
    filtered_estimated_snr = calculate_filtered_snr(normalized_coch, ibm)    
    
    # Step 7: Apply SNR transformation
    broadband_estimated_snr = snr_transform(filtered_estimated_snr, noise_profile)
    
    # Return the estimated SNR
    return broadband_estimated_snr
    
        
# Example usage
if __name__ == "__main__":
    # Load example audio (replace with your noisy speech file)
    signal, sr = librosa.load(librosa.example('trumpet'), sr=16000)
    print(f"Signal length: {len(signal)} samples")
    print(f"Sample rate: {sr} Hz")
    noise = np.random.normal(0, 0.1, len(signal))
    noisy_signal = signal + noise
    
    # Step 1: Compute Gammatone cochleagram
    cochleagram = compute_gammatone_erb(noisy_signal, sr)
    cochleagram_signal = compute_gammatone_erb(signal, sr)
    cochleagram_signal_half = compute_gammatone_erb(signal[:len(signal)//2], sr)
    
    
    print("cochleagram computed: ", cochleagram.shape)
    normalized_coch = normalize_filterbank(cochleagram)
    print("normalized cochleagram  computed: ", normalized_coch.shape)
    ibm = compute_true_ibm(signal, noise, 16000)
    
    # Step 2: Estimate noise profile
    noise_profile = estimate_noise_profile(normalized_coch)
    print("Noise profile: ", noise_profile.shape)
    
    # Step 3: Estimate IBM
    ibm_estimated = estimate_ibm_simple(normalized_coch, noise_profile)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(4,1,1)
    plt.imshow(10*np.log10(normalized_coch + 1e-10), aspect='auto', origin='lower')
    plt.title('Cochleagram  noisy signal(dB)')
    plt.subplot(4,1,2)
    plt.imshow(10*np.log10(cochleagram_signal + 1e-10), aspect='auto', origin='lower')
    plt.title('Cochleagram Signal (dB)')
    plt.subplot(4,1,3)
    plt.imshow(ibm, aspect='auto', origin='lower', cmap='gray')
    plt.title('True IBM')
    plt.subplot(4,1,4)
    plt.imshow(ibm_estimated, aspect='auto', origin='lower', cmap='gray')
    plt.title('Estimated IBM')
    plt.tight_layout()
    plt.show()
    
    # Step 4: Calculate filtered SNR
    filtered_estimated_snr = calculate_filtered_snr(normalized_coch, ibm_estimated)
    filtered_snr = calculate_filtered_snr(normalized_coch, ibm)
    
    
    # Step 5: Apply SNR transformation
    broadband_estimated_snr = snr_transform(filtered_estimated_snr, noise_profile)
    broadband_snr = snr_transform(filtered_snr, noise_profile)
    
    
    print(f"Filtered SNR: {filtered_snr:.2f} dB")
    print(f"Broadband SNR: {broadband_snr:.2f} dB")
        
    print(f"Filtered estimated SNR: {filtered_estimated_snr:.2f} dB")
    print(f"Estimated Broadband SNR: {broadband_estimated_snr:.2f} dB")
    print(f"True SNR: {calculate_true_snr(signal, noise)} dB")
    
    
    # # Visualization
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2,1,1)
    # plt.imshow(10*np.log10(normalized_coch + 1e-10), aspect='auto', origin='lower')
    # plt.title('Cochleagram (dB)')
    # plt.subplot(2,1,2)
    # plt.imshow(ibm, aspect='auto', origin='lower', cmap='gray')
    # plt.title('Estimated IBM')
    # plt.tight_layout()
    # plt.show()