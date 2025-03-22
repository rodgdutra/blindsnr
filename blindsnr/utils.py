import numpy as np

def generate_noisy_signal(desired_snr, signal, noise):
    """
    Generate a noisy signal with the desired SNR.

    Parameters:
    desired_snr: Desired Signal-to-Noise Ratio in dB
    signal: Numpy array of the original signal
    noise: Numpy array of the noise

    Returns:
    noisy_signal: Numpy array of the generated noisy signal
    """

    # Normalize the audio and noise
    signal = signal / np.max(np.abs(signal))
    noise = noise / np.max(np.abs(noise))

    # Repeat or cut the noise to match the length of the signal
    if len(noise) < len(signal):
        noise = np.tile(noise, (len(signal) + len(noise) - 1) // len(noise))[:len(signal)]
    else:
        noise = noise[:len(signal)]

    # Calculate the power of the signal and noise
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    g = np.sqrt(10.0 ** (-desired_snr/10) * signal_power / noise_power)
    
    # Assumes signal and noise to be decorrelated
    # and calculate (a, b) such that energy of 
    # a*signal + b*noise matches the energy of the input signal
    a = np.sqrt(1 / (1 + g**2))
    b = np.sqrt(g**2 / (1 + g**2))

    # # Compute the scaling factor for the noise to achieve the desired SNR
    # snr_linear = 10**(desired_snr / 10.0)
    # scale_factor = np.sqrt(snr_linear * noise_power / signal_power)

    # Generate the noisy signal
    noisy_signal = signal * a +  noise * b
    print("Desired SNR:", desired_snr, "dB")
    print("True SNR:", calculate_true_snr(signal * a, noise * b), "dB")
    return noisy_signal

def calculate_true_snr(signal_audio, noise_audio):
    """
    Calculate the TRUE SNR in dB using signal and noise audio.

    Parameters:
    signal_audio: numpy array of the signal audio
    noise_audio: numpy array of the noise audio

    Returns:
    snr_db: True SNR in dB
    """
    # Compute power of signal and noise
    signal_power = np.mean(signal_audio**2)
    noise_power = np.mean(noise_audio**2)

    # Calculate SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db
