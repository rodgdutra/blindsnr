import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from .utils.matlab_to_numpy import medianf
from .utils.audio_utils import read_audio
from .utils.logging_utils  import logger

@read_audio
def guess_vad(noisy_signal, sample_rate=16000, tsm=0.25):
    """
    Estimate voice activity times from a waveform.
    Return T as a set of [voice_start voice_end] rows, in seconds.

    Args:
        D (numpy.ndarray or str): The input audio waveform (numpy array)
                                  or path to a WAV file (str).
        SR (int): The sampling rate of the audio.
        tsm (float, optional): Median smoothing time in seconds. Defaults to 0.25 s.

    Returns:
        numpy.ndarray: A 2D array where each row is [voice_start, voice_end] in seconds.
    """
    # Choose FFT size; 256 for sr = 16000
    # MATLAB: nfft = 2^round(log(256 * SR/16000)/log(2));
    # Python: Equivalent using numpy's log2 and power
    nfft = int(2**np.round(np.log2(256 * sample_rate / 16000)))

    # Define hop length for spectrogram (50% overlap, as in MATLAB's specgram default)
    hop_length = nfft // 2

    # Make spectrogram (magnitude spectrogram in dB)
    # librosa.stft returns complex STFT; need to take absolute value for magnitude
    S_complex = librosa.stft(y=noisy_signal, n_fft=nfft, hop_length=hop_length, window='hann')
    DC = 20 * np.log10(np.abs(S_complex) + 1e-10) # Add a small epsilon to avoid log(0)

    # Frame rate for spectrogram
    fr = sample_rate / hop_length

    # Take energy in voice region 100..1000 Hz
    vxfmin = 100
    vxfmax = 1000
    # Convert frequency to bin indices (MATLAB uses 1-based, Python 0-based)
    leb = int(np.round(vxfmin * nfft / sample_rate))
    ueb = int(np.round(vxfmax * nfft / sample_rate))

    # Ensure leb and ueb are within valid bounds
    leb = max(0, leb)
    ueb = min(DC.shape[0] - 1, ueb)

    # Voicing energy as largest bin in that range (max across frequency bins, axis=0)
    # Adjust for 0-based indexing: DC[freq_bin_idx, time_frame_idx]
    if leb > ueb: # Handle cases where frequency range might be invalid
        DCE = np.full(DC.shape[1], -np.inf)
    else:
        DCE = np.max(DC[leb:ueb + 1, :], axis=0) # +1 because slicing is exclusive at end

    # Remove -Infs, or anything more than 50 dB below peak
    # np.max(DCE) handles potential -inf correctly by ignoring them if any finite values exist
    max_dce_val = np.max(DCE[np.isfinite(DCE)]) if np.isfinite(DCE).any() else -np.inf
    lowthresh = max_dce_val - 50.0
    DCE[DCE < lowthresh] = lowthresh

    # Then threshold as 1/3rd way between 10th and 90th percentile, in dB
    # np.percentile takes values between 0 and 100 for the percentile argument
    p10 = np.percentile(DCE, 10)
    p90 = np.percentile(DCE, 90)
    thr = 0.33 * p10 + 0.67 * p90

    # Median smooth over tsm seconds
    # odd-length median filter window
    # MATLAB: medwin = 1+2*round(tsm/2*SR/(nfft/2));
    medwin = 1 + 2 * int(np.round(tsm / 2 * sample_rate / hop_length))
    # Ensure medwin is odd for median filter. If it's even, make it odd by adding 1.
    if medwin % 2 == 0:
        medwin += 1

    # VAD estimate
    vad = (DCE > thr).astype(int) # Convert boolean to integer (0 or 1)

    # Blur out 1 bin each way (effectively expanding active regions)
    # MATLAB: vad = max([vad;[0,vad(1:end-1)];[vad(2:end),0]]);
    # Python equivalent using np.roll and np.maximum
    vad_shifted_left = np.roll(vad, 1) # Shift right by 1, fills 0 at the start
    vad_shifted_left[0] = 0 # Explicitly set first element to 0
    vad_shifted_right = np.roll(vad, -1) # Shift left by 1, fills 0 at the end
    vad_shifted_right[-1] = 0 # Explicitly set last element to 0
    vad = np.maximum(vad, vad_shifted_left)
    vad = np.maximum(vad, vad_shifted_right)

    # Median filter
    vad = medianf(vad, medwin) # Using the custom medianf function for consistency
    # Convert back to boolean for logical operations if needed later
    vad = (vad > 0.5).astype(int) # Thresholding after median filter to get 0/1 values

    # Debug plot
    debug = 0
    if debug:
        plt.figure(figsize=(12, 8))

        plt.subplot(211)
        # librosa.display.specshow is great for spectrograms
        librosa.display.specshow(DC, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram (dB)')
        plt.tight_layout()

        plt.subplot(212)
        tt = np.arange(len(DCE)) * hop_length / sample_rate # Time axis for DCE
        plt.plot(tt, DCE, '-b', label='Voicing Energy')
        plt.plot([tt[0], tt[-1]], [thr, thr], '-r', label=f'Threshold={thr:.2f} dB')
        plt.plot([tt[0], tt[-1]], [p10, p10], ':g', label=f'10th Percentile={p10:.2f} dB')
        plt.plot([tt[0], tt[-1]], [p90, p90], ':r', label=f'90th Percentile={p90:.2f} dB')
        plt.plot(tt, vad * np.max(DCE) / 2, 'k', label='VAD (scaled)') # Scale VAD for visibility
        plt.title('Voicing Energy and VAD')
        plt.xlabel('Time (s)')
        plt.ylabel('dB')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Convert to times
    tt = np.arange(len(DCE)) / fr # Time points for each frame center

    # Find transitions from non-voice to voice (voiceon) and voice to non-voice (voiceoff)
    # Pad vad with zeros at start/end for boundary conditions
    padded_vad = np.concatenate(([0], vad, [0]))

    # Voice onset: where (current frame is 0 AND next frame is 1)
    voiceon_indices = np.where((padded_vad[:-1] == 0) & (padded_vad[1:] == 1))[0]
    # Voice offset: where (current frame is 1 AND next frame is 0)
    voiceoff_indices = np.where((padded_vad[:-1] == 1) & (padded_vad[1:] == 0))[0]

    # Adjust indices back to original DCE length
    # voiceon_indices directly correspond to the start of a new voice segment in tt
    # voiceoff_indices are 1-based, need to subtract 1 to align with the end of the voice segment in tt
    # The `+1` in MATLAB's find is implicitly handled by `np.where` on the padded array.

    # Ensure voiceon and voiceoff have corresponding pairs
    # If the signal starts with voice, voiceon might be empty or come after first voiceoff
    # If the signal ends with voice, voiceoff might be shorter than voiceon
    if len(voiceon_indices) > len(voiceoff_indices):
        # If last segment is voice, add an end point at the last frame
        voiceoff_indices = np.append(voiceoff_indices, len(DCE))
    elif len(voiceoff_indices) > len(voiceon_indices):
        # If signal starts with voice, add a start point at the beginning
        voiceon_indices = np.insert(voiceon_indices, 0, 0) # Insert at the beginning

    # Match up onsets and offsets
    T = []
    for i in range(len(voiceon_indices)):
        start_frame_idx = voiceon_indices[i]
        # Find the first offset that occurs after the current onset
        # Use np.searchsorted for efficiency to find the correct offset
        offset_idx = np.searchsorted(voiceoff_indices, start_frame_idx)

        if offset_idx < len(voiceoff_indices):
            end_frame_idx = voiceoff_indices[offset_idx]
            # Make sure end_frame_idx does not exceed the bounds of tt
            end_frame_idx = min(end_frame_idx, len(tt) - 1)
            T.append([tt[start_frame_idx], tt[end_frame_idx]])
        else:
            # This case should ideally be handled by the padding/matching logic above
            # but as a fallback, if no offset is found, consider it extends to the end.
            T.append([tt[start_frame_idx], tt[-1]])


    return np.array(T)

@read_audio
def simple_vad_estimate_snr(noisy_signal, sample_rate=16000, vad_intervals=None, tsm=0.25):
    """
    Estimate SNR using VAD to separate signal and noise segments.
    
    Args:
        noisy_signal (numpy.ndarray): Input audio signal
        sample_rate (int): Sampling rate in Hz
        vad_intervals (numpy.ndarray, optional): Pre-computed VAD intervals from guess_vad()
        tsm (float, optional): Median smoothing time for VAD (if not provided). Defaults to 0.25 s.
    
    Returns:
        float: Estimated SNR in dB
    """
    # Get VAD intervals if not provided
    if vad_intervals is None:
        vad_intervals = guess_vad(noisy_signal, sample_rate=sample_rate, tsm=tsm)
    
    # Convert VAD intervals to sample indices
    signal_mask = np.zeros_like(noisy_signal, dtype=bool)
    for start, end in vad_intervals:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        signal_mask[start_idx:end_idx] = True
    
    # Calculate signal and noise power
    signal_power = np.mean(noisy_signal[signal_mask]**2)
    noise_power = np.mean(noisy_signal[~signal_mask]**2)
    
    # Handle case where no noise segments found
    if noise_power == 0:
        return np.inf
    
    return 10 * np.log10(signal_power / noise_power)


if __name__ == '__main__':
    # Example usage:
    # Load test audio file
    audio_path = "/media/rodrigo/Novo volume/projects/git/blindsnr/audio/snr_10_white_noise.wav"
    
    D_test, SR_test = librosa.load(audio_path, sr=16000)
    
    # create time axis based on sample rate and number of samples
    t = np.arange(0, len(D_test)) / SR_test
    
    # 2. Call guess_vad
    vad_intervals = guess_vad(audio_path)

    logger.info("\nDetected VAD Intervals (seconds):")
    if vad_intervals.size > 0:
        for start, end in vad_intervals:
            logger.info(f"  Start: {start:.3f}s, End: {end:.3f}s")
    else:
        logger.info("No voice activity detected.")
    
    estimated_snr = estimate_snr(audio_path)
    logger.info(f"VAD estimated SNR: {estimated_snr:.2f} dB")

    # Optional: Plot the signal with VAD intervals for visualization
    plt.figure(figsize=(12, 4))
    plt.plot(t, D_test, label='Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal with Detected Voice Activity')

    if vad_intervals.size > 0:
        for start, end in vad_intervals:
            plt.axvspan(start, end, color='green', alpha=0.3, label='Voice Activity' if start == vad_intervals[0,0] else "")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    