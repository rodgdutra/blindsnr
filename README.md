# blindsnr

A Python library for blind SNR estimation.

## Installation

```bash
pip install .
```

## Usage

```python
import blindsnr
import numpy as np

# Example usage
noisy_signal = np.random.randn(44100)  # Example noisy signal
snr_wada = blindsnr.wada_original(noisy_signal)
print(f"Estimated SNR (Wada Original): {snr_wada} dB")
```

## Methods

*   `wada(noisy_signal, frame_size=2048, hop_size=1024)`: Estimates SNR using the Wada method.

## Contributing

Contributions are welcome!