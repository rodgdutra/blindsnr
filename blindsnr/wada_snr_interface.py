import subprocess
import wave
import struct
import os
import logging
import hashlib
import re
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_audio(
    input_file,
    output_file,
    target_sample_rate=16000,
    target_channels=1,
    target_sample_width=2,
):
    """
    Converts an audio file to the format required by WADASNR: 16 kHz, mono, 2 bytes per sample.
    Uses the wave module for basic conversion. More sophisticated conversion might be needed
    for complex audio formats.
    """
    try:
        with wave.open(input_file, "rb") as wf_in:
            frame_rate = wf_in.getframerate()
            channels = wf_in.getnchannels()
            sample_width = wf_in.getsampwidth()  # Bytes per sample

            # Check if conversion is needed
            if (
                frame_rate == target_sample_rate
                and channels == target_channels
                and sample_width == target_sample_width
            ):
                # No conversion needed, just copy the file
                with open(input_file, "rb") as f_in, open(
                    output_file, "wb"
                ) as f_out:
                    f_out.write(f_in.read())
                logger.debug("No audio conversion needed. Copying file.")
                return

            # Basic conversion using wave module (limited capabilities)
            with wave.open(output_file, "wb") as wf_out:
                wf_out.setnchannels(target_channels)
                wf_out.setsampwidth(target_sample_width)
                wf_out.setframerate(target_sample_rate)

                # Read and convert frames
                num_frames = wf_in.getnframes()
                for _ in range(num_frames):
                    frame = wf_in.readframes(1)
                    # Handle different sample widths (e.g., 1 byte to 2 bytes)
                    if sample_width == 1 and target_sample_width == 2:
                        # Convert 8-bit audio to 16-bit (assuming unsigned 8-bit)
                        for sample in struct.unpack("B", frame):
                            wf_out.writeframes(
                                struct.pack("<h", (sample - 128) * 256)
                            )  # Scale and convert to signed 16-bit
                    elif sample_width == 2 and target_sample_width == 1:
                        # Convert 16-bit audio to 8-bit (clipping)
                        for sample in struct.unpack("<h", frame):
                            wf_out.writeframes(
                                struct.pack(
                                    "B", max(0, min(255, (sample // 256) + 128))
                                )
                            )  # Scale and convert to unsigned 8-bit
                    elif sample_width == target_sample_width:
                        wf_out.writeframes(frame)  # No conversion needed
                    else:
                        logger.warning(
                            f"Unsupported sample width conversion from {sample_width} to {target_sample_width}. Skipping frame."
                        )
    except wave.Error as e:
        logger.error(f"Error processing audio: {e}")
        return None

def check_conversion_needed(input_file,  target_sample_rate=16000,
    target_channels=1,
    target_sample_width=2):
    with wave.open(input_file, "rb") as wf_in:
        frame_rate = wf_in.getframerate()
        channels = wf_in.getnchannels()
        sample_width = wf_in.getsampwidth()  # Bytes per sample

        # Check if conversion is needed
        if (
            frame_rate == target_sample_rate
            and channels == target_channels
            and sample_width == target_sample_width
        ):
            logger.debug("No audio conversion needed. Copying file.")
            return False
        return True

def run_wada_snr(audio_file_path, wada_snr_exe_path="WadaSNR/Exe/WADASNR", table_file="WadaSNR/Table/Alpha0.400000.txt", input_file_format="nist"):
    """
    Runs the Original WADA SNR executable with the given audio file.
    The source code must be donwloaded from the link:
    http://www.cs.cmu.edu/~robust/archive/algorithms/WADA_SNR_IS_2008/WadaSNR.tar.gz


    Args:
        audio_file_path (str): Path to the audio file.
        wada_snr_executable_path (str): Path to the WADASNR executable.
        table_file_path (str): Path to the table file (e.g., Alpha0.400000.txt).
        input_file_format (str): Input file format (e.g., "nist", "mswav", "raw"). Defaults to "nist".

    Returns:
        subprocess.CompletedProcess: The result of the subprocess call, or None if an error occurred.
    """
    try:
        temp_audio_file = audio_file_path
        # Create a temporary file for the converted audio
        if check_conversion_needed(audio_file_path):
            temp_audio_file = (
            f"temp_audio_{hashlib.sha256(audio_file_path.encode()).hexdigest()}.wav"
            )
            logger.debug("Converting audio")
            convert_audio(audio_file_path, temp_audio_file)
            logger.debug("Done converting audio")

        # Construct the command
        command = [
        wada_snr_exe_path,
            "-i",
            temp_audio_file,
            "-t",
        table_file,
            "-ifmt",
            input_file_format,
        ]

        logger.debug(f"Running WADASNR with command: {command}")
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Clean up the temporary file
        if temp_audio_file != audio_file_path:
            os.remove(temp_audio_file)
        logger.debug("Temporary audio file deleted.")

        return result

    except FileNotFoundError:
        logger.error(f"WADASNR executable not found at {wada_snr_exe_path}")
        return None

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

def process_wada_result(result):
    """
    Process the WADASNR output and extract relevant information.
    Args:
        result (subprocess.CompletedProcess): The result of the subprocess call, or None if an error occurred.
    Returns:
    snr_value (float): The SNR value extracted from the output.
    """
    # Use regex to extract the SNR value in dB from the output
    snr_match = re.search(r"Total SNR is (\d+\.\d+) dB\.", result.stdout)
    snr_value = np.nan
    if snr_match:
        snr_value = float(snr_match.group(1))
        logger.debug(f"Extracted SNR Value: {snr_value} dB")
    else:
        logger.warning("SNR value not found in WADASNR output.")
    return snr_value

def wada_original(audio_file_path, wada_snr_exe_path="WadaSNR/Exe/WADASNR", table_file="WadaSNR/Table/Alpha0.400000.txt"):
    """
    Processes the given audio file using WADASNR and returns the SNR value.

    Args:
        audio_file_path (str): Path to the input audio file.
    wada_snr_executable_path (str): Path to the WADASNR executable.
    table_file_path (str): Path to the table file (e.g., Alpha0.400000.txt).

    Returns:
        snr_value (float): The extracted SNR value from the WADASNR output, or np.nan if an error occurred.
    """
    result = run_wada_snr(
            audio_file_path,
        wada_snr_exe_path=wada_snr_exe_path,
        table_file=table_file
        )
    if result:
        return process_wada_result(result)
    else:
        logger.error("WADASNR execution failed.")
        return np.nan

if __name__ == "__main__":
    # Example usage (replace with your actual paths)
    audio_file = "../audio/sb01_10dB_Music.wav"  # Example audio file

    snr_value = wada_original(audio_file)

    if not np.isnan(snr_value):
        logger.debug(f"Final SNR Value: {snr_value} dB")
    else:
        logger.error("Failed to extract SNR value.")