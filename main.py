import os
import threading
import time as pytime
import queue
import numpy as np
import pvporcupine
import webrtcvad
import sounddevice as sd
from hailo_whisper_pipeline import HailoWhisperPipeline
from common.preprocessing import preprocess, improve_input_audio
from common.postprocessing import clean_transcription
from dotenv import load_dotenv

load_dotenv()


base_path = os.path.dirname(os.path.abspath(__file__))

# Define paths for the encoder and decoder HEF files
# These paths should point to the HEF files for the Hailo Whisper model
encoder_path = os.path.join(
    base_path, "hefs", "h8", "tiny", "tiny-whisper-encoder-10s_15dB.hef"
)
decoder_path = os.path.join(
    base_path,
    "hefs",
    "h8",
    "tiny",
    "tiny-whisper-decoder-fixed-sequence-matmul-split.hef",
)

# Define the variant of the Whisper model to use
# Currently, only the "tiny" variant is available for Hailo Whisper
variant = "tiny"

# Define the path to the Porcupine wake word model file
# This file should be the Porcupine model for the wake word "sky-net"
pv_file_path = os.path.join(
    base_path,
    "common",
    "assets",
    "Sky-net_en_raspberry-pi_v3_0_0",
    "Sky-net_en_raspberry-pi_v3_0_0.ppn",
)

porcupine = pvporcupine.create(
    access_key=os.getenv("PORCUPINE_KEY"), keyword_paths=[pv_file_path]
)

# Sample rate for Porcupine, WebRTC VAD, and Whisper
# If the sample rate is different, you may need to resample the audio
# using something like ALSA or a similar library.
SAMPLE_RATE = 16000

# We use a frame length of 2560 samples, which corresponds to 160ms at 16kHz
# This is the standard frame length for WebRTC VAD and requires less manipulation
# when processing the audio data.
# Picovoice requires 512 samples which means we control the buffer size when passivly listening.
# All audio is mono channel
FRAME_LENGTH = 2560

# 3 is the aggressiveness mode (0-3)
vad = webrtcvad.Vad(3)

# Global variables for managing the listening state and audio frames
# listening is set to True when the wake word is detected and we start processing audio
# listening remains True until silence is detected for a certain period
listening = False
processing = False  # Flag to indicate if we are currently processing audio
frames_queue = queue.Queue()

print("Initializing Hailo Whisper pipeline...")
# Initialize the Hailo Whisper pipeline
# This pipeline will handle the Whisper model inference using Hailo's hardware acceleration
# The encoder and decoder paths should point to the HEF files for the Hailo Whisper model
# The variant should be set to "tiny" for the current implementation
# multi_process_service is set to False to use a single process for the pipeline
whisper_hailo = HailoWhisperPipeline(
    encoder_path, decoder_path, variant, multi_process_service=False
)
print("Hailo Whisper pipeline initialized.")

# Set the is_nhwc flag to True for NHWC format, which is required by the Hailo Whisper pipeline
is_nhwc = True


# Thread for processing speech after the wake word is detected
# This thread will handle the audio frames, apply VAD, and send the processed audio to
# the Hailo Whisper pipeline for transcription
def speech_processing_thread():
    global listening, frames_queue, processing
    VAD_FRAME_LENGTH = 160  # WebRTC VAD frame length in samples (10ms at 16kHz)
    FALSE_POSITIVE_CHECK = 200 # 2 seconds of silence to avoid false positives
    SILENCE_LIMIT = 100  # 1 second of silence or 10 frames of silence at 16000Hz
    CHUNK_LENGTH = 10  # Length of each chunk in seconds for Whisper processing
    false_positive_count = 0  # Counter for false positives
    silence_count = 0
    speech_detected = False
    vad = webrtcvad.Vad(3)  # Initialize VAD with aggressiveness level 3

    print("Speech processing thread started...")

    # Initialize PCM data storage
    pcm_data = np.empty(0, dtype=np.int16)

    while listening:
        frames_2560 = frames_queue.get(block=True, timeout=1)
        for i in range(0, FRAME_LENGTH, VAD_FRAME_LENGTH):
            # Process 160-sample VAD buffer
            chunk_160 = frames_2560[i:i + VAD_FRAME_LENGTH]

            chunk_160 = np.array(chunk_160, dtype=np.int16)
            is_speech = vad.is_speech(chunk_160.tobytes(), sample_rate=SAMPLE_RATE)
            if not is_speech and not speech_detected:
                false_positive_count += 1
                if false_positive_count > FALSE_POSITIVE_CHECK:
                    print("False positive detected, stopping listening.")
                    listening = False
                    processing = False
                    return
                continue  # Skip if not speech and no speech detected yet

            if is_speech:
                speech_detected = True
                silence_count = 0
            else:
                silence_count += 1
            
            pcm_data = np.append(pcm_data, chunk_160)

            print(f"Silence count: {silence_count}", end="\r")

            if silence_count > SILENCE_LIMIT:
                listening = False
                processing = True
                print("Silence detected, stopping listening.")
                break
    
    if pcm_data.size == 0:
        print("No audio data collected, exiting speech processing thread.")
        return
    
    pcm_data = pcm_data.astype(np.float32) / 32768.0
    frames_queue.queue.clear()

    # Check audio gain and increase it if necessary
    improved_pcm_data = improve_input_audio(pcm_data, low_audio_gain=True)

    mel_spectrograms = preprocess(
        improved_pcm_data,
        chunk_length=CHUNK_LENGTH,
        max_duration=10,
        overlap=0.1,
        is_nhwc=is_nhwc,
    )

    for mel in mel_spectrograms:
        whisper_hailo.send_data(mel)
        pytime.sleep(0.1)
        transcription = clean_transcription(
            whisper_hailo.get_transcription()
        )
        print(f"\n{transcription}")
    
    processing = False  # Reset processing flag


# Callback function for the audio input stream
# This function is called whenever new audio data is available
# It checks for the wake word and starts the speech processing thread when detected
def audio_callback(indata, frames, time, status):
    global listening, processing
    PV_FRAME_LENGTH = 512  # Porcupine frame length in samples (32ms at 16kHz)
    if status:
        print(f"{status}")
    
    frames_2560 = indata.flatten()  # Numpy Array Shape: (2560,), dtype: int16

    if not listening and not processing:
        frame_count = len(frames_2560)
        if frame_count < FRAME_LENGTH:
            frames_2560 = np.pad(frames_2560, (0, FRAME_LENGTH - frame_count), mode='constant')
            frame_count = FRAME_LENGTH
        
        for i in range(0, frame_count, PV_FRAME_LENGTH):
            chunk_512 = frames_2560[i:i + PV_FRAME_LENGTH]
            keyword_index = porcupine.process(chunk_512)
            if keyword_index >= 0:
                listening = True
                # Start the speech processing thread
                threading.Thread(target=speech_processing_thread, daemon=True).start()

    elif listening and not processing:
        frames_queue.put(frames_2560)
    
    else:
        pytime.sleep(0.1)
        return



# Main function to initialize the audio input stream and start listening for wake words
def main():

    print(
        f"Starting Whisper pipeline. Wake words are 'picovoice' and 'bumblebee'.\nSample rate: {SAMPLE_RATE}, Frame length: {FRAME_LENGTH}"
    )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_LENGTH,
        dtype="int16",
        channels=1,
        device="default",
        callback=audio_callback,
    ):
        print("Listening (press Ctrl-C to quit)…")
        try:
            sd.sleep(int(1e6))
        except KeyboardInterrupt:
            pass

    # 3) Clean up
    porcupine.delete()


if __name__ == "__main__":
    main()
