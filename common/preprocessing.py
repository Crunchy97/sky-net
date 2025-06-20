import numpy as np
import common.audio_utils


def preprocess(audio, is_nhwc=False, chunk_length = 10, max_duration = 60, overlap=0.0):
    """
    Generate the mel spectrograms
    
    Parameters:
    - audio: The audio sample.
    - chunk_length: Length in seconds of each audio chunk to process. This must match the input length of the model.
    - max_duration: Max duration of the audio sample to process.
    - overlap: Overlap between chunks. This is useful for continuous audio processing. Add some overlap (e.g. 0.2) when processing an audio longer than 10 seonds.
    """
    # Limit the audio duration
    sample_rate = common.audio_utils.SAMPLE_RATE
    max_samples = max_duration * sample_rate

    # Define parameters for chunking
    segment_duration = chunk_length  # in seconds
    segment_samples = segment_duration * sample_rate
    step = int(segment_samples * (1 - overlap))

    audio = audio[:max_samples]
    mel_spectrograms = []

    for start in range(0, len(audio), step):
        end = int(start + segment_samples)
        if start >= len(audio):
            break
        chunk = audio[start:end]

        # Ensure the chunk is 10s long (Whisper requires this)
        chunk = common.audio_utils.pad_or_trim(chunk, int(segment_duration * sample_rate))

        # Convert to Mel spectrogram
        mel = common.audio_utils.log_mel_spectrogram(chunk).to("cpu")
        # Run the encoder

        mel = np.expand_dims(mel, axis=0)  # Add new axis to match shape (1, 80, 1, 1000)
        #print(mel.shape)
        mel = np.expand_dims(mel, axis=2) 
        #print(mel.shape)

        if is_nhwc:
            mel = np.transpose(mel, [0, 2, 3, 1])

        mel_spectrograms.append(mel)

    return mel_spectrograms


def apply_gain(audio, gain_db):
    """
    Apply gain to the audio signal.
    Parameters:
    - audio: The audio sample.
    - gain_db: Gain in decibels (dB).
    """
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


def improve_input_audio(audio, low_audio_gain = True):
    """
    Improve the input audio by applying gain and detecting speech.
    Parameters:
    - audio: The audio sample.
    - vad: Boolean indicating whether to apply voice activity detection (VAD).
    - low_audio_gain: Boolean indicating whether to apply gain if the audio level is low.
    """
    
    # print(f"Max audio level: {np.max(audio)}")
    if (low_audio_gain == True) and (np.max(audio) < 0.1):
        if np.max(audio) < 0.1:
            audio = apply_gain(audio, gain_db=20)  # Increase by 15 dB
        elif np.max(audio) < 0.2:
            audio = apply_gain(audio, gain_db=10)  # Increase by 10 dB
        print(f"New max audio level: {np.max(audio)}")
    return audio
