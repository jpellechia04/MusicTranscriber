import pathlib
from pydub import AudioSegment
import librosa
import scipy.io.wavfile as wavfile
import os
import numpy as np
import soundfile as sf
import scipy
import soundfile as sf
import tensorflow as tf
import sys
import shutil
from zipfile import ZipFile
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#define local paths
current_path = os.path.realpath(__file__)
directory_path = os.path.dirname(current_path)
output_path = directory_path + "/output"
inputs_path = directory_path + "/inputs"
splits_path = directory_path + "/splits"
chart_path = output_path + "/chord_chart.txt"
temps_path = directory_path + "/temp_audiofiles"

"""
inputs: (int) harmonic_intensity, string file_path 
action: IDs chords based on the chord templates made from harmonic_intensity
outputs: dictionary with key=beat# and chord=defition
"""
def chords_on_beats(harmonic_intensity, file_path):

    print("generating chord templates . . .")
    Chords = generate_chord_templates(harmonic_intensity)

    # Estimate beats
    print("approximating tempo . . .  ")
    print("Estimating beats . . . ")
    y, sr = librosa.load(file_path)
    
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print("Recognized tempo: ",tempo, "beats per minute")

    # seperates audio_file into harmonic and percussive tracks. harmonic_track, percussive_track are file paths
    print("separating harmonic and percussive tracks . . . ")
    #harmonic_track, percussive_track = separate_harmonic_percussive(audio_file)
    harmonic_track = file_path

    # seperate foreground and background of harmonic_track into feature chromagrams
    print("Separating foreground and background . . . ")
    foreground_chroma, background_chroma = extract_foreground_background(harmonic_track)

    # enhance the chromagram
    print("Enhancing chromagram . . . ")
    librosa.load(harmonic_track)
    chroma = chroma_enhance(librosa.feature.chroma_cqt(y=y, sr=sr))

    # Calculate the hop length based on the estimated beat period
    print("Calculating hop length . . . ")
    hop_length = int(sr * (60 / tempo))

    chords_dict = {}
    measure_count = 0
    beat_times = librosa.frames_to_time(beats, sr=sr)
    beat_count = 0
    with open(chart_path, "w") as f:
        print("Recognized tempo is ", tempo, file=f)
        print("", file=f)
        for frame_number in range(len(beats)):
            # Ensure that the frame indices are within the bounds of the chromagram matrix
            start_frame = beats[frame_number]
            end_frame = beats[frame_number] + 4

            # Extract the chroma vector for the current beat segment
            chroma_segment = chroma[:, start_frame:end_frame]
            #print(chroma_segment)
            bass_note = identify_bass_note(chroma_segment)
        
            # Print the recognized chord for the current segment
            recognized_chord = recognize_chord(chroma, Chords, start_frame, end_frame)

            chords_dict[frame_number + 1] = recognized_chord

            # Increment measures
            if beat_count % 4 == 0:
                measure_count += 1

            time_stamp = beat_times[beat_count]

            chords_dict[frame_number + 1] = [recognized_chord, time_stamp]
        
            print(f"Measure {measure_count}, At beat {frame_number + 1}, {time_stamp} seconds: {recognized_chord}/{bass_note}", file=f,)
            beat_count += 1
    return chords_dict

"""
    Recognizes a chord from a given chroma using a dataset of chord templates.
    Parameters:
    - chroma (numpy array): Input chromagram.
    - CHORD_TEMPLATES (dict): Dictionary containing chord templates.
    Returns:
    - str: Identified chord name.
"""
def recognize_chord(chroma, CHORD_TEMPLATES, start_frame=None, end_frame=None):
    if start_frame is None or end_frame is None:
        return "No frames provided"

    # Extract the chroma segment based on start_frame and end_frame
    chroma_segment = chroma[:, start_frame:end_frame]
    bass = identify_bass_note(chroma_segment)

    # Save the Euclidean distances between the input chroma and each template in a dictionary
    distances = {chord: np.linalg.norm(np.transpose(chroma_segment) - (template)) for chord, template in CHORD_TEMPLATES.items()}

    # Find the chord with the minimum distance
    recognized_chord = min(distances, key=distances.get)

    return recognized_chord


"""
Identifies the bass note of a chord from a chromagram segment.

Parameters:
 - chromagram (numpy array): Input chromagram.

Returns:
 - str: Identified bass note.
"""
def identify_bass_note(chromagram):
    # Get the pitch class with the highest energy
    max_pitch_class = np.argmax(np.sum(chromagram, axis=1))

    # Map pitch class index to note name
    note_names = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'Bb', 'B']
    bass_note = note_names[max_pitch_class]

    return bass_note

"""
Normalize an audio file to a target root mean square (RMS) level.

Parameters:
 - input_file (str): Path to the input audio file.
 - output_file (str): Path to the output normalized audio file.
 - target_rms (float): Target RMS level in decibels (dB). Default is -20 dB.

Returns:
 - None
 """
def normalize_audio(input_file, output_file, target_rms=-20):

    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)

    # Compute the root mean square (RMS) of the audio signal
    rms = librosa.feature.rms(y=y)[0][0]

    # Calculate the adjustment factor to reach the target RMS level
    adjustment_factor = np.power(10.0, (target_rms - 20 * np.log10(rms)))

    # Normalize the audio signal
    y_normalized = y * adjustment_factor

    # Save the normalized audio to a new file
    sf.write(output_file, y_normalized, sr)

"""
    Extracts chroma features from an audio file using librosa.
    Parameters:
    - audio_file (str): Path to the audio file.
    - hop_length (int): Hop length for computing chroma features.
    Returns:
    - numpy array: Chroma features.
"""
def extract_chroma(audio_file, hop_length=512):
    y, sr = librosa.load(audio_file)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    return chroma

"""
Creates templates of all triads and 7th chords.

An array with a single 1 would represent one note. For example, a Bb would look like this
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] :=
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'Bb', 0]

An array of all 1's would represent all 12 of the following notes:
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] := 
['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'Bb', 'B']

For example, this array would represent C Minor:
[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] :=
[C, 0, 0, D#/Eb, 0, 0, 0, G, 0, 0, 0, 0]
"""
def generate_chord_templates(intensity=3):
    if intensity < 1 or intensity > 3:
        raise ValueError("Difficulty level should be between 1 and 3")

    CHORD_TEMPLATES = {}
    note_names = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'Bb', 'B']

    for i in range(12):
        major_chord = np.zeros(12)
        major_chord[i] = 1  # Set the root note to 1
        major_chord[(i + 4) % 12] = 1  # Set the major third
        major_chord[(i + 7) % 12] = 1  # Set the perfect fifth
        CHORD_TEMPLATES[f'{note_names[i]}maj'] = major_chord

        minor_chord = np.zeros(12)
        minor_chord[i] = 1  # Set the root note to 1
        minor_chord[(i + 3) % 12] = 1  # Set the minor third
        minor_chord[(i + 7) % 12] = 1  # Set the perfect fifth
        CHORD_TEMPLATES[f'{note_names[i]}min'] = minor_chord

        diminished_chord = np.zeros(12)
        diminished_chord[i] = 1  # Set the root note to 1
        diminished_chord[(i + 3) % 12] = 1  # Set the minor third
        diminished_chord[(i + 6) % 12] = 1  # Set the diminished fifth
        CHORD_TEMPLATES[f'{note_names[i]}dim'] = diminished_chord

        if intensity >= 2:
            major_seventh_chord = np.zeros(12)
            major_seventh_chord[i] = 1  # Set the root note to 1
            major_seventh_chord[(i + 4) % 12] = 1  # Set the major third
            major_seventh_chord[(i + 7) % 12] = 1  # Set the perfect fifth
            major_seventh_chord[(i + 11) % 12] = 1  # Set the major seventh
            CHORD_TEMPLATES[f'{note_names[i]}maj7'] = major_seventh_chord

            minor_seventh_chord = np.zeros(12)
            minor_seventh_chord[i] = 1  # Set the root note to 1
            minor_seventh_chord[(i + 3) % 12] = 1  # Set the minor third
            minor_seventh_chord[(i + 7) % 12] = 1  # Set the perfect fifth
            minor_seventh_chord[(i + 10) % 12] = 1  # Set the minor seventh
            CHORD_TEMPLATES[f'{note_names[i]}min7'] = minor_seventh_chord

            # Generate min7b5 chords
            min7b5_chord = np.zeros(12)
            min7b5_chord[i] = 1  # Set the root note to 1
            min7b5_chord[(i + 3) % 12] = 1  # Set the minor third
            min7b5_chord[(i + 6) % 12] = 1  # Set the diminished fifth
            min7b5_chord[(i + 10) % 12] = 1  # Set the minor seventh
            CHORD_TEMPLATES[f'{note_names[i]}min7b5'] = min7b5_chord

            # Generate dominant 7th chords
            dominant_seventh_chord = np.zeros(12)
            dominant_seventh_chord[i] = 1  # Set the root note to 1
            dominant_seventh_chord[(i + 4) % 12] = 1  # Set the major third
            dominant_seventh_chord[(i + 7) % 12] = 1  # Set the perfect fifth
            dominant_seventh_chord[(i + 10) % 12] = 1  # Set the minor seventh
            CHORD_TEMPLATES[f'{note_names[i]}7'] = dominant_seventh_chord

        # Add more chords based on difficulty level 3
        if intensity == 3:

            # Generate diminished 7th chords
            diminished_seventh_chord = np.zeros(12)
            diminished_seventh_chord[i] = 1  # Set the root note to 1
            diminished_seventh_chord[(i + 3) % 12] = 1  # Set the minor third
            diminished_seventh_chord[(i + 6) % 12] = 1  # Set the diminished fifth
            diminished_seventh_chord[(i + 9) % 12] = 1  # Set the minor seventh
            CHORD_TEMPLATES[f'{note_names[i]}dim7'] = diminished_seventh_chord

            # Generate minor major 7th chords
            minor_major_seventh_chord = np.zeros(12)
            minor_major_seventh_chord[i] = 1  # Set the root note to 1
            minor_major_seventh_chord[(i + 3) % 12] = 1  # Set the minor third
            minor_major_seventh_chord[(i + 7) % 12] = 1  # Set the perfect fifth
            minor_major_seventh_chord[(i + 11) % 12] = 1  # Set the major seventh
            CHORD_TEMPLATES[f'{note_names[i]}minmaj7'] = minor_major_seventh_chord

            """
            # Generate major 7th b5 chords
            maj7b5_chord = np.zeros(12)
            maj7b5_chord[i] = 1  # Set the root note to 1
            maj7b5_chord[(i + 4) % 12] = 1  # Set the major third
            maj7b5_chord[(i + 6) % 12] = 1  # Set the diminished fifth
            maj7b5_chord[(i + 11) % 12] = 1  # Set the major seventh
            CHORD_TEMPLATES[f'{note_names[i]}min7b5'] = maj7b5_chord
            """

            # Generate #11 chords
            major_sharp11_chord = np.zeros(12)
            major_sharp11_chord[i] = 1  # Set the root note to 1
            major_sharp11_chord[(i + 4) % 12] = 1  # Set the major third
            major_sharp11_chord[(i + 7) % 12] = 1  # Set the perfect fifth
            major_sharp11_chord[(i + 11) % 12] = 1  # Set the major seventh
            major_sharp11_chord[(i + 6) % 12] = 1  # Set the sharp eleventh
            CHORD_TEMPLATES[f'{note_names[i]}maj7#11'] = major_sharp11_chord

    return CHORD_TEMPLATES

"""
input: chromagram
action: makes a temporary audio file constructed from the chromagram input
output: path of a temporary wav file of the chromagram
"""
def rewrite_temp(chromagram, temp_file_name, sr, y):

    # Reconstruct the audio data from the chromagram
    stft = np.fft.irfft(chromagram)
    phase = np.angle(np.fft.rfft(y))

    # Ensure the phase array has the same shape as the stft array
    if phase.shape != stft.shape:
        phase = phase[:, :stft.shape[1]]
    if phase.ndim == 1:
        phase = phase[:, np.newaxis]
    phase = np.angle(np.fft.rfft(y))

    reconstructed_audio = np.fft.irfft(stft * np.exp(1j * phase))

    # Write the reconstructed audio data to a WAV file
    wav = wavfile.write(temp_file_name+".wav", sr, reconstructed_audio.astype(np.int16))

    #make its path
    temp_wav_path = os.path.join(temps_path, temp_file_name+".wav")

    return temp_wav_path

"""
input: chromagram, temporary audio file path 
action: makes a temporary audio file constructed from the chromagram input
output: path of a temporary wav file of the chromagram
"""
def write_chromagram_to_temp(chromagram, output_file, sr, y):
    y = librosa.feature.synthesize(chromagram, n_fft=2048, hop_length=512)
    librosa.output.write_wav(output_file, y, sr)

"""
# Code source: Brian McFee
# License: ISC

##################
# Standard imports
Input
- audio file path
Outputs
- foreground chroma: seperated foreground audio signal (vocal, lead guitar, etc)
- background chroma: accompanying instruments
"""
def extract_foreground_background(audio_file):
    #user update
    y, sr = librosa.load(audio_file, duration=120)

    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))

    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimum
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)

    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                margin_i * (S_full - S_filter),
                                power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    y_foreground = librosa.istft(S_foreground * phase)
    y_background = librosa.istft(S_background * phase)

    # Rewrite istft's into feature chromas
    C_foreground = librosa.feature.chroma_stft(y=y_foreground, sr=sr)
    C_background = librosa.feature.chroma_stft(y=y_background, sr=sr)

    #turns STFT back into audio files
    sf.write("foreground.wav", y_foreground, sr)
    sf.write("background.wav", y_foreground, sr)

    return C_foreground, C_background


""""
# Code source: Brian McFee
# License: ISC
# sphinx_gallery_thumbnail_number = 5
Input: chromagram
Output: enhanced chromagram
"""
def chroma_enhance(chromagram):
    # Isolate harmonic component of chromagram
    chroma_harm = np.maximum(chromagram,
                           librosa.decompose.nn_filter(chromagram,
                                                       aggregate=np.median,
                                                       metric='cosine'))
    
    # Remove noise with non-local filtering
    chroma_filter = np.minimum(chroma_harm,
                           librosa.decompose.nn_filter(chroma_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))
    
    # Local discontinuities and transients can be suppressed by using a horizontal median filter.
    chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))

    return chroma_smooth
    
def convert_wav_to_mp3(input_wav_path, output_mp3_path):
    """
    Convert a .wav file to a .mp3 file using pydub.

    :param input_wav_path: Path to the input .wav file.
    :param output_mp3_path: Path to the output .mp3 file.
    """
    # Load the .wav file
    audio = AudioSegment.from_wav(input_wav_path)

    # Export as .mp3
    audio.export(output_mp3_path, format="mp3")
    print(f"Converted {input_wav_path} to {output_mp3_path}")

"""
input: a dictionary with key=beat# and def=[chord, time stamp]
output: a dictionary but removing repetative chords.
"""
def just_the_changes(chords_dict):
    keys_to_remove = []
    prev_chord = None

    for key in chords_dict:
        chord, time_stamp = chords_dict[key]
        if chord == prev_chord:
            keys_to_remove.append(key)
        else:
            prev_chord = chord

    for key in keys_to_remove:
        del chords_dict[key]

    return chords_dict

"""
input: user
output: booleans for bass, drums, vocals, piano
"""
def bass_drum_piano_vocal_other_chart():
    bass = True
    drums = True
    piano = True
    vocal = True
    other = True
    chart = True
    return bass, drums, piano, vocal, other, chart

"""
Input: '/path/to/filename.ext'
Output: 'filename.ext'
"""
def trim_file_path(file_path):
    # get the base name of the file (file name with extension)
    base_name = os.path.basename(file_path)

    # get the file name without the extension
    file_name = os.path.splitext(base_name)[0]

    return file_name
"""
inputs: 
- an audio file path containing noise and pitched sound 
returns: 
- harmonic_track path, made of the pitched elements in the input_audio_file
- percussive_track path, made of the percussive/noise sources in the input_audio_file
"""
def separate_harmonic_percussive(input_audio_file, output_dir="output"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the audio file
    y, sr = librosa.load(input_audio_file, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # Separate harmonic and percussive components
    harmonic, percussive = librosa.effects.hpss(y)

    # Define output file paths
    harmonic_file_wav = os.path.join(directory_path, "harmonic_track.wav")
    percussive_file_wav = os.path.join(directory_path, "percussive_track.wav")

    # Save the separated tracks to WAV files
    sf.write(harmonic_file_wav, harmonic, sr)
    sf.write(percussive_file_wav, percussive, sr)

    # Convert WAV files to MP3
    harmonic_file_mp3 = os.path.join(directory_path, "harmonic_track.mp3")
    percussive_file_mp3 = os.path.join(directory_path, "percussive_track.mp3")

    AudioSegment.from_wav(harmonic_file_wav).export(harmonic_file_mp3, format="mp3")
    AudioSegment.from_wav(percussive_file_wav).export(percussive_file_mp3, format="mp3")

    # Remove the temporary WAV files
    os.remove(harmonic_file_wav)
    os.remove(percussive_file_wav)

    return harmonic_file_mp3, percussive_file_mp3

"""Use this to create a model and pass it to separate()"""
def load_model(): 
    from spleeter.separator import Separator
    tf.compat.v1.config.experimental_run_functions_eagerly(True)
    separator = Separator('spleeter:5stems')
    return separator

"""
Spleeter's separate_to_file
"""
def separate(audio_file, output_directory, separator):
    separator.separate_to_file(audio_file, output_directory, synchronous=True)

"""
Input: path to a stereo audio file
Converts it to mono
"""
import wave
import struct
def convert_to_mono(input_wav_file):
    with wave.open(input_wav_file, 'rb') as infile:
        # Get the number of channels from the input WAV file
        num_channels = infile.getnchannels()
        # Get the sample width from the input WAV file
        sample_width = infile.getsampwidth()
        # Get the frame rate from the input WAV file
        frame_rate = infile.getframerate()
        # Get the number of frames from the input WAV file
        num_frames = infile.getnframes()

        # Convert the input WAV file to a mono WAV file
        if num_channels == 2:
            frames = infile.readframes(num_frames)
            left_frames = frames[0::num_channels * sample_width]
            right_frames = frames[1::num_channels * sample_width]
            # Average the left and right frames to get the mono frames
            mono_frames = b''.join([struct.pack('<h', (left + right) // 2) for left, right in zip(left_frames, right_frames)])

            # Create the output mono WAV file
            output_wav_file = input_wav_file[:-4] + '_mono.wav'
            with wave.open(output_wav_file, 'wb') as outfile:
                outfile.setnchannels(1)
                outfile.setsampwidth(sample_width)
                outfile.setframerate(frame_rate)
                outfile.writeframes(mono_frames)

            # Delete the stereo version of the WAV file
            os.remove(input_wav_file)

"""
Uses basic-pitch to convert audio files to midi
input: path/to/input/file.wav
output path/to/directory/folder
"""

import tensorflow as tf
import pathlib
import sys

def to_midi(path, midi_directory):
    from basic_pitch.inference import predict_and_save

    tf.config.experimental_run_functions_eagerly(True)
    tf.config.run_functions_eagerly(True)
    tf.compat.v1.enable_eager_execution()

    midi_directory = pathlib.Path(midi_directory)
    print("converting ", path, "to ", midi_directory)
    try:
        predict_and_save([path], midi_directory, True, False, False, False)
    except Exception as e:
        print(f"An error occurred: {e}")
        pass

@tf.function
def to_midi_graph(paths, midi_directory):
    midi_directory = pathlib.Path(midi_directory)
    for path in paths:
        predict_and_save([path], midi_directory, True, False, False, False)

def to_midi2(paths, midi_directory):
    midi_directory = pathlib.Path(midi_directory)
    for path in paths:
        print("converting ",path, "to ", midi_directory)
        try:
            to_midi_graph([path], midi_directory)
        except Exception as e:
            print(f"An error occurred: {e}")
            pass


import zipfile
def make_zip_with_zip64(output_path, archive_name, source_dir):
    """
    Create a ZIP archive of a directory with ZIP64 support.

    :param output_path: Path to the directory where the ZIP file will be saved.
    :param archive_name: Name of the output ZIP file (without extension).
    :param source_dir: Path to the directory to be archived.
    """
    archive_path = os.path.join(output_path, archive_name + ".zip")
    
    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=source_dir)
                    zipf.write(file_path, arcname)
        print(f"Created ZIP file at {archive_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def make_zip_with_shell(source_path, output_path, archive_name):
    """
    Create a ZIP archive of a directory using shell commands.

    :param source_path: Path to the directory containing files to be zipped.
    :param output_path: Path to the directory where the ZIP file will be saved.
    :param archive_name: Name of the output ZIP file (without extension).
    """
    archive_path = os.path.join(output_path, archive_name + ".zip")
    try:
        # Ensure the output path exists
        os.makedirs(output_path, exist_ok=True)
        
        # Construct the shell command
        command = ["zip", "-r", archive_path, "."]
        
        # Change to the source directory and zip its contents
        result = subprocess.run(command, cwd=source_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"Created ZIP file at {archive_path}")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# Function to convert all midi files in parallel
import concurrent.futures
def to_midi_all(directory, midi_directory, use_concurrency):
    audio_files = [file for file in os.listdir(directory) if file.endswith('.mp3')]

    if use_concurrency:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(to_midi, [os.path.join(directory, file) for file in audio_files], [midi_directory] * len(audio_files))
    else:
        for audio_file in audio_files:
            to_midi(os.path.join(directory, audio_file), midi_directory)

import stat
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def delete_files_in_output():
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file != "chord_chart.txt":
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                except PermissionError:
                    os.chmod(file_path, stat.S_IWRITE)
                    os.remove(file_path)

def delete_files_and_directories_in_splits():
    if os.path.exists(splits_path):
        for filename in os.listdir(splits_path):
            file_path = os.path.join(splits_path, filename)
            if filename != 'build.txt':
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path, onerror=remove_readonly)
                else:
                    os.remove(file_path)

def delete_files_and_directories_in_inputs():
    if os.path.exists(inputs_path):
        shutil.rmtree(inputs_path, onerror=remove_readonly)
        os.makedirs(inputs_path)

def process_music(file_path, separator, use_concurrency):
    delete_files_in_output()

    with open(output_path + "/chord_chart.txt", 'w') as file:
        file.write('')

    for file in os.listdir(output_path):
        if file != "chord_chart.txt" and file != ".DS_Store":
            try:
                os.remove(os.path.join(output_path, file))
            except PermissionError as e:
                print(f"Permission error: {e}")

    for file in os.listdir(splits_path):
        if file != ".DS_Store":
            try:
                os.remove(os.path.join(splits_path, file))
            except PermissionError as e:
                print(f"Permission error: {e}")

    audio_file = os.path.realpath(file_path)
    trimmed = trimmed_name = trim_file_path(audio_file)
    
    y, sr = librosa.load(audio_file, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    bass, drums, piano, vocal, other, chord_chart = bass_drum_piano_vocal_other_chart()
    
    print("separating sources . . . ")
    logger.info("Starting music processing...")
    logger.info("separating sources . . .")
    separate(audio_file, splits_path, separator)
    logger.info(f"Finished separation for {audio_file}...")

    print('converting to mp3')
    for stem in ['vocals', 'drums', 'piano', 'other', 'bass']:
        to_mp3_in = splits_path + '/' + trimmed + '/' + stem + '.wav'
        to_mp3_out = splits_path + '/' + trimmed + '/' + stem + '.mp3'
        print('Converting ', to_mp3_in, ' to ', to_mp3_out, ' in ', 'splits_path')
        convert_wav_to_mp3(to_mp3_in, to_mp3_out)
        os.remove(to_mp3_in)

    tf.compat.v1.config.experimental_run_functions_eagerly(True)

    print('converting to midi')    
    logger.info("Sources separated. Converting to MIDI...")
    to_midi_all(splits_path + '/' + trimmed, output_path, use_concurrency=use_concurrency)

    'writing chord dictionary'
    chords_dict = chords_on_beats(2, audio_file)

    print(".mid's and chord_chart.txt all done")

    shutil.move(splits_path + '/' + trimmed + '/vocals.mp3', output_path)
    print('moved vocals.mp3 from splits to output')
    shutil.move(splits_path + '/' + trimmed + '/drums.mp3', output_path)
    print('moved drums.mp3 from splits to output')
    shutil.move(splits_path + '/' + trimmed + '/piano.mp3', output_path)
    print('moved piano.mp3 from splits to output')
    shutil.move(splits_path + '/' + trimmed + '/other.mp3', output_path)
    print('moved other.mp3 from splits to output')
    shutil.move(splits_path + '/' + trimmed + '/bass.mp3', output_path)
    print('moved bass.mp3 from splits to output')

    print('attempting to zip...')
    logger.info("Creating zip file...")
    make_zip_with_shell(output_path, output_path, trimmed)
    print('all zipped up!')
    logger.info(f"Zip file created")
    delete_files_and_directories_in_inputs()
    delete_files_and_directories_in_splits()
    print(".mid's and chord_chart.txt are HOT out the oven in /MusicTranscribe/music_transcriber/output/")

if __name__ == '__main__':
    delete_files_in_output()

    with open(output_path + "/chord_chart.txt", 'w') as file:
        file.write('')

    for file in os.listdir(output_path):
        if file != "chord_chart.txt" and file != ".DS_Store":
            try:
                os.remove(os.path.join(output_path, file))
            except PermissionError as e:
                print(f"Permission error: {e}")

    for file in os.listdir(splits_path):
        if file != ".DS_Store":
            try:
                os.remove(os.path.join(splits_path, file))
            except PermissionError as e:
                print(f"Permission error: {e}")

    if len(sys.argv) > 2 or len(sys.argv) == 0:
        print("Please enter a valid file path as your only input.")
        sys.exit()

    try: 
        librosa.load(os.path.realpath(sys.argv[1]), sr=None)
        print(os.path.realpath(sys.argv[1]), 'is a valid filepath.')
    except:
        print("The filepath you entered was not found or could not be loaded.")
        print("filepath: ", os.path.realpath(sys.argv[1]))
        sys.exit()

    audio_file = os.path.realpath(sys.argv[1])
    trimmed = trimmed_name = trim_file_path(audio_file)
    
    y, sr = librosa.load(audio_file, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    bass, drums, piano, vocal, other, chord_chart = bass_drum_piano_vocal_other_chart()
    
    print("loading model . . . ")
    separator = load_model()

    print("separating sources . . . ")
    separate(audio_file, splits_path, separator)

    for stem in ['vocals', 'drums', 'piano', 'other', 'bass']:
        to_mp3_in = splits_path + '/' + trimmed + '/' + stem + '.wav'
        to_mp3_out = splits_path + '/' + trimmed + '/' + stem + '.mp3'
        print('Converting ', to_mp3_in, ' to ', to_mp3_out, ' in ', 'splits_path')
        convert_wav_to_mp3(to_mp3_in, to_mp3_out)
        os.remove(to_mp3_in)

    tf.compat.v1.config.experimental_run_functions_eagerly(True)

    to_midi_all(splits_path + '/' + trimmed, output_path, use_concurrency=True)

    chords_dict = chords_on_beats(2, audio_file)

    print(".mid's and chord_chart.txt all done")

    shutil.move(splits_path + '/' + trimmed + '/vocals.mp3', output_path)
    print('moved vocals.mp3 from splits to output')
    shutil.move(splits_path + '/' + trimmed + '/drums.mp3', output_path)
    print('moved drums.mp3 from splits to output')
    shutil.move(splits_path + '/' + trimmed + '/piano.mp3', output_path)
    print('moved piano.mp3 from splits to output')
    shutil.move(splits_path + '/' + trimmed + '/other.mp3', output_path)
    print('moved other.mp3 from splits to output')
    shutil.move(splits_path + '/' + trimmed + '/bass.mp3', output_path)
    print('moved bass.mp3 from splits to output')

    print('attempting to zip...')
    make_zip_with_shell(output_path, output_path, trimmed)
    print('all zipped up!')
    delete_files_and_directories_in_inputs()
    delete_files_and_directories_in_splits()
    print(".mid's and chord_chart.txt are HOT out the oven in /MusicTranscribe/music_transcriber/output/")