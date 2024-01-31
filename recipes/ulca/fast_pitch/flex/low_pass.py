from scipy.signal import butter, lfilter
import librosa
import scipy.io
import sys

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

assert len(sys.argv) == 2
wav_path = sys.argv[1]
new_wav_path = wav_path[:-4] + "_fil.wav"
wav, fs = librosa.load(wav_path)
fil_wav = butter_lowpass_filter(wav, cutoff=5512, fs=fs, order=2)
scipy.io.wavfile.write(new_wav_path, fs, fil_wav)
