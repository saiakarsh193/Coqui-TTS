import os
import time
import numpy as np
import librosa
from tqdm import tqdm
from joblib import Parallel, delayed

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets.dataset import string2filename

dataset_config = BaseDatasetConfig(
    formatter="ulca",
    dataset_name="ulca",
    path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Hindi_mono_male",
    language="hn",
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
    do_trim_silence=False, # major change
)

train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=False)
ap = AudioProcessor(verbose=False, **audio_config)

def compute_f0(item):
    pitch_path = os.path.join(cache_path, string2filename(item["audio_unique_name"]) + "_pitch.npy")
    if os.path.isfile(pitch_path):
        return
    wav = ap.load_wav(item["audio_file"])
    # changed padding
    wav = np.pad(wav, ((ap.fft_size - ap.hop_length) // 2, (ap.fft_size - ap.hop_length) // 2), mode="reflect")
    f0, voiced_mask, _ = librosa.pyin(
        y=wav.astype(np.double),
        fmin=ap.pitch_fmin,
        fmax=ap.pitch_fmax,
        sr=ap.sample_rate,
        frame_length=ap.win_length,
        win_length=ap.win_length // 2,
        hop_length=ap.hop_length,
        pad_mode=ap.stft_pad_mode,
        center=False, # changed
        n_thresholds=100,
        beta_parameters=(2, 18),
        boltzmann_parameter=2,
        resolution=0.1,
        max_transition_rate=35.92,
        switch_prob=0.01,
        no_trough_prob=0.01,
    )
    f0[~voiced_mask] = 0.0
    np.save(pitch_path, f0)
    return f0

def compute_pitch_stats(cache_path):
    pitch_vecs = []
    for fl in tqdm(os.listdir(cache_path)):
        if fl.endswith(".npy"):
            fl = os.path.join(cache_path, fl)
            pitch_vecs.append(np.load(fl))
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]] for v in pitch_vecs])
    pitch_mean, pitch_std = np.mean(nonzeros), np.std(nonzeros)
    print(f"mean:{pitch_mean}, std:{pitch_std}")
    pitch_stats = {"mean": pitch_mean, "std": pitch_std}
    np.save(os.path.join(cache_path, "pitch_stats"), pitch_stats, allow_pickle=True)

cache_path = "f0_cache"
assert not os.path.isdir(cache_path)
os.mkdir(cache_path)

Parallel(n_jobs=-1)(delayed(compute_f0)(item) for item in tqdm(train_samples))
compute_pitch_stats(cache_path)
