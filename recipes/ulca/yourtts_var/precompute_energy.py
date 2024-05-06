import os
import time
import numpy as np
import librosa
from tqdm import tqdm
from joblib import Parallel, delayed

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import compute_energy as calculate_energy
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

def compute_energy(item):
    energy_path = os.path.join(cache_path, string2filename(item["audio_unique_name"]) + "_energy.npy")
    if os.path.isfile(energy_path):
        return
    wav = ap.load_wav(item["audio_file"])
    # changed padding
    wav = np.pad(wav, ((ap.fft_size - ap.hop_length) // 2, (ap.fft_size - ap.hop_length) // 2), mode="reflect")
    energy = calculate_energy(
        wav,
        fft_size=ap.fft_size,
        hop_length=ap.hop_length,
        win_length=ap.win_length,
        center=False,
    )
    np.save(energy_path, energy)
    return energy

def compute_energy_stats(cache_path):
    energy_vecs = []
    for fl in tqdm(os.listdir(cache_path)):
        if fl.endswith(".npy"):
            fl = os.path.join(cache_path, fl)
            energy_vecs.append(np.load(fl))
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]] for v in energy_vecs])
    energy_mean, energy_std = np.mean(nonzeros), np.std(nonzeros)
    print(f"mean:{energy_mean}, std:{energy_std}")
    energy_stats = {"mean": energy_mean, "std": energy_std}
    np.save(os.path.join(cache_path, "energy_stats"), energy_stats, allow_pickle=True)

cache_path = "energy_cache"
assert not os.path.isdir(cache_path)
os.mkdir(cache_path)

# Parallel(n_jobs=-1)(delayed(compute_energy)(item) for item in tqdm(train_samples))
for item in tqdm(train_samples):
    compute_energy(item)
compute_energy_stats(cache_path)
