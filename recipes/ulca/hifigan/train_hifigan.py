import os
import numpy as np

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.models.gan import GAN

def load_wav_feat_data_ulca(wav_path, feat_path, eval_split_size):
    paths = []
    for wpath in os.listdir(wav_path):
        basename, ext = os.path.splitext(wpath)
        if ext == ".wav":
            wpath = os.path.join(wav_path, wpath)
            fpath = os.path.join(feat_path, basename + ".npy")
            if os.path.isfile(fpath):
                paths.append((wpath, fpath))
            else:
                print(f" [!] feat file does not exist for basename: {basename}")
    np.random.seed(0)
    np.random.shuffle(paths)
    return paths[eval_split_size:], paths[:eval_split_size]

def load_wav_data_ulca(wav_path, eval_split_size):
    paths = []
    for wpath in os.listdir(wav_path):
        basename, ext = os.path.splitext(wpath)
        if ext == ".wav":
            wpath = os.path.join(wav_path, wpath)
            paths.append(wpath)
    np.random.seed(0)
    np.random.shuffle(paths)
    return paths[eval_split_size:], paths[:eval_split_size]

dataset_path = "/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Hindi_mono_male"
output_path = os.path.dirname(os.path.abspath(__file__))

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = HifiganConfig(
    run_name="hifigan_hindi_ulca",
    batch_size=128,
    eval_batch_size=128,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    # use_cache=False,
    data_path=dataset_path,
    output_path=output_path,
)

# init audio processor
ap = AudioProcessor.init_from_config(config)

# load training samples
# train_samples, eval_samples = load_wav_feat_data_ulca(
#     config.data_path,
#     "/data/saiakarsh/codes/Coqui-TTS/recipes/ulca/fast_pitch/fp_tts_hindi_melspecs",
#     config.eval_split_size
# )

train_samples, eval_samples = load_wav_data_ulca(
    config.data_path,
    config.eval_split_size
)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
