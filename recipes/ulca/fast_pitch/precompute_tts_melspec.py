import os
import numpy as np
from tqdm import tqdm
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.synthesizer import Synthesizer

dataset_config = BaseDatasetConfig(
    formatter="ulca",
    dataset_name="ulca",
    path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Hindi_mono_male",
    # path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Telugu_mono_male",
    language="hn",
    # language="tl",
    # phonemizer="unified_parser"
)
samples, _ = load_tts_samples(dataset_config, eval_split=False)

model_path = "fast_pitch_hindi_ulca-December-21-2023_08+11PM-72cf7551/"
# model_path = "fast_pitch_telugu_ulca-December-21-2023_03+34PM-72cf7551/"

syn = Synthesizer(
    tts_checkpoint = model_path + "best_model.pth",
    tts_config_path = model_path + "config.json",
    use_cuda = True,
)

MEL_CACHE_PATH = "fp_tts_hindi_melspecs"
if not os.path.isdir(MEL_CACHE_PATH):
    os.mkdir(MEL_CACHE_PATH)

for sample in tqdm(samples):
    audio_name = os.path.splitext(os.path.basename(sample['audio_file']))[0]
    mel_path = os.path.join(MEL_CACHE_PATH, audio_name + ".npy")
    if not os.path.isfile(mel_path):
        try:
            wav, outp = syn.tts(text = sample['text'])
            mel_spec = outp['outputs']['model_outputs'].squeeze(0).cpu().numpy().T
            np.save(mel_path, mel_spec)
        except Exception as e:
            print(e)
            print(sample)
