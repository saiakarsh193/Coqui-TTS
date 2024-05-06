import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_var_config import VitsVarConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits_var import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk

torch.set_num_threads(24)

# pylint: disable=W0105
"""
    This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
    YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
    In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
    If you are interested in multilingual training, we have commented on parameters on the VitsArgs class instance that should be enabled for multilingual training.
    In addition, you will need to add the extra datasets following the VCTK as an example.
"""
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Name of the run for the Trainer
RUN_NAME = "yourtts_hin"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))  # "/raid/coqui/Checkpoints/original-YourTTS/"

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = None  # "/root/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth"

# To resume training from a run directory
CONTINUE_PATH = None

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 128

# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 22050

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 15

# init configs
hin_config = BaseDatasetConfig(
    formatter="ulca",
    dataset_name="ulca",
    path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Hindi_mono_male",
    language="hn"
)

tel_config = BaseDatasetConfig(
    formatter="ulca",
    dataset_name="ulca",
    path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Telugu_mono_male",
    language="tl",
)

eng_config = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path="/data/saiakarsh/data/ytts_coq/ljspeech",
    meta_file_train="metadata.csv",
    language="en",
)

# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to add new datasets, just add them here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [hin_config]
# DATASETS_CONFIG_LIST = [hin_config, tel_config, eng_config]

### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

# Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it
for dataset_conf in DATASETS_CONFIG_LIST:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)

# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    use_speaker_encoder_as_loss=True,
    # Useful parameters to enable multilingual training
    use_language_embedding=True,
    embedded_language_dim=4,
    use_pitch=True,
    use_energy=True
)

# General training config, here you can change the batch size and others useful parameters
config = VitsVarConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description="""
            - Original YourTTS trained using VCTK dataset
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=5000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    epochs=1000,
    compute_f0=True,
    f0_cache_path="f0_cache",
    compute_energy=True,
    energy_cache_path="energy_cache",
    use_phonemes=False,
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=False,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz¯·ßàáâãäæçèéêëìíîïñòóôõöùúûüÿāąćēęěīıłńōőœśūűźżǎǐǒǔабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґँंःअआइईउऊऋऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसह़ऄऽािीुूृॄॅॆेैॉॊोौ्ॐक़ख़ग़ज़ड़ढ़फ़य़ॠ।॥०१२३४५६७८९॰ॲఁంఃఅఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహాిీుూృౄెేైొోౌ్ౕౖౙ౦౩\u200c\u200d–",
        punctuations="!'(),-.:;? ",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "फसल अच्छी होने के कारण किसान बहुत खुश था",
            "ulca_Hindi_mono_male",
            None,
            "hn",
        ],
        # [
        #     "ఏడు ప్రధాన పెలికాన్ జాతులు ఉన్నాయి గోధుమ పెలికాన్, పెరువియన్ పెలికాన్",
        #     "ulca_Telugu_mono_male",
        #     None,
        #     "tl",
        # ],
        # [
        #     "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        #     "ljspeech",
        #     None,
        #     "en",
        # ]
    ],
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config, verbose=False)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH, continue_path=CONTINUE_PATH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
