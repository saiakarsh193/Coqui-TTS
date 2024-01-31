import os
import sys
sys.path = ["/hdd6/Ashish/emo_tts"] + sys.path

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
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
RUN_NAME = "YourTTS-Emo-Content-MultiLingual-WO-EmoLoss"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))  # "/raid/coqui/Checkpoints/original-YourTTS/"

# To resume training
# CONTINUE_PATH = "YourTTS-MultiLingual-June-09-2023_01+34PM-dfb51e06"
CONTINUE_PATH = None

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = None  # "/root/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth"

# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 32

# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 22050

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 20

src_dir = "/hdd6/Ashish/TTS/recipes/vctk/yourtts/"
vctk_data = os.path.join(src_dir, "VCTK_resampled")
marathi_data = os.path.join(src_dir, "Marathi_resampled")
telugu_data = os.path.join(src_dir, "Telugu_resampled")
hindi_data = os.path.join(src_dir, "Anim_resampled")

vctk_config = BaseDatasetConfig(
    formatter="vctk",
    dataset_name="vctk",
    path=vctk_data,
    language="en",
    ignored_speakers=["p261", "p225", "p294", "p347", "p238", "p234", "p248", "p335", "p245", "p326", "p302"]
)
mar_config = BaseDatasetConfig(
    formatter="your_multi",
    dataset_name="marathi",
    path=marathi_data,
    language="mr"
)
tel_config = BaseDatasetConfig(
    formatter="your_multi",
    dataset_name="telugu",
    path=telugu_data,
    language="tl"
)
hin_config = BaseDatasetConfig(
    formatter="your_multi",
    dataset_name="hindi",
    path=hindi_data,
    language="hn"
)

# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to add new datasets, just add them here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [vctk_config, mar_config, tel_config, hin_config]
# DATASETS_CONFIG_LIST = [vctk_config]
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
            old_spakers_file=None,
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
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
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
    use_phonemes=False,
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz¯·ßàáâãäæçèéêëìíîïñòóôõöùúûüÿāąćēęěīıłńōőœśūűźżǎǐǒǔабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґँंःअआइईउऊऋऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसह़ऄऽािीुूृॄॅॆेैॉॊोौ्ॐक़ख़ग़ज़ड़ढ़फ़य़ॠ।॥०१२३४५६७८९॰ॲఁంఃఅఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహాిీుూృౄెేైొోౌ్ౕౖౙ౦౩\u200c\u200d–!'(),-.:;? ",
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
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "VCTK_p277",
            None,
            "en",
            "Angry"
        ],
        [
            "Be a voice, not an echo.",
            "VCTK_p239",
            None,
            "en",
            "Sad"
        ],
        [
            "फसल अच्छी होने के कारण किसान बहुत खुश था",
            "multi_pNobita",
            None,
            "hn",
            "Normal"
        ],
        [
            "अरे जा जा तेरे जैसे बहुत देखे हैं",
            "multi_pSpeaker4",
            None,
            "hn",
        ],
        [
            "अभद्र व्यवहार करने का दुत्साहस करने लगे",
            "multi_pMickey_Mouse",
            None,
            "hn",
        ],
        [
            "ఏడు ప్రధాన పెలికాన్ జాతులు ఉన్నాయి గోధుమ పెలికాన్, పెరువియన్ పెలికాన్",
            "multi_ptem",
            None,
            "tl",
        ],
        [
            "ఏడు ప్రధాన పెలికాన్ జాతులు ఉన్నాయి గోధుమ పెలికాన్, పెరువియన్ పెలికాన్",
            "multi_ptef",
            None,
            "tl",
        ],
        [
            "नत्रयुक्त घटकांच्या पूर्ततेसाठी कंद झोटोबॅक्टर २५ किलो आणि द्रवरूप सिटोबॅक्टर ५००",
            "multi_pmam",
            None,
            "mr",
        ],
        [
            "नत्रयुक्त घटकांच्या पूर्ततेसाठी कंद झोटोबॅक्टर २५ किलो आणि द्रवरूप सिटोबॅक्टर ५००",
            "multi_pmaf",
            None,
            "mr",
        ]
    ],
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
    use_language_embedding=True,
    cont_loss_alpha=1.0, # content loss
    emotion_encoder_loss_alpha=0.0 # emotion consistency loss
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config)

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
