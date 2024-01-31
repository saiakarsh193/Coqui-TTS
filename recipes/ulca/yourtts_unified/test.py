from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.dataset import PhonemeDataset
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import CharactersConfig

tel_config = BaseDatasetConfig(
    formatter="ulca",
    dataset_name="ulca",
    path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Telugu_mono_male",
    language="tl",
    phonemizer="unified_parser"
)

hin_config = BaseDatasetConfig(
    formatter="ulca",
    dataset_name="ulca",
    path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Hindi_mono_male",
    language="hn",
    phonemizer="unified_parser"
)

tam_config = BaseDatasetConfig(
    formatter="ulca",
    dataset_name="ulca",
    path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Tamil_mono_male",
    language="ta",
    phonemizer="unified_parser"
)

config = VitsConfig(
    project_name="YourTTS",
    run_description="""
            - Original YourTTS trained using VCTK dataset
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    batch_group_size=48,
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
    use_phonemes=True,
    phonemizer="multi_phonemizer",
    phoneme_cache_path="phoneme-cache",
    precompute_num_workers=12,
    add_blank=True,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        punctuations=';:,.!?¡¿—…"«»“”',
        phonemes="iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧʲɚ˞ɫ",
        is_unique=True,
        is_sorted=True,
    ),
    datasets=[tel_config, hin_config, tam_config],
    start_by_longest=True,
    cudnn_benchmark=False,
    mixed_precision=True,
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
    use_language_embedding=True,
    # test sentences for tensorboard dashboard
    test_sentences=[
        [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "ljspeech",
            None,
            "en",
        ],
        [
            "फसल अच्छी होने के कारण किसान बहुत खुश था",
            "ulca_Hindi_mono_male",
            None,
            "hn",
        ],
        [
            "ఏడు ప్రధాన పెలికాన్ జాతులు ఉన్నాయి గోధుమ పెలికాన్, పెరువియన్ పెలికాన్",
            "ulca_Telugu_mono_male",
            None,
            "tl",
        ],
        [
            "ఏడు ప్రధాన పెలికాన్ జాతులు ఉన్నాయి గోధుమ పెలికాన్, పెరువియన్ పెలికాన్",
            "ulca_Tamil_mono_male",
            None,
            "ta",
        ],
   ],
)

tokenizer, new_config = TTSTokenizer.init_from_config(config)
tokenizer.print_logs()

train_samples, test_samples = load_tts_samples(config.datasets)
print(len(train_samples))
print(len(test_samples))

# ph_dataset = PhonemeDataset(
#     samples=test_samples,
#     tokenizer=tokenizer,
#     cache_path=None,
#     precompute_num_workers=10
# )

# ph_dataset[10]