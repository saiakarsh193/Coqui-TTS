import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fast_pitch_config import FastPitchConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS, ForwardTTSArgs
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import CharactersConfig

output_path = os.path.dirname(os.path.abspath(__file__))

dataset_config = BaseDatasetConfig(
    formatter="ulca",
    dataset_name="ulca",
    # path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Telugu_mono_male",
    path="/data/saiakarsh/data/ytts_coq/IITM_TTS_data_Phase2_Hindi_mono_male",
    # language="tl",
    language="hn",
    # phonemizer="unified_parser"
)

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

# model_args = ForwardTTSArgs(
#     use_energy=True
# )

config = FastPitchConfig(
    # model_args=model_args,
    # run_name="fast_pitch_telugu_ulca",
    run_name="fast_pitch_hindi_ulca",
    audio=audio_config,
    batch_size=128,
    eval_batch_size=128,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    compute_input_seq_cache=True,
    compute_f0=True,
    # f0_cache_path=os.path.join(output_path, "f0_cache_telugu"),
    f0_cache_path=os.path.join(output_path, "f0_cache_hindi"),
    compute_energy=True,
    # energy_cache_path=os.path.join(output_path, "energy_cache_telugu"),
    energy_cache_path=os.path.join(output_path, "energy_cache_hindi"),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
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
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    precompute_num_workers=4,
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    max_seq_len=500000,
    max_audio_len=22050 * 15,
    output_path=output_path,
    datasets=[dataset_config],
    test_sentences=[
        # "ఏడు ప్రధాన పెలికాన్ జాతులు ఉన్నాయి గోధుమ పెలికాన్, పెరువియన్ పెలికాన్",
        "अभद्र व्यवहार करने का दुत्साहस करने लगे",
    ]
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)
trainer = Trainer(
    # TrainerArgs(),
    TrainerArgs(continue_path="fast_pitch_hindi_ulca-December-21-2023_08+11PM-72cf7551"),
    config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
