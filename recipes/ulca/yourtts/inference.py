from TTS.utils.synthesizer import Synthesizer

src_dir = "yourtts_hindi_ulca-January-25-2024_09+16AM-4ca03eb8/"

syn = Synthesizer(
    # tts_checkpoint = src_dir + "best_model.pth",
    tts_checkpoint = src_dir + "checkpoint_30000.pth",
    tts_config_path = src_dir + "config.json",
    use_cuda = True,
)

wav = syn.tts(
    text = "अब, यह मानव-केन्द्रित दृष्टिकोण या मानव-केन्द्रित दृष्टिकोण का प्रतिस्थापन है",
)

syn.save_wav(wav, "test1.wav")
