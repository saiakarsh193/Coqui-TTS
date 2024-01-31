from TTS.utils.synthesizer import Synthesizer

src_dir = "yourtts_hindi_ulca-January-25-2024_09+16AM-4ca03eb8/"

syn = Synthesizer(
    # tts_checkpoint = src_dir + "best_model.pth",
    tts_checkpoint = src_dir + "checkpoint_30000.pth",
    tts_config_path = src_dir + "config.json",
    use_cuda = True,
)

wav = syn.tts(
    text = "और आप उन्हें बता सकते हैं कि यह मेरी और से है",
)

syn.save_wav(wav, "test3.wav")
