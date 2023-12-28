from TTS.utils.synthesizer import Synthesizer

pretrained_path = "/data/saiakarsh/.dcache/tts/"
vocoder_path = pretrained_path + "vocoder_models--en--ljspeech--hifigan_v2/"
# model_path = pretrained_path + "tts_models--en--ljspeech--tacotron2-DCA/"
model_path = pretrained_path + "tts_models--en--ljspeech--fast_pitch/"

syn = Synthesizer(
    tts_checkpoint = model_path + "model_file.pth",
    tts_config_path = model_path + "config.json",
    # tts_checkpoint = "run-October-19-2023_03+54PM-bf68848f/best_model.pth",
    # tts_config_path = "run-October-19-2023_03+54PM-bf68848f/config.json",
    vocoder_checkpoint = vocoder_path + "model_file.pth",
    vocoder_config = vocoder_path + "config.json",
    use_cuda = False,
)

# print(syn.tts_model.speaker_manager.name_to_id)
# print(syn.tts_model.language_manager.name_to_id)

wav = syn.tts(
    text = "this is a sample sentence generated using LJSpeech in CoquiTTS toolkit",
    # speaker_name = "ljspeech",
    # language_name = "en",
)

syn.save_wav(wav, "test.wav")
