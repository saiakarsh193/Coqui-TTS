from TTS.utils.manage import ModelManager

manager = ModelManager()
# model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/fast_pitch")
# model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")

model_path, config_path, _ = manager.download_model("vocoder_models/en/ljspeech/hifigan_v2")
model_path, config_path, _ = manager.download_model("vocoder_models/en/ljspeech/multiband-melgan")
