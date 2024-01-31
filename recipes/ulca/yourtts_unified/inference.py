from TTS.utils.synthesizer import Synthesizer

syn = Synthesizer(
    tts_checkpoint = "YourTTS-ULCA-T1-October-20-2023_06+21PM-bf68848f/best_model.pth",
    tts_config_path = "YourTTS-ULCA-T1-October-20-2023_06+21PM-bf68848f/config.json",
    use_cuda = True,
)

print(syn.tts_model.speaker_manager.name_to_id)
print(syn.tts_model.language_manager.name_to_id)

# exit()
wav = syn.tts(
    text = "ఈ గ్రామంలో ప్రజల ప్రధాన వృత్తి వ్యవసాయం.",
    speaker_name = "ulca_Telugu_mono_male",
    language_name = "tl",
    # text = "और आप उन्हें बता सकते हैं कि यह मेरी और से है",
    # speaker_name = "ulca_Hindi_mono_male",
    # language_name = "hi",
    # text = "this is a sample sentence generated using LJSpeech on Tacotron2 in CoquiTTS toolkit",
    # speaker_name = "ljspeech",
    # language_name = "en",
)

syn.save_wav(wav, "test.wav")
