from TTS.utils.synthesizer import Synthesizer
import matplotlib.pyplot as plt

# pretrained_path = "/data/saiakarsh/.dcache/tts/"
# vocoder_path = pretrained_path + "vocoder_models--en--ljspeech--hifigan_v2/"
vocoder_path = "../hifigan/hifigan_hindi_ulca-December-27-2023_05+20PM-72cf7551/"

model_path = "fast_pitch_hindi_ulca-December-21-2023_08+11PM-72cf7551/"
# model_path = "fast_pitch_telugu_ulca-December-21-2023_03+34PM-72cf7551/"
# model_path = pretrained_path + "tts_models--en--ljspeech--fast_pitch/"

syn = Synthesizer(
    tts_checkpoint = model_path + "best_model.pth",
    # tts_checkpoint = model_path + "model_file.pth",
    tts_config_path = model_path + "config.json",
    vocoder_checkpoint = vocoder_path + "checkpoint_10000.pth",
    # vocoder_checkpoint = vocoder_path + "model_file.pth",
    vocoder_config = vocoder_path + "config.json",
    use_cuda = True,
)

wav, outp = syn.tts(text = "अन्दर घुसते ही, उन्हें नशे में धुत एक व्यक्ति दिखा, वे उसके पास गए और पूछा")
print(outp['outputs']['model_outputs'].shape)
plt.imshow(outp['outputs']['model_outputs'].detach().cpu().numpy().T, origin='lower', aspect='auto')
plt.savefig("temp.png")
syn.save_wav(wav, "test_hi_voc.wav")
