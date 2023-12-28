import time
import torch
import numpy as np
from typing import List

from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.tts.utils.synthesis import trim_silence
from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
from TTS.vocoder.models import setup_model as setup_vocoder_model

class FPSyn:
    def __init__(
        self,
        tts_checkpoint: str = "",
        tts_config_path: str = "",
        vocoder_checkpoint: str = "",
        vocoder_config_path: str = "",
        use_cuda: bool = False,
    ) -> None:
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config_path = vocoder_config_path
        self.use_gl = (self.vocoder_checkpoint == "")
        self.use_cuda = use_cuda
        self.device = "cuda:0" if self.use_cuda else "cpu"
    
        if self.tts_checkpoint:
            self.tts_config = load_config(self.tts_config_path)
            self.tts_model = setup_tts_model(config=self.tts_config)
            self.tts_model.load_checkpoint(self.tts_config, self.tts_checkpoint, eval=True)
            if self.use_cuda:
                self.tts_model.cuda()
            self.output_sample_rate = self.tts_config.audio["sample_rate"]

        if self.vocoder_checkpoint:
            self.vocoder_config = load_config(self.vocoder_config_path)
            self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config.audio)
            self.vocoder_model = setup_vocoder_model(self.vocoder_config)
            self.vocoder_model.load_checkpoint(self.vocoder_config, self.vocoder_checkpoint, eval=True)
            if self.use_cuda:
                self.vocoder_model.cuda()
            self.output_sample_rate = self.vocoder_config.audio["sample_rate"]
    
    def tts_syn(self, text):
        text_inputs = np.asarray(
            self.tts_model.tokenizer.text_to_ids(text),
            dtype=np.int32,
        ) # [T]
        text_inputs = torch.as_tensor(text_inputs, dtype=torch.long, device=self.device)
        text_inputs = text_inputs.unsqueeze(0) # [1, T]
        input_lengths = torch.tensor(text_inputs.shape[1:2]).to(self.device)
        outputs = self.tts_model.inference(
            text_inputs,
            aux_input={
                "x_lengths": input_lengths,
                "speaker_ids": None,
                "d_vectors": None,
            },
        )
        mel_spec = outputs["model_outputs"][0].data.cpu().numpy().T # [C_spec, T]
        alignments = outputs["alignments"] # [1, T_de, T_en]
        if self.use_gl:
            wav = self.tts_model.ap.inv_melspectrogram(mel_spec)
        else:
            wav = self.voc_syn(mel_spec)
        wav = wav.squeeze() # [1, T]
        if "do_trim_silence" in self.tts_config.audio and self.tts_config.audio["do_trim_silence"]:
            wav = trim_silence(wav, self.tts_model.ap)
        return_dict = {
            "wav": wav,
            "alignments": alignments,
            "text_inputs": text_inputs,
            "outputs": outputs,
        }
        return return_dict
    
    def voc_syn(self, mel_spec):
        mel_spec = self.tts_model.ap.denormalize(mel_spec) # [C, T]
        vocoder_input = self.vocoder_ap.normalize(mel_spec) # [C, T]
        vocoder_input = torch.tensor(vocoder_input).unsqueeze(0) # [1, C, T]
        wav = self.vocoder_model.inference(vocoder_input.to(self.device)).cpu().numpy()
        return wav

    def tts(self, text: str = "") -> List[int]:
        start_time = time.time()
        outputs = self.tts_syn(text)
        wav = list(outputs["wav"])
        process_time = time.time() - start_time
        audio_time = len(wav) / self.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wav
        # return wav, outputs
    
    def save_wav(self, wav: List[int], path: str, pipe_out=None) -> None:
        save_wav(wav=np.array(wav), path=path, sample_rate=self.output_sample_rate, pipe_out=pipe_out)


if __name__ == "__main__":
    pretrained_path = "/data/saiakarsh/.dcache/tts/"
    vocoder_path = pretrained_path + "vocoder_models--en--ljspeech--hifigan_v2/"
    # vocoder_path = "../hifigan/hifigan_hindi_ulca-December-27-2023_05+20PM-72cf7551/"

    model_path = "fast_pitch_hindi_ulca_backup/"
    # model_path = "fast_pitch_hindi_ulca-December-21-2023_08+11PM-72cf7551/"
    # model_path = "fast_pitch_telugu_ulca-December-21-2023_03+34PM-72cf7551/"

    syn = FPSyn(
        tts_checkpoint = model_path + "best_model.pth",
        tts_config_path = model_path + "config.json",
        vocoder_checkpoint = vocoder_path + "model_file.pth",
        vocoder_config_path = vocoder_path + "config.json",
        use_cuda = True,
    )

    wav = syn.tts(text = "अन्दर घुसते ही, उन्हें नशे में धुत एक व्यक्ति दिखा, वे उसके पास गए और पूछा")
    syn.save_wav(wav, "test_dum.wav")
