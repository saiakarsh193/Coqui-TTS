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
from TTS.tts.utils.helpers import sequence_mask

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

    @torch.no_grad()
    def fp_inference(self, x):
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).to(x.dtype).float()
        # encoder pass
        x_emb = self.tts_model.emb(x) # [B, T, C]
        o_en = self.tts_model.encoder(torch.transpose(x_emb, 1, -1), x_mask, g=None)
        # duration predictor pass
        o_dr_log = self.tts_model.duration_predictor(o_en.squeeze(), x_mask)
        # format durations
        o_dr = (torch.exp(o_dr_log) - 1) * x_mask * self.tts_model.length_scale
        o_dr[o_dr < 1] = 1.0
        o_dr = torch.round(o_dr).squeeze(1)
        print("durations:", o_dr.shape)
        # o_dr[:, :40] *= 2
        # o_dr = torch.round(o_dr)
        # print("new_durations:", o_dr)
        y_lengths = o_dr.sum(1)

        # pitch predictor pass
        o_pitch = None
        if self.tts_model.args.use_pitch:
            o_pitch = self.tts_model.pitch_predictor(o_en, x_mask)
            o_pitch_emb = self.tts_model.pitch_emb(o_pitch)
            o_en = o_en + o_pitch_emb
            # o_pitch = torch.full(o_pitch.shape, fill_value=torch.mean(o_pitch))
            # o_pitch[:, :, :20] = 2
            print("pitch:", o_pitch.shape)

        # energy predictor pass
        o_energy = None
        if self.tts_model.args.use_energy:
            o_energy = self.tts_model.energy_predictor(o_en, x_mask)
            o_energy_emb = self.tts_model.energy_emb(o_energy)
            o_en = o_en + o_energy_emb
            print("energy:", o_energy.shape)

        # decoder pass
        o_de, attn = self.tts_model._forward_decoder(o_en, o_dr, x_mask, y_lengths, g=None)
        return o_de
    
    def text_to_ids(self, tokenizer, text):
        if tokenizer.text_cleaner is not None:
            text = tokenizer.text_cleaner(text)
        # if tokenizer.use_phonemes:
        #     text = tokenizer.phonemizer.phonemize(text, separator="", language=language)
        token_ids = []
        for char in text:
            if char in tokenizer.characters._char_to_id:
                idx = tokenizer.characters._char_to_id[char]
                token_ids.append(idx)
        text = token_ids
        # if tokenizer.add_blank:
        #     text = tokenizer.intersperse_blank_char(text, True)
        # if tokenizer.use_eos_bos:
        #     text = tokenizer.pad_with_bos_eos(text)
        return text

    def tts_syn(self, text):
        text_inputs = self.text_to_ids(self.tts_model.tokenizer, text)
        text_inputs = torch.as_tensor(text_inputs, dtype=torch.long, device=self.device).unsqueeze(0) # [1, T]
        mel_spec = self.fp_inference(text_inputs)
        mel_spec = mel_spec.squeeze(0).cpu().numpy().T # [C_spec, T]
        if self.use_gl:
            wav = self.tts_model.ap.inv_melspectrogram(mel_spec)
        else:
            wav = self.voc_syn(mel_spec)
        wav = wav.squeeze() # [1, T]
        if self.tts_config.audio.get("do_trim_silence", False):
            wav = trim_silence(wav, self.tts_model.ap)
        return wav
    
    def voc_syn(self, mel_spec):
        mel_spec = self.tts_model.ap.denormalize(mel_spec) # [C, T]
        vocoder_input = self.vocoder_ap.normalize(mel_spec) # [C, T]
        vocoder_input = torch.tensor(vocoder_input).unsqueeze(0) # [1, C, T]
        wav = self.vocoder_model.inference(vocoder_input.to(self.device)).cpu().numpy()
        return wav

    def tts(self, text: str = "") -> List[int]:
        start_time = time.time()
        wav = list(self.tts_syn(text))
        process_time = time.time() - start_time
        audio_time = len(wav) / self.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wav
    
    def save_wav(self, wav: List[int], path: str, pipe_out=None) -> None:
        save_wav(wav=np.array(wav), path=path, sample_rate=self.output_sample_rate, pipe_out=pipe_out)


if __name__ == "__main__":
    vocoder_path = "../../hifigan/hifigan_hindi_ulca-December-29-2023_09+32AM-1f95b882/"
    # model_path = "../fast_pitch_hindi_ulca-December-21-2023_08+11PM-72cf7551/"
    model_path = "../engy_fast_pitch_hindi_ulca-December-29-2023_07+22AM-c9e1b371/"

    syn = FPSyn(
        tts_checkpoint = model_path + "best_model.pth",
        tts_config_path = model_path + "config.json",
        # vocoder_checkpoint = vocoder_path + "checkpoint_30000.pth",
        # vocoder_config_path = vocoder_path + "config.json",
        use_cuda = True,
    )

    wav = syn.tts(text = "अन्दर घुसते ही, उन्हें नशे में धुत एक व्यक्ति दिखा, वे उसके पास गए और पूछा")
    # syn.save_wav(wav, "test.wav")
