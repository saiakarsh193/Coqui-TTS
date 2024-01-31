import time
import torch
import numpy as np
from typing import List, Tuple

from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer

from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.tts.utils.helpers import sequence_mask
from TTS.tts.utils.synthesis import trim_silence
from TTS.utils.audio.numpy_transforms import save_wav

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
    def fp_inf(self, x, word_st_en, bump_map):
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
        for ind, word in enumerate(word_st_en):
            if bump_map[ind] == 1:
                o_dr[:, word[0]: word[1] + 1] *= 1.5
                o_dr[:, word[1] + 1] *= 2.0
        o_dr = torch.round(o_dr)
        print("durations:", o_dr.shape)
        y_lengths = o_dr.sum(1)

        # pitch predictor pass
        o_pitch = None
        if self.tts_model.args.use_pitch:
            o_pitch = self.tts_model.pitch_predictor(o_en, x_mask)
            for ind, word in enumerate(word_st_en):
                if bump_map[ind] == 1:
                    o_pitch[:, :, word[0]: word[1] + 1] *= 2.5
            o_pitch_emb = self.tts_model.pitch_emb(o_pitch)
            o_en = o_en + o_pitch_emb
            print("pitch:", o_pitch.shape)

        # energy predictor pass
        o_energy = None
        if self.tts_model.args.use_energy:
            o_energy = self.tts_model.energy_predictor(o_en, x_mask)
            for ind, word in enumerate(word_st_en):
                if bump_map[ind] == 1:
                    o_energy[:, :, word[0]: word[1] + 1] *= 1.5
            o_energy_emb = self.tts_model.energy_emb(o_energy)
            o_en = o_en + o_energy_emb
            print("energy:", o_energy.shape)

        # decoder pass
        o_de, attn = self.tts_model._forward_decoder(o_en, o_dr, x_mask, y_lengths, g=None)
        return o_de
    
    def text_to_ids(self, tokenizer: TTSTokenizer, text: List[str]) -> Tuple[List[int], List[Tuple[int, int]]]:
        # if tokenizer.text_cleaner is not None:
        #     text = tokenizer.text_cleaner(text)
        # if tokenizer.use_phonemes:
        #     text = tokenizer.phonemizer.phonemize(text, separator="", language=language)
        text_ids = []
        word_st_en = []
        for word in text:
            word_ids = [tokenizer.characters._char_to_id[char] for char in word if char in tokenizer.characters._char_to_id]
            word_st_en.append((len(text_ids), len(text_ids) + len(word_ids) - 1)) # word start and end index
            text_ids += word_ids + [tokenizer.characters._char_to_id[' ']] # add the word and a space
        text = text_ids[: -1] # to remove the last space
        # if tokenizer.add_blank:
        #     text = tokenizer.intersperse_blank_char(text, True)
        # if tokenizer.use_eos_bos:
        #     text = tokenizer.pad_with_bos_eos(text)
        return text, word_st_en

    def tts_syn(self, text: List[str], bump_map: List[int]) -> List[int]:
        assert len(text) == len(bump_map)
        text_ids, word_st_en = self.text_to_ids(self.tts_model.tokenizer, text)
        text_ids = torch.as_tensor(text_ids, dtype=torch.long, device=self.device).unsqueeze(0) # [1, T]
        mel_spec = self.fp_inf(text_ids, word_st_en, bump_map)
        mel_spec = mel_spec.squeeze(0).cpu().numpy().T # [C_spec, T]
        if self.use_gl:
            wav = self.tts_model.ap.inv_melspectrogram(mel_spec)
        else:
            wav = self.voc_syn(mel_spec)
        wav = wav.squeeze() # [1, T]
        if self.tts_config.audio.get("do_trim_silence", False):
            wav = trim_silence(wav, self.tts_model.ap)
        return list(wav)
    
    def voc_syn(self, mel_spec):
        mel_spec = self.tts_model.ap.denormalize(mel_spec) # [C, T]
        vocoder_input = self.vocoder_ap.normalize(mel_spec) # [C, T]
        vocoder_input = torch.tensor(vocoder_input).unsqueeze(0) # [1, C, T]
        wav = self.vocoder_model.inference(vocoder_input.to(self.device)).cpu().numpy()
        return wav
    
    def save_wav(self, wav: List[int], path: str, pipe_out=None) -> None:
        save_wav(wav=np.array(wav), path=path, sample_rate=self.output_sample_rate, pipe_out=pipe_out)


if __name__ == "__main__":
    vocoder_path = "../../hifigan/hifigan_hindi_ulca-December-29-2023_09+32AM-1f95b882/"
    model_path = "../fast_pitch_hindi_ulca-December-21-2023_08+11PM-72cf7551/"
    # model_path = "../engy_fast_pitch_hindi_ulca-December-29-2023_07+22AM-c9e1b371/"

    syn = FPSyn(
        tts_checkpoint = model_path + "best_model.pth",
        tts_config_path = model_path + "config.json",
        vocoder_checkpoint = vocoder_path + "checkpoint_30000.pth",
        vocoder_config_path = vocoder_path + "config.json",
        use_cuda = True,
    )

    # wav = syn.tts_syn(text = "अन्दर घुसते ही, उन्हें नशे में धुत एक व्यक्ति दिखा, वे उसके पास गए और पूछा")
    # wav = syn.tts_syn(
    #     text = ['अन्दर', 'घुसते', 'ही,', 'उन्हें', 'नशे', 'में', 'धुत', 'एक', 'व्यक्ति', 'दिखा,', 'वे', 'उसके', 'पास', 'गए', 'और', 'पूछा'],
    #     bump_map = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # )
    # syn.save_wav(wav, "test2.wav")

    # import json
    # with open("/data/saiakarsh/codes/whisperX/mt_align/outp.json", 'r') as f:
    #     inp = json.load(f)
    # print(inp['text'])
    # print(inp['bump_map'])
    # wav = syn.tts_syn(
    #     text = inp['text'],
    #     bump_map = inp['bump_map']
    # )
    # syn.save_wav(wav, "sst.wav")

    wav = syn.tts_syn(
        text = ['फिर', 'भी,', 'मनुष्यों', 'में', 'जो', 'डीएनए', 'पाया', 'जाता', 'है', 'वह', 'पहले', 'के', 'जीवों', 'में', 'पाए', 'जाने', 'वाले', 'डीएनए', 'से', 'थोड़ा', 'अलग', 'होता', 'है'],
        bump_map = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    syn.save_wav(wav, "sst.wav")
