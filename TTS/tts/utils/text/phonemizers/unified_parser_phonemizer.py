import os
import re
import subprocess
from typing import Dict

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import is_tool

class Unified_Parser_Phonemizer(BasePhonemizer):
    _UNIF_PAR_DIR = "/data/saiakarsh/codes/unified/"
    _UNIF_PAR_COMM = './unified-parser "{sentence}" 1 0 0 0'
    _UNIF_PAR_COMM_VAL = 'valgrind -q --log-file="val.log" ./unified-parser "{sentence}" 1 0 0 0'
    
    def __init__(self, language: str = "hn", keep_puncs=True):
        super().__init__(language=language, keep_puncs=keep_puncs)
        self._punc_tok = "<_PUNC_>"
        self._punc_restore_list = []
    
    @staticmethod
    def name():
        return "unified_parser_phonemizer"

    def is_available(self) -> bool:
        return is_tool("valgrind") and os.path.isfile(os.path.join(self._UNIF_PAR_DIR, "unified-parser"))
    
    def version(self) -> str:
        return "3.0"

    @staticmethod
    def supported_languages() -> Dict:
        return {
            "ma": "malayalam",
            "ta": "tamil",
            "tl": "telugu",
            "ka": "kannada",
            "hn": "hindi",
            "be": "bengali",
            "gj": "gujarathi",
            "od": "odiya"
        }

    def _phonemize(self, text: str, separator: str = "") -> str:
        try: # first check without valgrind
            ph = subprocess.check_output(self._UNIF_PAR_COMM.format(sentence=text), shell=True, cwd=self._UNIF_PAR_DIR).decode().strip()
        except: # if that fails use valgrind (very slow)
            try:
                ph = subprocess.check_output(self._UNIF_PAR_COMM_VAL.format(sentence=text), shell=True, cwd=self._UNIF_PAR_DIR).decode().strip()
            except subprocess.CalledProcessError as e: # if valgrind also fails, ignore
                print(e)
                ph = ""
        if separator != "":
            return separator.join(ph.split())
        return ph

    def _match_group(self, txt):
        self._punc_restore_list.append(txt.group())
        return self._punc_tok

    def phonemize(self, text: str, separator: str = "", language: str = None) -> str:
        self._punc_restore_list = []
        text = re.sub(f"[{self._punctuator.puncs}]", self._match_group, text)
        text = text.split(self._punc_tok)
        ph = []
        for ind, txt in enumerate(text):
            txt = txt.strip()
            if txt:
                txt = self._phonemize(txt, separator=separator)
            ph.append(txt)
            if ind < len(self._punc_restore_list):
                ph.append(self._punc_restore_list[ind])
        text = ''.join(ph)
        return text

if __name__ == "__main__":
    # texts = "ఈ గ్రామంలో ప్రజల, ప్రధాన వృత్తి వ్యవసాయం."
    texts = "ఆడ గుడ<0c4d>లగూబ తల<0c4d>ల<0c3f>న<0c3f>"
    e = Unified_Parser_Phonemizer()
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())
    print(e.phonemize(texts, separator="|"))
