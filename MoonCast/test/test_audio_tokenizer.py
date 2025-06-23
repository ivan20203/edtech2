import sys
sys.path.append('.')
from modules.audio_tokenizer.audio_tokenizer import get_audio_tokenizer
import torch

if __name__ == '__main__':
    audio_tokenizer = get_audio_tokenizer()

    input_wav = torch.zeros(1, 8000)
    semantic_token = audio_tokenizer.tokenize(input_wav)
    semantic_token = semantic_token.cpu().numpy().tolist()
    assert semantic_token == [[ 765, 3512, 7469, 7469, 7028, 2567, 6008, 7469, 6217, 2567, 7649, 7469,
         3292, 2567, 7649, 7469, 3292, 2567,  948, 7469, 3292, 2567,  948, 7469]]

    print("All tests passed!")