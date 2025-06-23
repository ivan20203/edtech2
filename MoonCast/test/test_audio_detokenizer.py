import sys
import torch
sys.path.append('.')
from modules.audio_tokenizer.audio_tokenizer import get_audio_tokenizer
from modules.audio_detokenizer.audio_detokenizer import get_audio_detokenizer, detokenize
import torchaudio
import librosa

if __name__ == '__main__':
    audio_tokenizer = get_audio_tokenizer()
    audio_detokenizer = get_audio_detokenizer()

    input_wav_16k, _ = librosa.load("en_prompt0.wav", sr=16000)
    input_wav_24k, _ = librosa.load("en_prompt0.wav", sr=24000)

    prompt_sec = 1
    prompt_wav_16k = input_wav_16k[:16000*prompt_sec]
    prompt_wav_24k = input_wav_24k[:24000*prompt_sec]
    input_wav_16k = input_wav_16k[16000*prompt_sec:]
    input_wav_24k = input_wav_24k[24000*prompt_sec:]

    prompt_wav_24k = torch.tensor(prompt_wav_24k)[None, :].cuda()
    prompt_wav_16k = torch.tensor(prompt_wav_16k)[None, :].cuda()
    input_wav_24k = torch.tensor(input_wav_24k)[None, :].cuda()
    input_wav_16k = torch.tensor(input_wav_16k)[None, :].cuda()

    semantic_token = audio_tokenizer.tokenize(input_wav_16k)
    prompt_semantic_token = audio_tokenizer.tokenize(prompt_wav_16k)

    recon_wav = detokenize(audio_detokenizer, semantic_token, prompt_wav_24k, prompt_semantic_token)
    print(recon_wav.shape)    
    torchaudio.save("test/tmp_recon_en_prompt0.wav", recon_wav.cpu(), 24000)

    print("All tests passed!")