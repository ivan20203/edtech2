import torch
import librosa
import yaml
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
import safetensors
import accelerate
import soundfile as sf
import math
from einops import rearrange
from modules.audio_tokenizer.rep_codec import RepCodec


class AudioTokenizer(object):
    def __init__(self, **kwargs):
        self.device = kwargs.pop('device')
        print(self.device)
        # tokenize
        feat_stats = kwargs.pop('feat_stats')
        feat_stats = torch.load(feat_stats, map_location='cpu')
        self.feat_mean = feat_stats['mean']
        self.feat_std = torch.sqrt(feat_stats['var'])
        wav2vec_ckpt = kwargs.pop("wav2vec_ckpt")
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(wav2vec_ckpt)
        self.semantic_model.eval()
        self.semantic_model.to(self.device)
        self.semantic_processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        self.semantic_codec = RepCodec()
        self.semantic_codec.eval()
        pretrained_path = kwargs.pop("semantic_codec_ckpt") 
        safetensors.torch.load_model(self.semantic_codec, pretrained_path)
        self.semantic_codec.to(self.device)

        self.max_length = 2048
        

    @torch.no_grad()
    def tokenize(self, speech):
        # Input:
        # speech: torch tensor, shape[B, N_speech]
        # Output:
        # semantic token: torch tensor, shape[B, N]

        inputs = self.semantic_processor(speech.cpu(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        seg_num = math.ceil(input_features.shape[1] / self.max_length)
        pad_num = seg_num * self.max_length - input_features.shape[1]
        input_features = torch.nn.functional.pad(input_features, (0, 0, 0, pad_num, 0,0), value=0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_num, 0, 0), value=0)
        input_features = rearrange(input_features, "b (s n) d -> (b s) n d", s =seg_num)
        attention_mask = rearrange(attention_mask, "b (s n) -> (b s) n", s=seg_num)


        feats = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = feats.hidden_states[17]  
        feat = rearrange(feat, "(b s) n d -> b (s n) d", s=seg_num)
        feat = feat[:, :feat.shape[1]-pad_num, :]
        feat = (feat - self.feat_mean.to(feat)) / self.feat_std.to(feat)
        semantic_token, _ = self.semantic_codec.quantize(feat)  
        return semantic_token

def get_audio_tokenizer():
    config = dict()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['feat_stats'] = 'resources/audio_tokenizer/stats.pt'
    config['wav2vec_ckpt'] = 'facebook/w2v-bert-2.0'
    config['semantic_codec_ckpt'] = 'resources/audio_tokenizer/model.safetensors'
    audio_tokenizer = AudioTokenizer(**config)
    return audio_tokenizer

