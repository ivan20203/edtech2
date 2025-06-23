from abc import ABC
from abc import abstractmethod
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from typing import Any, Union
import numpy as np
from dataclasses import dataclass

def encode_pieces(sp_model: spm.SentencePieceProcessor, text: str, sample=False):
    """Encode text into sentence pieces. Only supports py3."""

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)

    return pieces


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))


class SPieceTokenizer(AbstractTokenizer):
    def __init__(self, spm_file: str):
        super().__init__('Sentence Piece')
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(spm_file)
        self.eod_id = self.get_token_id('</s>')

        self.special_ids = set([
            self.sp_model.pad_id(),
            self.sp_model.eos_id(),
            self.sp_model.bos_id(),
            self.sp_model.unk_id(),
            self.eod_id,
        ])

        # initialize index_2_bytes
        self._initialize_index_2_bytes()
    
    def encode_pieces(self, text: str, sample=False):
        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        return pieces

    def _initialize_index_2_bytes(self):
        proto = sp_pb2_model.ModelProto()
        proto.ParseFromString(self.sp_model.serialized_model_proto())
        self.index_2_numbytes = [0] * len(proto.pieces)
        for i, p in enumerate(proto.pieces):
            clean_piece = p.piece.replace('▁', '')
            self.index_2_numbytes[i] = len(clean_piece.encode('utf-8'))

    def set_add_dummy_prefix(self, add_dummy_prefix: bool = False):
        proto = sp_pb2_model.ModelProto()
        proto.ParseFromString(self.sp_model.serialized_model_proto())
        if proto.normalizer_spec.add_dummy_prefix != add_dummy_prefix:
            proto.normalizer_spec.add_dummy_prefix = add_dummy_prefix
            self.sp_model.LoadFromSerializedProto(proto.SerializeToString())
            print(f"> set add_dummy_prefix to {add_dummy_prefix} ...", flush=True)

    def add_special_id(self, token_id):
        self.special_ids.add(token_id)

    @property
    def has_dummy_prefix(self):
        pieces = self.sp_model.EncodeAsPieces("hello")
        return pieces[0].startswith('▁')

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        return self.sp_model

    def get_array_bytes(self, array):
        return sum(self.index_2_numbytes[i] if i < self.vocab_size else 2 for i in array)

    def tokenize(self, text):
        tokens = encode_pieces(self.sp_model, text)
        return self.convert_tokens_to_ids(tokens)
    
    def encode(self, text: str, bos: bool=False, eos: bool=False, **kwargs: Any) -> list[int]:
        tokens = self.encode_pieces(text)
        t = self.convert_tokens_to_ids(tokens)
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.sp_model.PieceToId(tokens)
        return [self.sp_model.PieceToId(token) for token in tokens]

    def detokenize(self, token_ids):
        if isinstance(token_ids, list):
            pieces = [self.sp_model.IdToPiece(id) for id in token_ids]
        else:
            pieces = [self.sp_model.IdToPiece(id) for id in token_ids.tolist()]
        return pieces
    
    def decode(self, token_ids: Union[int, list[int]], skip_special_tokens: bool = False) -> str:
        assert not skip_special_tokens, "skip_special_tokens is not supported"
        if isinstance(token_ids, (int, np.integer)):
            return self.detokenize([int(token_ids)])[0]
        return ''.join(self.detokenize(token_ids))

    def get_token_id(self, token):
        return self.sp_model.PieceToId(token)

    def inv_vocab(self):
        # TODO: to be implemented
        return {}

    def decode_pieces(self, pieces):
        return self.sp_model.DecodePieces(pieces)

    @property
    def eod(self):
        return self.eod_id

    @property
    def pad_id(self):
        return self.sp_model.pad_id()

    @property
    def eos_id(self):
        return self.sp_model.eos_id()

    @property
    def bos_id(self):
        return self.sp_model.bos_id()

    @property
    def unk_id(self):
        return self.sp_model.unk_id()
    
    @property
    def pad_token_id(self):
        return self.pad_id

    @property
    def eos_token_id(self):
        return self.eos_id

    
@dataclass
class ExtraTokens:
    msg_end: int
    user_msg_start: int
    assistant_msg_start: int
    name_end: int
    media_begin: int
    media_content: int
    media_end: int
    pad: int


def instantiate_extra_tokens(tokenizer: AbstractTokenizer):
    if isinstance(tokenizer, SPieceTokenizer):
        map_fn = lambda x: tokenizer.convert_tokens_to_ids(x)
    else:
        raise ValueError(f"Invalid tokenizer type: {type(tokenizer)}")

    return ExtraTokens(
        msg_end=map_fn('[extra_id_0]'),
        user_msg_start=map_fn('[extra_id_1]'),
        assistant_msg_start=map_fn('[extra_id_2]'),
        name_end=map_fn('[extra_id_12]'),
        media_begin=map_fn('[extra_id_13]'),
        media_content=map_fn('[extra_id_14]'),
        media_end=map_fn('[extra_id_15]'),
        pad=tokenizer.pad_id
    )

def get_tokenizer_and_extra_tokens():
    sp_model_path = "resources/tokenizer/160k.model"
    tokenizer = SPieceTokenizer(sp_model_path)
    tokenizer.set_add_dummy_prefix(False)
    extra_tokens = instantiate_extra_tokens(tokenizer)
    return tokenizer, extra_tokens
