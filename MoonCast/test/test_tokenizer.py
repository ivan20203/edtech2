import sys
sys.path.append('.')
from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens



if __name__ == '__main__':
    tokenizer, extra_tokens = get_tokenizer_and_extra_tokens()

    assert tokenizer.encode("user") == [1495]
    assert tokenizer.decode([1495]) == "user"

    assert tokenizer.encode("0") == [501]
    assert tokenizer.decode([501]) == "0"

    assert tokenizer.encode("1") == [503]
    assert tokenizer.decode([503]) == "1"

    assert tokenizer.encode("assistant") == [110866]
    assert tokenizer.decode([110866]) == "assistant"

    assert tokenizer.encode("audio") == [26229]
    assert tokenizer.decode([26229]) == "audio"
    

    assert extra_tokens.msg_end == 260
    assert extra_tokens.user_msg_start == 261
    assert extra_tokens.assistant_msg_start == 262
    assert extra_tokens.name_end == 272
    assert extra_tokens.media_begin == 273
    assert extra_tokens.media_content == 274
    assert extra_tokens.media_end == 275

    assert [tokenizer.convert_tokens_to_ids(i) for i in ['<0x0A>', '</s>', '[extra_id_0]']] == [14, 1, 260]

    print("All tests passed!")
    