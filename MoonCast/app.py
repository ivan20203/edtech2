import gradio as gr
from huggingface_hub import snapshot_download 
snapshot_download(repo_id="jzq11111/mooncast", local_dir='./resources/')

from inference import Model
import base64

model = Model()
model.generate_config.max_new_tokens = 50 * 50 # no more than 50s per turn


def process_json_and_generate_audio(prompt_audio_role0_file, prompt_text_role0, prompt_audio_role1_file, prompt_text_role1, json_dialogue_input_str):
    try:
        print(json_dialogue_input_str, type(json_dialogue_input_str))
        print(prompt_audio_role0_file, prompt_text_role0, prompt_audio_role1_file, prompt_text_role1)
        # json_data = json.loads(json_dialogue_input_str)
        json_data = eval(json_dialogue_input_str.strip())
        print(json_data, type(json_data))    

        def validate_json(data):
            try:
                if not isinstance(data, list):
                    return "json must be a dictionary"
                cur_spk_should_be = 0
                for item in data:
                    if item['role'] != str(cur_spk_should_be):
                        return f"role should be {cur_spk_should_be} in item {item}"
                    cur_spk_should_be = 1 - cur_spk_should_be
                return None 
            except Exception as e:
                return str(e)


        validation_error = validate_json(json_data)
        if validation_error:
            raise gr.Error(validation_error)
        
        role_mapping = {
            "0": {
                "ref_audio": prompt_audio_role0_file,
                "ref_text": prompt_text_role0, 
            },
            "1": {
                "ref_audio": prompt_audio_role1_file, 
                "ref_text": prompt_text_role1,
            }
        }

        # 完整输入 JSON (你需要根据你的模型调整)
        model_input_json = {
            "role_mapping": role_mapping,
            "dialogue": json_data, # 从用户输入的 JSON 中获取 dialogue
        }
        print("模型推理输入 JSON:", model_input_json)


        # 4. **[重要] 调用你的 Model 类的 `inference` 方法**
        # audio_bytes = model.inference(model_input_json) 

        # 5. 返回音频 bytes 给 Gradio (Gradio 会自动处理音频 bytes 并播放)
        # return base64.b64decode(audio_bytes)
        for cur_chunk in model.inference(model_input_json, streaming=True):
            yield base64.b64decode(cur_chunk)

    except Exception as e:
        # return str(e) # 返回错误信息给 Gradio
        raise gr.Error(str(e))

title_en = "# PODCAST generator (supports English and Chinese)"
title_zh = "# 播客生成 (支持英文和中文)"

instruct_en = "## See [Github](https://github.com/jzq2000/MoonCast) for podcast script generation."
instruct_zh = "## 播客剧本生成请参考 [Github](https://github.com/jzq2000/MoonCast)。"

input_labels_en = ["Prompt Audio for Role 0", "Prompt Text for Role 0", "Prompt Audio for Role 1", "Prompt Text for Role 1", "Script JSON Input"]
input_labels_zh = ["角色 0 的 Prompt 音频", "角色 0 的 Prompt 文本", "角色 1 的 Prompt 音频", "角色 1 的 Prompt 文本", "剧本 JSON 输入"]

output_label_en = "Generated Audio Output (streaming)"
output_label_zh = "生成的音频输出(流式)"

example_prompt_text_role0_en = "Yeah, no, this is my backyard. It's never ending So just the way I like it. So social distancing has never been a problem."
example_prompt_text_role0_zh = "可以每天都骑并且可能会让你爱上骑车，然后通过爱上骑车的你省了很多很多钱。"
example_prompt_text_role1_en = "I'm doing great And. Look, it couldn't be any better than having you at your set, which is the outdoors."
example_prompt_text_role1_zh = "他最后就能让同样食材炒出来的菜味道大大提升。"

text_placeholder_zh = "对话轮流进行, 每轮最多50秒。文本越自然, 生成的音频效果越好。"
text_placeholder_en = "Dialogue alternates between roles. Limit each turn to a maximum of 50 seconds. The more natural the text, the better the generated audio."


example_json_en = '''[
       {
            "role": "0",
            "text": "In an awesome time, And, we're even gonna do a second episode too So. This is part one part two, coming at some point in the future There. We are.",
        },
       {
            "role": "1",
            "text": "I love it. So grateful Thank you So I'm really excited. That's awesome. Yeah.",
       },
       {
            "role": "0",
            "text": "All I was told, which is good because I don't want to really talk too much more is that you're really really into fitness and nutrition And overall holistic I love it Yes.",
       },
        {
            "role": "1",
            "text": "Yeah So I started around thirteen Okay But my parents were fitness instructors as well. Awesome So I came from the beginning, and now it's this transition into this wholeness because I had to chart my. Own path and they weren't into nutrition at all So I had to learn that part."
        }
]'''
example_json_zh = '''[
        {
            "role": "0",
            "text": "我觉得啊，就是经历了这么多年的经验， 就是补剂的作用就是九分的努力， 十分之一的补剂。 嗯，选的话肯定是九分更重要，但是我觉得补剂它能够让你九分的努力更加的有效率，更加的避免徒劳无功。 嗯，就是你，你你得先得真的锻炼，真的努力，真的健康饮食，然后再考虑补剂， 那你再加十十分之一的补剂的话，他可能就是说啊， 一半是心理作用，"
        },
        {
            "role": "1",
            "text": "对，其实很多时候心理作用是非常重要的。嗯，然后我每次用补剂的时候，我就会更加努力，就比如说我在健身之前我喝了一勺蛋白粉，我就会督促自己多练，"
        },
        {
            "role": "0",
            "text": "其实心理作用只要能实现你的预期目的就可以了。 就比如说给自行车链条加油， 它其实不是必要的，但是它可以让你骑行更顺畅， 然后提高你骑行的频率。"
        }   
    ]
'''

# examples_en = [
#     ['./en_prompt0.wav', example_prompt_text_role0_en, './en_prompt1.wav', example_prompt_text_role1_en, example_json_en]
# ]
# examples_zh = [
#     ['./zh_prompt0.wav', example_prompt_text_role0_zh, './zh_prompt1.wav', example_prompt_text_role1_zh, example_json_zh]
# ]

examples = [
    ['./en_prompt0.wav', example_prompt_text_role0_en, './en_prompt1.wav', example_prompt_text_role1_en, example_json_en],
    ['./zh_prompt0.wav', example_prompt_text_role0_zh, './zh_prompt1.wav', example_prompt_text_role1_zh, example_json_zh]
]

# -------------------- 更新界面元素的函数 --------------------
def update_ui_language(language):
    if language == "English":
        return  gr.update(value=title_en), \
                gr.update(value=instruct_en), \
                gr.update(label="UI Language"), \
                gr.update(label=input_labels_en[0]), \
                gr.update(label=input_labels_en[1]), \
                gr.update(label=input_labels_en[2]), \
                gr.update(label=input_labels_en[3]), \
                gr.update(label=input_labels_en[4], placeholder=text_placeholder_en), \
                gr.update(label=output_label_en), \
                gr.update(value="Submit"), \
                gr.update(label="Examples (Demonstration Use Only. Do Not Redistribute.)", headers=input_labels_en)
    
    elif language == "中文":
        return  gr.update(value=title_zh), \
                gr.update(value=instruct_zh), \
                gr.update(label="UI 语言"), \
                gr.update(label=input_labels_zh[0]), \
                gr.update(label=input_labels_zh[1]), \
                gr.update(label=input_labels_zh[2]), \
                gr.update(label=input_labels_zh[3]), \
                gr.update(label=input_labels_zh[4], placeholder=text_placeholder_zh), \
                gr.update(label=output_label_zh), \
                gr.update(value="提交"), \
                gr.update(label="示例 (仅用于展示，切勿私自传播。)", headers=input_labels_zh)

    else:
        raise ValueError("Invalid language selected")


audio_output = gr.Audio(label=output_label_en, streaming=True) 
css = """
.centered-title { /* CSS rule for centering title */
    text-align: center !important;
}
"""
# -------------------- Gradio 界面定义 (修改) --------------------
with gr.Blocks(css=css) as iface:

    title_output = gr.Markdown(value=title_zh, elem_classes="centered-title")
    instruct_output = gr.Markdown(value=instruct_zh)
    language_choice = gr.Radio(["中文", "English"], value="中文", label="UI语言") 

    with gr.Row(): # Main row to create two columns
        with gr.Column(scale=2): 
            json_input = gr.TextArea(label=input_labels_zh[4], lines=15, placeholder=text_placeholder_zh) # Dialogue JSON Input

        with gr.Column(scale=1): # Right column (narrower - scale=1) for prompt inputs
            audio_input_role0 = gr.Audio(type="filepath", label=input_labels_zh[0]) # Prompt Audio for Role 0
            text_input_role0 = gr.TextArea(label=input_labels_zh[1], lines=2) # Prompt Text for Role 0

        with gr.Column(scale=1): # 
            audio_input_role1 = gr.Audio(type="filepath", label=input_labels_zh[2]) # Prompt Audio for Role 1
            text_input_role1 = gr.TextArea(label=input_labels_zh[3], lines=2) # Prompt Text for Role 1

    examples_component = gr.Examples(
        examples=examples,
        inputs=[audio_input_role0, text_input_role0, audio_input_role1, text_input_role1, json_input],
        cache_examples=False,
        label="示例(仅用于展示，切勿私自传播。)",
    )
    
    submit_button = gr.Button("提交")
    
    submit_button.click(
        fn=process_json_and_generate_audio,
        inputs=[audio_input_role0, text_input_role0, audio_input_role1, text_input_role1, json_input],
        outputs=audio_output
    )
    audio_output.render()
    
    language_choice.change(
        fn=update_ui_language,
        inputs=language_choice,
        outputs=[title_output, instruct_output, language_choice, audio_input_role0, text_input_role0, audio_input_role1, text_input_role1, json_input, audio_output, submit_button, examples_component.dataset]
    )


iface.launch(share=True)
