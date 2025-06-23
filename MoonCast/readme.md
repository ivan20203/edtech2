# MoonCast: High-Quality Zero-Shot Podcast Generation

<p align="center">
    <picture>
        <img src="./fig/logo.png" width="40%">
    </picture>
</p>

## Overview
Demo page: [demo](https://mooncastdemo.github.io)

Paper: [paper](https://arxiv.org/abs/2503.14345)

2025/03/26 UPDATE: We also host a [HuggingFace space](https://huggingface.co/spaces/jzq11111/mooncast) for testing audio generation.

We open-source this system to advance the field of human-like speech synthesis. Our goal is to create more natural and expressive synthetic voices that bridge the gap between machines and humans. We hope this project will inspire researchers and developers to explore new possibilities in voice technology. We warmly welcome contributions from anyone interested in this project. Whether through code, documentation, feedback, or sharing your insights, every input helps make this project better.


## Environment Setup
- Create conda environment.

``` sh
conda create -n mooncast -y python=3.10
conda activate mooncast
pip install -r requirements.txt 
pip install flash-attn --no-build-isolation
pip install huggingface_hub
pip install gradio==5.22.0
```

- Download the pretrained weights.
``` sh
python download_pretrain.py
```

## Example Usage

### Script Generation
For podcast script generation, we utilize specific LLM prompts defined in ``zh_llmprompt_script_gen.py`` (Chinese) and ``en_llmprompt_script_gen.py`` (English). We have selected the [Gemini 2.0 Pro Experimental 02-05](https://cloud.google.com/vertex-ai/generative-ai/docs/gemini-v2#2.0-pro) model for this task, favoring its ability to produce conversational language, design natural dialogue, and offer broad topic coverage. Our process involves two stages: first, we generate a concise summary by providing the input knowledge source as an attachment along with the ``INPUT2BRIEF`` prompt. Subsequently, this summary, paired with the ``BRIEF2SCRIPT`` prompt, is used to generate the final podcast script in JSON format.

### Speech Generation
The audio prompts used in this project are sourced from publicly available podcast segments and are intended solely for demonstration purposes. Redistribution of these audio files, whether in their original form or as generated audio, is strictly prohibited. If you have any concerns or questions regarding the use of these audio files, please contact us at juzeqian@mail.ustc.edu.cn

```sh
CUDA_VISIBLE_DEVICIES=0 python inference.py
```

2025/03/26 UPDATE: We add a Gradio-based user interface for audio generation. Deploy it locally using:

```sh
CUDA_VISIBLE_DEVICIES=0 python app.py
```

## Disclaimer  
This project is intended for **research purposes only**. We strongly encourage users to **use this project and its generated audio responsibly**. **We are not responsible for any misuse or abuse of this project**. By using this project, you agree to comply with all applicable laws and ethical guidelines.