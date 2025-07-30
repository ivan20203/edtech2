# Education Technology Project

A comprehensive education technology platform featuring a modern frontend UI and an advanced Text-to-Speech (TTS) generation system powered by MoonCast.

## 🏗️ Project Overview

This project consists of two main components:

1. **Frontend UI** - A modern web interface for user interaction
2. **TTS Generation System** - Advanced audio generation using MoonCast and GPT-4.1

### Current Status

⚠️ **Note**: The Docker container integration with RunPod is currently not functional, so the frontend cannot generate TTS directly. However, the frontend serves as a complete UI design showcase, and the TTS system works independently.

## 🎯 TTS System Features

The TTS system under `MoonDIA/trained_mapper` provides:

- **Multi-speaker audio generation** with consistent voice characteristics
- **GPT-4.1 integration** for intelligent script generation
- **Scalable local processing** with sliding window optimization
- **Semantic token conversion** for high-quality audio output
- **Configurable duration** (currently limited to 15,000 tokens, expandable to 60,000+ for 1+ hour audio)

### How It Works

1. User inputs text → GPT-4.1 generates a script
2. Script is processed line-by-line with MoonCast
3. Text is converted to semantic tokens
4. High-quality audio is generated with speaker consistency

## 🚀 Quick Start

### Frontend Setup

1. **Install Dependencies**
   ```bash
   pnpm install
   ```

2. **Set Up Environment**
   - Sign up for AI provider accounts (OpenAI, Anthropic, etc.)
   - Obtain API keys
   - Copy `.env.example` to `.env` and fill in your API keys

3. **Create Python Environment**
   ```bash
   virtualenv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Launch Development Server**
   ```bash
   pnpm dev
   ```

### MoonDIA TTS Setup

#### Prerequisites
- Conda installed on your system
- CUDA-compatible GPU (recommended)
- At least 8GB GPU memory

#### Step-by-Step Installation

1. **Create and Activate Conda Environment**
   ```bash
   conda env create -f environment.yml
   conda activate mooncast
   ```

2. **Install MoonCast Dependencies**
   ```bash
   cd MoonCast/
   pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   pip install huggingface_hub
   pip install gradio==5.22.0
   ```

   ⏱️ **Note**: `flash-attn` installation can take up to 5 hours

3. **Download Pre-trained Models**
   ```bash
   python download_pretrain.py
   ```

4. **Set Up MoonDIA**
   ```bash
   cd ../MoonDIA/
   
   # Copy resources from MoonCast
   cp -r ../MoonCast/resources/ CustomBuild/
   
   # Install additional requirements
   cd trained_mapper/
   pip install -r requirements_mooncast_2wice.txt
   pip install -r requirements_seq2seq.txt
   ```

5. **Configure Environment**
   ```bash
   # Create and configure .env file in trained_mapper directory
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## 🎵 Using the TTS System

The main TTS code is located in `MoonDIA/trained_mapper/` with three main scripts:

### Available Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `MoonCast_seed.py` | Generates audio with 2 consistent speakers | `python MoonCast_seed.py --input-file <file> --duration 5` |
| `MoonCast_no_prompt.py` | Generates audio with random speakers throughout | `python MoonCast_no_prompt.py --input-file <file> --duration 5` |
| `MoonCast_seed_explainer.py` | Generates audio with 2 speakers + explanations | `python MoonCast_seed_explainer.py --input-file <file> --duration 5` |

### Example Usage
```bash
cd MoonDIA/trained_mapper/
python MoonCast_seed.py --input-file script.txt --duration 10
```

## 🔧 Technical Details

### Architecture
- **Frontend**: Modern web UI with AI provider integration
- **TTS Engine**: MoonCast-based semantic token generation
- **AI Integration**: GPT-4.1 for intelligent script processing
- **Optimization**: 10-turn sliding window for speaker consistency

### Performance Notes
- Current token limit: 15,000 (expandable to 60,000+)
- GPU memory requirement: 8GB+ recommended
- Processing time varies based on input length and GPU capability

## 📁 Project Structure

```
edtech/
├── README.md
├── environment.yml
├── requirements.txt
├── .env.example
├── MoonCast/
│   ├── resources/
│   ├── requirements.txt
│   └── download_pretrain.py
└── MoonDIA/
    ├── CustomBuild/
    │   └── resources/  # Copied from MoonCast
    └── trained_mapper/
        ├── MoonCast_seed.py
        ├── MoonCast_no_prompt.py
        ├── MoonCast_seed_explainer.py
        ├── requirements_mooncast_2wice.txt
        ├── requirements_seq2seq.txt
        └── .env
```

## 🤝 Contributing

This project demonstrates advanced TTS capabilities with local processing. The frontend serves as a design reference for future integration.

## 📝 License

This project is part of an educational technology initiative.

---














