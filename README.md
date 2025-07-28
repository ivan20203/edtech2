//////

This is my Education Technology project

there are essentially 2 codebases here. 

1 for Front end ui

another one for the TTS generation

I could not get the docker container to work on RunPod

So the frontend sadly does not allow the user to generate TTS

In any case the front end code is a ui design of what it should be

And the TTS code is under MoonDIA

Specifically under MoonDIA/trained_mapper

In trained_mapper if the environment is set up correctly 

it allows the generation of TTS audio. 

The limiting factor is the 15,000 output token limit in the code currently

but since its 4.1 you can easily edit that up to 60,000 for 1 hour+ audio.


In this TTS system generation we use MoonCast.
In my code a user inputs text and it calls GPT 4.1

GPT4.1 returns a script.

That script is then read line by line with MoonCast. 

It converts the script into Semantic Tokens and then creates audio. 

The hard part was making it scalable locally. 

I had to come up with a sliding window of 10 turns to maintain speaker consistency 
while not overflowing the prompt  and nuking my GPU.


Overall I am proud of my work. I was able to create a TTS system.

However I ran out of time and need to write 10 pages so I could not 
get the docker container to work and connect to the vercel front end. 



////////////////////////////


FRONT END CODE:
========================================

1. Sign up for accounts with the AI providers you want to use (e.g., OpenAI, Anthropic).
2. Obtain API keys for each provider.
3. Set the required environment variables as shown in the `.env.example` file, but in a new file called `.env`.
4. `pnpm install` to install the required Node dependencies.
5. `virtualenv venv` to create a virtual environment.
6. `source venv/bin/activate` to activate the virtual environment.
7. `pip install -r requirements.txt` to install the required Python dependencies.
8. `pnpm dev` to launch the development server.




TTS Code:
MoonDIA is the code.
need to install everything in environment.yml

there is requirements.txt in /MoonCast
requirements_mooncast_2wice.txt in /MoonDIA/trained_mapper
requirements_seq2seq.txt in /MoonDIA/trained_mapper

it uses the MoonCast conda environment which is under the MoonCast/Readme 
instructions. 

the main code is in /MoonDIA/trained_mapper

MoonCast_seed.py function call is :
python MoonCast_seed.py --input-file --duration 5

same standard for MoonCast_no_prompt.py 
and for MoonCast_seed_explainer.py

MoonCast_seed.py generates audio with 2 speakers.
MoonCast_no_prompt.py generates audio with random speakers throughout
MoonCast_seed_explainer.py generates audio with 2 speakers but explains more. 



This uses GPT4.1


Do this under MoonCast/
folder

conda create -n mooncast -y python=3.10
conda activate mooncast
pip install -r requirements.txt 
pip install flash-attn --no-build-isolation
pip install huggingface_hub
pip install gradio==5.22.0

python download_pretrain.py

 
flash-attn takes 5 hours to install

once everything is good pip install the 3 requirements
there is requirements.txt in /MoonCast
requirements_mooncast_2wice.txt in /MoonDIA/trained_mapper
requirements_seq2seq.txt in /MoonDIA/trained_mapper


You will need to switch to MoonDIA/CustomBuild
copy in the /resources/ from MoonCast

Also fill out the .env in the directory of /trained_mapper













