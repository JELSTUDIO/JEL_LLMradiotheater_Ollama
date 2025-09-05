#venv311\scripts\activate

#Ollama server seems to default to a context-window length of 4096, so this script assumes that much is available. Change the max_tokens variable on line 607

#version2.3.7, Chatterbox multilingual version

#import logging
#logging.basicConfig(level=logging.DEBUG)

import tkinter as tk
from tkinter import scrolledtext, simpledialog
import threading
import requests
import time
import json
import os
import numpy as np
import io
import shutil
import wave
import re
import logging
import torch  # Needed for tensor handling
import pygame  # For advanced audio playback with mixing and looping
from datetime import datetime  # For timestamped filenames

import soundfile as sf

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Clear the file when the script starts
open("llm_conversation.txt", "w", encoding="utf-8").close()

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Attempt to import Chatterbox TTS
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
except ModuleNotFoundError as e:
    print(f"Error: Could not import Chatterbox TTS module. Ensure 'chatterbox-tts' is installed.\n{e}")
    exit(1)

# Contraction expansion dictionary, only needed for Coqui-TTS
#CONTRACTIONS = {
#    "we're": "we are",
#    "you're": "you are",
#    "they're": "they are",
#    "I'm": "I am",
#    "he's": "he is",
#    "she's": "she is",
#    "it's": "it is",
#    "don't": "do not",
#    "can't": "cannot",
#    "won't": "will not",
#    "I'll": "I will",
#    "you'll": "you will",
#    "he'll": "he will",
#    "she'll": "she will",
#    "they'll": "they will",
#    "I've": "I have",
#    "you've": "you have",
#    "we've": "we have",
#    "they've": "they have",
#    "isn't": "is not",
#    "aren't": "are not",
#    "wasn't": "was not",
#    "weren't": "were not",
#    "doesn't": "does not",
#    "didn't": "did not",
#    "hasn't": "has not",
#    "haven't": "have not",
#    "hadn't": "had not",
#}

# Contraction expansion dictionary, only needed for Coqui-TTS, so empty while using Chatterbox
CONTRACTIONS = {}

#old preprocess
#def preprocess_text(text):
#    """Preprocess text for TTS: expand contractions and remove special symbols."""
#    text_lower = text.lower()
#    for contraction, expanded in CONTRACTIONS.items():
#        text_lower = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text_lower, flags=re.IGNORECASE)
#    text_clean = re.sub(r'\*([^\*]+)\*', r'\1', text_lower)
#    text_clean = re.sub(r'\*+', '', text_lower)  # Removes all * characters
#    text_clean = re.sub(r'[^\w\s,.!?]', '', text_clean)
#    return text_clean

#new preprocess, when lower-case is needed (Not used for Chatterbox)
#def preprocess_text(text):
#    """Preprocess text for TTS: expand contractions, preserving punctuation for chunking."""
#    text_lower = text.lower()
#    for contraction, expanded in CONTRACTIONS.items():
#        text_lower = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text_lower, flags=re.IGNORECASE)
#    return text_lower  # Preserve all punctuation for chunking

#new preprocess, when case-change is not needed (Used for Chatterbox that can handle both upper- and lower-case)
def preprocess_text(text):
    """Preprocess text for TTS: expand contractions, preserving original case and punctuation for chunking."""
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text, flags=re.IGNORECASE)
    return text  # Return original text with no case change

def clean_response(text):
    """Remove unwanted prefixes like 'Assistant:' from model responses."""
    #print(f"Raw response: {text}") #For debug purposes to see if cleaning works. Switch OFF this one print line when not needed
    text = re.sub(r'\s*(Assistant|Opponent|User):\s*', '', text, flags=re.IGNORECASE)  # Remove prefixes
    text = re.sub(r'^"|"$', '', text)  # Remove leading and trailing double quotes
    #print(f"Cleaned response: {text}") #For debug purposes to see if cleaning works. Switch OFF this one print line when not needed

    """Remove unwanted prefixes like 'Assistant:' or 'Opponent:' and mid-text occurrences from model responses."""
    #text = re.sub(r'\s*(Assistant|Opponent|User):\s*', '', text, flags=re.IGNORECASE)  # Remove anywhere in text

    return text.strip()




class LLMDuetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Radio Theater / version2.3.7 / 20250905 / Chatterbox multi-lingual")
        self.root.geometry("800x900")

        #self.model_1 = "hammerai/rocinante-v1.1:12b-q4_k_m" #ends up speaking for both green and blue, so don't use
        #self.model_1 = "mistral-nemo:12b" #ends up speaking for both green and blue, so don't use
        #self.model_1 = "qwen3:14b"
        #self.model_1 = "llama3.1:8b"
        #self.model_1 = "wizardlm-uncensored:13b"
        #self.model_1 = "deepcoder:1.5b" #too small, too stupid
        #self.model_1 = "gpt-oss:20b" #too big for my GPU :(
        self.model_1 = "gemma3:12b"
        #self.model_1 = "gemma3:4b"

        #self.model_2 = "hammerai/rocinante-v1.1:12b-q4_k_m"
        #self.model_2 = "qwen3:14b"
        #self.model_2 = "wizardlm-uncensored:13b"
        #self.model_2 = "gpt-oss:20b" #too big for my GPU :(
        self.model_2 = "gemma3:12b"


        #Green LLM
        self.system_prompt_1 = """
            "You are the wife. Choose your own name. "
            "Today is another day in your marriage. Have a natural conversation as a married couple. "
            "Only use dialogue as if it was spoken to somebody, so no internal dialogue or third-person descriptive language."
        """

        #Blue LLM
        self.system_prompt_2 = """
            "You are the husband. Choose your own name. "
            "Today is another day in your marriage. Have a natural conversation as a married couple. "
            "Only use dialogue as if it was spoken to somebody, so no internal dialogue or third-person descriptive language."
        """

        self.tts_enabled = True
        self.running = False
        self.paused = False
        #self.history_1 = [{"role": "user", "content": "Hello? Who's calling please?"}] #Green LLM
        self.history_1 = [{"role": "user", "content": """
            Who are you? Please introduce yourself.
        """}] #Green LLM
        #self.history_2 = [{"role": "user", "content": ""}] #Blue LLM
        self.history_2 = [{"role": "user", "content": """
            Who are you? Please introduce yourself.
        """}] #Blue LLM
        #self.history_1 = [{"role": "user", "content": "Hello, I am box and I am ready for you."}]
        #self.history_2 = [{"role": "user", "content": "I'm not box, but I'm also ready."}]
        self.speaker = 0
        self.wrapping_up = False
        self.wrap_stage = 0
        self.start_time = None

        self.totalaccumulatedspeechduration_elapsed_time = 0

        self.timer_lock = True  # Should timer cut talk? ON by default to enable timer stop. Now user can switch OFF timer-stop if conversation is too good to stop when timer runs out

        self.use_speech_time = True  # or False, depending on button state. Should timer track content-time or real-time? ON for content-time by default
        
        self.timer_duration_seconds = 1500  # Default timer duration. 1500 = 25 minutes

        # TTS settings for each speaker. Default value is 0.5 for both values
        # these defaults are for the default voices
        #self.exaggeration_1 = 0.75  # GreenLLM: how expressive
        #self.cfg_weight_1 = 0.3    # GreenLLM: pacing
        #self.exaggeration_2 = 0.85   # BlueLLM: how expressive
        #self.cfg_weight_2 = 0.1     # BlueLLM: pacing

        # Default Voice clone files (ensure these files exist in the working directory)
        #self.voice_female = "voice_female.wav"
        #self.voice_male = "voice_male.wav"

        self.voice_female = "voice_female.wav"
        self.exaggeration_1 = 0.75  # GreenLLM: how expressive, default is 0.5
        self.cfg_weight_1 = 0.3    # GreenLLM: pacing, default is 0.5
        self.language_id_1 = "en"

        # Supported Languages:
        # Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)

        self.voice_male = "voice_male.wav"
        self.exaggeration_2 = 0.85   # BlueLLM: how expressive, default is 0.5
        self.cfg_weight_2 = 0.1     # BlueLLM: pacing, default is 0.5
        self.language_id_2 = "en"




        # Voice clone files (ensure these files exist in the working directory)

        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\MartinGeeson_RX9JEL_PodcastReady_-24dB.wav"
        #self.voice_female = r"D:\TEMP\ChatterboxTEST\PodcastTestVoices\ccc17_whileshepherdswatched_at_64kb - BookAngel7.wav"

        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\pratibhanair_Hindi-female.wav"
        #self.exaggeration_1 = 1.25  # GreenLLM: how expressive
        #self.cfg_weight_1 = 0.35    # GreenLLM: pacing

        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\doonaboon(SoundsLikeHAL9000)_RX9JEL_PodcastReady_-24dB.wav"
        #self.exaggeration_1 = 0.25  # GreenLLM: how expressive
        #self.cfg_weight_1 = 0.05    # GreenLLM: pacing

        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\LeneGrebo_BonBonLand_TunnelBane_mod02.wav"
        #self.exaggeration_1 = 0.00  # GreenLLM: how expressive
        #self.cfg_weight_1 = 1.0    # GreenLLM: pacing

        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\ArianeRibeiro_Portuguese-female.wav"
        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\MargaretEspaillat_(Female-VeryClearSound).wav"
        #self.exaggeration_1 = 1.0   # how expressive
        #self.cfg_weight_1 = 1.0     # pacing

        #professor
        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\AshleighJane_English-female.wav"
        #self.exaggeration_1 = 0.5   # how expressive
        #self.cfg_weight_1 = 0.7     # pacing

        #aggressive
        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\AshleighJane_English-female.wav"
        #self.exaggeration_1 = 0.75   # how expressive
        #self.cfg_weight_1 = 0.9     # pacing

        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama_Multilingual\voices\Danish_MEL_PodcastReady.wav"
        #self.exaggeration_1 = 0.10  # GreenLLM: how expressive
        #self.cfg_weight_1 = 0.8    # GreenLLM: pacing
        #self.language_id_1 = "da"
        
        #self.voice_male = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\Newgatenovelist(female,VerySoft).wav"
        #self.exaggeration_2 = 0.5   # how expressive, default is 0.95
        #self.cfg_weight_2 = 0.5     # pacing, default is 0.05

        #grumpy man and LLM defaults below

        #self.voice_female = r"D:\AI\JEL_LLMradiotheater_Ollama\voices\CynthiaMoyer_(Female)_RX9JEL_24K32B_PodcastReady_-24dB.wav" #default English LLM
        #self.exaggeration_1 = 0.75  # GreenLLM: how expressive
        #self.cfg_weight_1 = 0.35    # GreenLLM: pacing

        #self.voice_female = "LeonardWilson(1930-2024)_RX9JEL_24K32B_PodcastReady_-24dB.wav"
        #self.exaggeration_1 = 0.95   # how expressive, default is 0.95
        #self.cfg_weight_1 = 0.05     # pacing, default is 0.05

        #Green LLM above

        #Blue LLM below

        #self.voice_male = "LeonardWilson(1930-2024)_RX9JEL_24K32B_PodcastReady_-24dB.wav"
        #self.exaggeration_2 = 0.95   # how expressive, default is 0.95
        #self.cfg_weight_2 = 0.05     # pacing, default is 0.05

        #self.voice_male = r"D:\AI\JEL_LLMradiotheater_Ollama_Multilingual\voices\Danish_JELyoutubeTalk_PodcastReady.wav"
        #self.exaggeration_2 = 1.0   # how expressive, default is 0.95
        #self.cfg_weight_2 = 0.2     # pacing, default is 0.05
        #self.language_id_2 = "da"




        #this voice is for the hard-coded static end-message (See end of code to change it)
        #self.voice_narrator = r"D:\AI\JEL_LLMradiotheater_Ollama_Multilingual\voices\Danish_JELyoutubeTalk_PodcastReady.wav"
        #self.exaggeration_2 = 0.5   # how expressive, default is 0.5
        #self.cfg_weight_2 = 0.5     # pacing, default is 0.5


        # Set device to CUDA if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for TTS: {self.device}")

        # Initialize Chatterbox TTS
        #REVISION_HASH = "1b475dffa71fb191cb6d5901215eb6f55635a9b6"  # English-only snapshot
        #REVISION_HASH = "beb087b8e081cdd93db3d4420eef79792f7d3a01"  # multi-lingual (Includes Danish) snapshot

        try:
            #self.chat_tts = ChatterboxMultilingualTTS.from_pretrained(device=self.device, revision=REVISION_HASH)
            #self.chat_tts = ChatterboxMultilingualTTS.from_pretrained(device=self.device, revision=REVISION_HASH)
            self.chat_tts = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                #device=self.device
                #revision="1b475dffa71fb191cb6d5901215eb6f55635a9b6" #the English-only version, commit-hash: 1b475df
                #cache_dir=r"C:\Users\jelst\.cache\huggingface\hub\models--ResembleAI--chatterbox\snapshots\1b475dffa71fb191cb6d5901215eb6f55635a9b6",
            #)
            #print(self.chat_tts.model_path)
            print(f"Chatterbox TTS loaded on {self.device}.")
        except Exception as e:
            logging.error(f"Could not load Chatterbox TTS: {e}")
            self.chat_tts = None

        # Initialize Pygame mixer for audio (frequency based on Chatterbox sr ~24kHz, mono)
        #pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
        pygame.mixer.init(frequency=12000, size=-16, channels=1, buffer=1024)
        # Background and beep sounds (WAV files in working directory, same format as TTS)
        try:
            self.background_sound = pygame.mixer.Sound("pictureambience_JEL1.wav")
            self.beep_sound = pygame.mixer.Sound("amb12b.wav")
            self.background_channel = pygame.mixer.Channel(0)  # Dedicated channel for background loop
            self.speech_channel = pygame.mixer.Channel(1)  # Dedicated channel for TTS and beep
        except pygame.error as e:
            logging.error(f"Audio load error: {e}")
            self.append_text(f"\nAudio Error: {e}\n", "error")
            self.background_sound = None
            self.beep_sound = None

        # Alternative random sounds (uncomment if you prefer this over background loop)
        # import glob, random
        # self.random_sounds = [pygame.mixer.Sound(f) for f in glob.glob("random_sound_*.wav")]



        # Create audio_output subfolder
        self.audio_output_dir = "audio_output"
        os.makedirs(self.audio_output_dir, exist_ok=True)

        # Clear contents of the folder
        for filename in os.listdir(self.audio_output_dir):
            file_path = os.path.join(self.audio_output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        


        # Create widgets after all attributes are defined
        self.create_widgets()
        


        #Auto-wrap logic things begin

        self.wrap_up_initiator = 0  # 0 for model_1 which is Green, 1 for model_2 which is Blue
        #this defines which speaker begins the wrap-up process

        self.wrap_progress = 0  # 0 = not started, 1 = initiator wrapped, 2 = other wrapped
        self.wrap_triggered = False  # To detect when wrap-up begins

        self.wrap_state = 0  # 0 = not started, 1 = initiator wraps, 2 = other wraps, 3 = done
        
        self.wrap_turns_completed = 0  # Count how many wrap-up turns have occurred

        self.wrap_phase = 0  # 0 = normal, 1 = initiator wraps, 2 = other wraps, 3 = end

        #Auto-wrap logic things end






    def open_youtube(self):
        import webbrowser
        webbrowser.open_new("https://www.youtube.com/watch?v=n3rlzWN01KA")
        # Replace with your actual link

    
    #def open_youtube():
        #webbrowser.open_new("https://www.youtube.com/watch?v=n3rlzWN01KA")

    def create_widgets(self):
        self.root.configure(bg="#1e1e1e")
        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state='normal', font=("Segoe UI", 14),
            bg="#1e1e1e", fg="#dcdcdc", insertbackground="white"
        )
        self.text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.text_area.tag_configure("GreenLLM", foreground="#90ee90", font=("Segoe UI", 12, "bold"))
        self.text_area.tag_configure("BlueLLM", foreground="#87cefa", font=("Segoe UI", 12, "bold"))
        self.text_area.tag_configure("error", foreground="#ff6b6b", font=("Segoe UI", 14, "bold"))

        frame = tk.Frame(self.root, bg="#1e1e1e")
        frame.pack(fill=tk.X)

        self.start_button = tk.Button(frame, text="Start", command=self.toggle_run, bg="#333333", fg="white")
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(frame, text="Pause", command=self.toggle_pause, bg="#333333", fg="white")
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.topic_button = tk.Button(frame, text="Change Topic", command=self.change_topic, bg="#333333", fg="white")
        self.topic_button.pack(side=tk.LEFT, padx=5)

        self.tts_button = tk.Button(frame, text="TTS: On", command=self.toggle_tts, bg="#333333", fg="white")
        self.tts_button.pack(side=tk.LEFT, padx=5)

        #self.wrap_button = tk.Button(frame, text="Wrap", command=self.trigger_wrap, bg="#333333", fg="white")
        #self.wrap_button.pack(side=tk.LEFT, padx=5)
        
        #self.timer_lock_button = tk.Button(frame, text="Timer Enforced: ON", command=self.toggle_timer_lock, bg="#333333", fg="white")
        #self.timer_lock_button.pack(side=tk.LEFT, padx=5)




    # TTS settings frame (Control sliders for Chatterbox)
        tts_frame = tk.Frame(self.root, bg="#1e1e1e")
        tts_frame.pack(fill=tk.X, pady=5)

        # GreenLLM sliders
        tk.Label(tts_frame, text="GreenLLM TTS", bg="#1e1e1e", fg="#90ee90").pack(side=tk.LEFT, padx=5)
        tk.Label(tts_frame, text="Exag:", bg="#1e1e1e", fg="#cccccc").pack(side=tk.LEFT)
        self.exag_1_scale = tk.Scale(tts_frame, from_=0.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#1e1e1e", fg="#cccccc", command=lambda v: setattr(self, 'exaggeration_1', float(v)))
        self.exag_1_scale.set(self.exaggeration_1)
        self.exag_1_scale.pack(side=tk.LEFT, padx=5)
        tk.Label(tts_frame, text="CFG:", bg="#1e1e1e", fg="#cccccc").pack(side=tk.LEFT)
        self.cfg_1_scale = tk.Scale(tts_frame, from_=0.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#1e1e1e", fg="#cccccc", command=lambda v: setattr(self, 'cfg_weight_1', float(v)))
        self.cfg_1_scale.set(self.cfg_weight_1)
        self.cfg_1_scale.pack(side=tk.LEFT, padx=5)

        # BlueLLM sliders
        tk.Label(tts_frame, text="BlueLLM TTS", bg="#1e1e1e", fg="#87cefa").pack(side=tk.LEFT, padx=5)
        tk.Label(tts_frame, text="Exag:", bg="#1e1e1e", fg="#cccccc").pack(side=tk.LEFT)
        self.exag_2_scale = tk.Scale(tts_frame, from_=0.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#1e1e1e", fg="#cccccc", command=lambda v: setattr(self, 'exaggeration_2', float(v)))
        self.exag_2_scale.set(self.exaggeration_2)
        self.exag_2_scale.pack(side=tk.LEFT, padx=5)
        tk.Label(tts_frame, text="CFG:", bg="#1e1e1e", fg="#cccccc").pack(side=tk.LEFT)
        self.cfg_2_scale = tk.Scale(tts_frame, from_=0.0, to=3.0, resolution=0.05, orient=tk.HORIZONTAL, bg="#1e1e1e", fg="#cccccc", command=lambda v: setattr(self, 'cfg_weight_2', float(v)))
        self.cfg_2_scale.set(self.cfg_weight_2)
        self.cfg_2_scale.pack(side=tk.LEFT, padx=5)

        #self.timer_label = tk.Label(frame, text="⏱ Seconds before talk will wrap up automatically", bg="#1e1e1e", fg="#cccccc", font=("Segoe UI", 12))
        #self.timer_label.pack(side=tk.RIGHT, padx=10)



        # Timer control frame (third row)
        timer_frame = tk.Frame(self.root, bg="#1e1e1e")
        timer_frame.pack(fill=tk.X, pady=5)


        self.wrap_button = tk.Button(timer_frame, text="Wrap", command=self.trigger_wrap, bg="#333333", fg="white")
        self.wrap_button.pack(side=tk.LEFT, padx=5)
        
        self.timer_lock_button = tk.Button(timer_frame, text="Timer Enforced: ON", command=self.toggle_timer_lock, bg="#333333", fg="white")
        self.timer_lock_button.pack(side=tk.LEFT, padx=5)



        initial_mode = "Speech-Time" if self.use_speech_time else "Real-Time"

        self.timer_mode_button = tk.Button(
            timer_frame,
            text=f"Timer Mode: {initial_mode}",
            command=self.toggle_timer_mode,
            bg="#333333",
            fg="white"
        )
        self.timer_mode_button.pack(side=tk.LEFT, padx=5)



        self.timer_label = tk.Label(
            timer_frame,
            text="⏱ Time before talk will wrap up automatically",
            bg="#1e1e1e",
            fg="#cccccc",
            font=("Segoe UI", 12)
        )
        self.timer_label.pack(side=tk.LEFT, padx=10)

        self.timer_label_progressed = tk.Label(
            timer_frame,
            text="⏱ duration",
            bg="#1e1e1e",
            fg="#cccccc",
            font=("Segoe UI", 12)
        )
        self.timer_label_progressed.pack(side=tk.LEFT, padx=10)



        # Footer frame
        footer_frame = tk.Frame(self.root, bg="#1e1e1e")
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Clickable label
        self.youtube_link = tk.Label(
            footer_frame,
            text="🎬 Watch one of my music-videos on YouTube",
            fg="#1e90ff",  # hyperlink-style color
            bg="#1e1e1e",
            cursor="hand2",
            font=("Segoe UI", 10, "underline")
        )
        self.youtube_link.pack()
        self.youtube_link.bind("<Button-1>", lambda e: self.open_youtube())





    def toggle_timer_mode(self):
        self.use_speech_time = not self.use_speech_time
        mode = "Speech-Time" if self.use_speech_time else "Real-Time"
        self.timer_mode_button.config(text=f"Timer Mode: {mode}")
        print(f"Timer mode switched to: {mode}")

    def toggle_run(self):
        self.running = not self.running
        self.start_button.config(text="Stop" if self.running else "Start")
        if self.running:
            self.start_time = time.time()
            self.wrapping_up = False  # Reset wrap state
            self.timer_label.config(text="⏱ time is reset")  # Reset display
            if self.background_sound:
                self.background_channel.play(self.background_sound, loops=-1)  # Start background loop
            threading.Thread(target=self.conversation_thread, daemon=True).start()
            threading.Thread(target=self.monitor_timer, daemon=True).start()
        else:
            self.timer_label.config(text="⏱ time is up")
            if self.background_sound:
                self.background_channel.stop()  # Stop background when conversation stops

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")
        # Optional: Pause background during global pause (uncomment if desired)
        # if self.paused and self.background_sound:
        #     self.background_channel.pause()
        # elif self.background_sound:
        #     self.background_channel.unpause()

    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        self.tts_button.config(text="TTS: On" if self.tts_enabled else "TTS: Off")

    def toggle_timer_lock(self):
            self.timer_lock = not self.timer_lock
            self.timer_lock_button.config(text="Timer Enforced: ON" if self.timer_lock else "Timer Enforced: OFF")

    def change_topic(self):
        topic = simpledialog.askstring("Input", "Enter new topic:")
        if topic:
            self.history_1 = [{"role": "user", "content": topic}]
            self.history_2 = [{"role": "user", "content": topic}]
            self.speaker = 0
            self.append_text(f"\nUser: {topic}\n", tag=None)
            self.log_to_file(f"User: {topic}\n")

    def monitor_timer(self):
        while self.running and not self.wrapping_up:
            # Choose which timer to use
            if self.use_speech_time:
                elapsed = self.totalaccumulatedspeechduration_elapsed_time
            else:
                elapsed = time.time() - self.start_time

            #remaining = max(0, 3600 - elapsed)
            remaining = max(0, self.timer_duration_seconds - elapsed)
            
            #minutes_remaining = int(remaining) // 60 #full minutes
            #seconds_remaining = int(remaining) % 60 #leftover seconds
            minutes_remaining = round(remaining) // 60 #full minutes
            seconds_remaining = round(remaining) % 60 #leftover seconds
            self.timer_label.config(text=f"⏱ remaining {minutes_remaining:02d}:{seconds_remaining:02d}") #this line shows the time formatted in to MM:SS

            #minutes_progressed = int(elapsed) // 60 #full minutes
            #seconds_progressed = int(elapsed) % 60 #leftover seconds
            minutes_progressed = round(elapsed) // 60 #full minutes
            seconds_progressed = round(elapsed) % 60 #leftover seconds
            self.timer_label_progressed.config(text=f"⏱ progressed {minutes_progressed:02d}:{seconds_progressed:02d}") #this line shows the time formatted in to MM:SS


            #self.timer_label.config(text=f"⏱ {int(remaining)}s") #this line shows total seconds only
            #self.timer_label_progressed.config(text=f"⏱ {int(elapsed)}s") #this line shows total seconds only

            if elapsed >= self.timer_duration_seconds and self.timer_lock:
                self.append_text("\n\n[System: Time is up. Triggering graceful wrap...]\n", tag="error")
                self.trigger_wrap()
                break

            time.sleep(1)

    def trigger_wrap(self):
        if not self.running:
            return
        self.wrapping_up = True
        self.append_text("\n\n[System: Wrapping up the conversation...]\n", tag="error")

    def build_prompt(self, system_prompt, history):
        prompt = system_prompt.strip() + "\n\n"
        prompt += "You’ve just received a message from a human user. Respond in a way that feels natural for your role and is engaging, and thoughtful. Don’t just answer—add something new to keep the conversation going. Maintain your role. Do NOT include 'Assistant:', 'Opponent:', or 'User:' in your response.\n\n"

        max_slice = 64  # Start with a larger context window
        min_slice = 3   # Minimum context to retain
        #max_tokens = 1600  # Target token limit (80% of 2048). This is the default used in v2.3.0
        max_tokens = 3200 # Target token limit (80% of 4096). The value of token-to-character ratio is just a rough average guess/approximation (Common rule of thumb is that 1 token is roughly 4 characters)

        # Dynamically adjust the history slice
        slice_size = max_slice
        while slice_size >= min_slice:
            temp_prompt = prompt
            for h in history[-slice_size:]:
                role = h["role"]
                content = h["content"]
                if role == "user":
                    temp_prompt += f"User: {content}\n"
                elif role == "assistant":
                    temp_prompt += f"Assistant: {content}\n"
                elif role == "opponent":
                    temp_prompt += f"Opponent: {content}\n"
            temp_prompt += "Assistant: "
            token_estimate = len(temp_prompt) // 4
            print(f"Testing slice {-slice_size}, token estimate: {token_estimate}") #dynamic context-window debug printer, can be switched off after testing
            if token_estimate <= max_tokens:
                prompt = temp_prompt
                break
            slice_size -= 1

        # If no safe slice found, use minimum
        if token_estimate > max_tokens:
            temp_prompt = prompt
            for h in history[-min_slice:]:
                role = h["role"]
                content = h["content"]
                if role == "user":
                    temp_prompt += f"User: {content}\n"
                elif role == "assistant":
                    temp_prompt += f"Assistant: {content}\n"
                elif role == "opponent":
                    temp_prompt += f"Opponent: {content}\n"
            prompt = temp_prompt + "Assistant: "

        token_estimate = len(prompt) // 4
        if token_estimate > max_tokens:
            self.append_text(f"\n[Warning: Prompt (~{token_estimate} tokens) may exceed context window, risking drift]\n", "error")
        #print(f"Prompt length: {len(prompt)} chars, ~{len(prompt)//4} tokens") # old static context-window debug-print that counts the context-length. If it's more than ~2048 truncated message is likely occurring
        print(f"Prompt length: {len(prompt)} chars, ~{token_estimate} tokens, using slice: {-slice_size if slice_size > min_slice else -min_slice}") # new dynamic context-window debug-print that counts the context-length. If it's more than ~2048 truncated message is likely occurring
        return prompt

    def stream_from_ollama(self, model, prompt_text):
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])  # Retry on server errors, including "too many requests" (429)
        session.mount('http://', HTTPAdapter(max_retries=retries))
        
        payload = {"model": model, "prompt": prompt_text, "stream": True}
        try:
            response = session.post("http://localhost:11434/api/generate", json=payload, stream=True, timeout=60)  # Increased timeout
            response.raise_for_status()
        except requests.RequestException as e:
            self.append_text(f"\nError: {e}\n", "error")
            return ""
        
        buffer = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data:"):
                    line_str = line_str[len("data:"):].strip()
                if line_str in ("[DONE]", "data: [DONE]"):
                    break
                data = json.loads(line_str)
                token = data.get("response", "")
                if token:
                    buffer += token
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        cleaned_buffer = clean_response(buffer)
        return cleaned_buffer

    def append_text(self, text, tag=None):
        self.text_area.configure(state='normal')
        self.text_area.insert(tk.END, text, tag)
        self.text_area.see(tk.END)
        self.text_area.configure(state='disabled')
        self.root.update_idletasks()

    def log_to_file(self, text):
        with open("llm_conversation.txt", "a", encoding="utf-8") as f:
            f.write(text)

    def split_text_into_chunks(self, text, max_chars=300):
        """Split text into chunks of <= max_chars at sentence boundaries."""
        # Split at sentence-ending punctuation, preserving the punctuation
        #sentences = re.split(r'([.!?])', text) #split sentences at . ! ?
        sentences = re.split(r'([.!?:;])', text) #split sentences at . ! ? : ;
        sentences = [s + p for s, p in zip(sentences[::2], sentences[1::2] + ['']) if s.strip()]
        
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                # If a single sentence is too long, split it further
                while len(current_chunk) > max_chars:
                    split_point = current_chunk.rfind(' ', 0, max_chars)
                    if split_point == -1:
                        split_point = max_chars
                    chunks.append(current_chunk[:split_point].strip())
                    current_chunk = current_chunk[split_point:].strip()
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out empty chunks
        return [chunk for chunk in chunks if chunk]
    
    def measure_speech_duration(self, sound):
        self.speech_channel.play(sound)
        start = time.time()
        while self.speech_channel.get_busy():
            pygame.time.wait(100)
        return time.time() - start

    def speak(self, text, voice_path, speaker_label, exaggeration, cfg_weight, language_id):
        """Play a reply using Chatterbox TTS with voice cloning, followed by beep, with background pause/resume, and save to WAV."""
        if not self.tts_enabled or not self.chat_tts:
            logging.warning("TTS is disabled or Chatterbox failed to load")
            return

#        self.totalaccumulatedspeechduration_elapsed_time = 0

        # Validate voice clone file
        if not os.path.exists(voice_path):
            logging.error(f"Voice clone file not found: {voice_path}")
            self.append_text(f"\nChatterbox Error: Voice clone file not found: {voice_path}\n", "error")
            return

        try:
            #tts_text = preprocess_text(text) # Pre-process the text so "I'm" becomes "I am" etc (See top of script)
            #tts_text = text # Use this if no pre-processing is needed (Will speak all written text, including thinking)
            tts_text = re.sub(r'<think>[^<]*?(</think>|$)', '', text, flags=re.DOTALL).strip() # Filter out <think> tags, including unclosed ones
                                                              
            if not tts_text:
                logging.warning("No text to speak after filtering <think> tags")
                return
                                            
            # Split text into chunks
            chunks = self.split_text_into_chunks(tts_text, max_chars=300)
            if not chunks:
                logging.warning("No valid chunks to process")
                return

            #warning, this paragraph is no longer correct with the new preprocessing function, only with the old
            # Preprocess the text to remove asterisks and other symbols
            #tts_text = preprocess_text(tts_text) #only use this if you want to use the preprocess-text function

            # Preprocess each chunk to remove asterisks before TTS. This paragraph is for the new preprocessing function
            processed_chunks = [preprocess_text(chunk).replace('*', '') for chunk in chunks]

            # Get beep audio as NumPy array (for saving)
            beep_raw = self.beep_sound.get_raw()
            fs = self.chat_tts.sr
            beep_pcm = np.frombuffer(beep_raw, dtype=np.int16).reshape(1, -1)

            # Add 1 second of silence (for saving)
            silence_samples = int(fs * 1.0)
            silence = np.zeros((1, silence_samples), dtype=np.int16)

            # Collect all chunk audio for saving
            all_pcm = []

            # Pause background during speech and beep
            self.background_channel.pause()
                                                                                    
            # Process each chunk
            for chunk in processed_chunks:
                # Pre-process the chunk (optional)
                # tts_chunk = preprocess_text(chunk)  # Use if needed
                tts_chunk = chunk

                # Generate audio (returns torch.Tensor)
                wav_tensor = self.chat_tts.generate(
                    text=tts_chunk,
                    language_id=language_id,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    audio_prompt_path=voice_path
                )

                # Convert to NumPy array
                audio_numpy = wav_tensor.to("cpu").numpy()
                # Get channels (assume mono; adjust if stereo)
                channels = 1 if audio_numpy.ndim == 1 else audio_numpy.shape[0]

                # Ensure 2D array
                if audio_numpy.ndim == 1:
                    audio_numpy = audio_numpy.reshape(1, -1)
                                                 
                # Convert to 16-bit PCM
                pcm = np.clip(audio_numpy * 32767, -32768, 32767).astype(np.int16)
                all_pcm.append(pcm)

                # Play chunk (non-blocking, but wait until done)
                tts_sound = pygame.mixer.Sound(buffer=pcm.tobytes())
                #self.speech_channel.play(tts_sound) #this is now called in the timer function
                while self.speech_channel.get_busy():
                    pygame.time.wait(100)
                accumulatedspeechduration = self.measure_speech_duration(tts_sound)
                self.totalaccumulatedspeechduration_elapsed_time += accumulatedspeechduration
                print(f"Chunk duration: {accumulatedspeechduration:.2f} seconds")
                print(f"Total duration: {self.totalaccumulatedspeechduration_elapsed_time:.2f} seconds")


            # Play beep sound after all chunks (blocking wait)
            self.speech_channel.play(self.beep_sound)
            while self.speech_channel.get_busy():
                pygame.time.wait(100)

            # Resume background
            self.background_channel.unpause()

            # Save combined audio: all chunks + silence (float32 instead of int16)
            if all_pcm:
                combined_float = np.concatenate(all_pcm + [silence.astype(np.float32)], axis=1).astype(np.float32)

                # Apply -100 dB gain reduction
                scale = 10 ** (-96 / 20)  # ~1e-5, reduce gain, -96 dB
                combined_float *= scale

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                filename = os.path.join(self.audio_output_dir, f"{timestamp}_{speaker_label}.wav")
                try:
                    # Write as 32-bit float WAV
                    sf.write(filename, combined_float.T, fs, subtype='FLOAT')
                    print(f"Saved audio to {filename}")
                except Exception as e:
                    logging.error(f"Failed to save audio to {filename}: {e}")
                    self.append_text(f"\nAudio Save Error: {e}\n", "error")

        except (ValueError, RuntimeError, AttributeError, FileNotFoundError, pygame.error) as exc:
            logging.error(f"Chatterbox/Audio Error: {exc}")
            self.append_text(f"\nChatterbox/Audio Error: {exc}\n", "error")

    def conversation_thread(self):
        wrap_prompt = """
            Please summarize and say goodbye in a natural and respectful way and please remember to tell the listeners that this was just a fictional radio-play by JELSTUDIO meant to serve as pleasurable entertainment (The people from Mars are NOT invading!)
        """

        while self.running:
            if self.paused:
                time.sleep(0.5)
                continue

            model = self.model_1 if self.speaker == 0 else self.model_2
            sys_prompt = self.system_prompt_1 if self.speaker == 0 else self.system_prompt_2
            history = self.history_1 if self.speaker == 0 else self.history_2
            label = "GreenLLM" if self.speaker == 0 else "BlueLLM"
            voice_path = self.voice_female if self.speaker == 0 else self.voice_male
            exaggeration = self.exaggeration_1 if self.speaker == 0 else self.exaggeration_2
            cfg_weight = self.cfg_weight_1 if self.speaker == 0 else self.cfg_weight_2
            language_id = self.language_id_1 if self.speaker == 0 else self.language_id_2

            self.append_text(f"\n\n{label}: ", tag=label)

            # 🧠 Decide which prompt to use
            if self.wrapping_up:
                if self.wrap_phase == 0:
                    # Let current speaker finish normally
                    prompt_text = self.build_prompt(sys_prompt, history)
                    if self.speaker == self.wrap_up_initiator:
                        self.wrap_phase = 1
                    else:
                        self.wrap_phase = -1  # Wait for initiator's next turn

                elif self.wrap_phase == -1 and self.speaker == self.wrap_up_initiator:
                    prompt_text = self.build_prompt(wrap_prompt, history)
                    self.wrap_phase = 1

                elif self.wrap_phase == 1 and self.speaker == self.wrap_up_initiator:
                    prompt_text = self.build_prompt(wrap_prompt, history)

                elif self.wrap_phase == 1 and self.speaker != self.wrap_up_initiator:
                    prompt_text = self.build_prompt(wrap_prompt, history)
                    self.wrap_phase = 2

                elif self.wrap_phase == 2 and self.speaker != self.wrap_up_initiator:
                    prompt_text = self.build_prompt(wrap_prompt, history)

                elif self.wrap_phase == 2 and self.speaker == self.wrap_up_initiator:
                    final_outro = (
                        """
                        All text was unscripted and procedurally generated on the fly by artificial intelligence models. All voices were spoken by the Chatterbox text to speech system. The end.
                        """
                    )
                    #self.speak(final_outro, self.voice_narrator, "System", 0.5, 0.5)
                    self.append_text(f"\n\n[System Outro]: {final_outro}\n", tag="system")

                    # End conversation after both wrap-ups
                    self.wrap_phase = 3
                    self.running = False
                    self.start_button.config(text="Start")
                    self.append_text("\n\n[System: Conversation has ended.]\n", tag="error")
                    if self.background_sound:
                        self.background_channel.stop()
                    continue  # Skip this turn

            else:
                # Normal conversation
                prompt_text = self.build_prompt(sys_prompt, history)

            try:
                reply = self.stream_from_ollama(model, prompt_text)
                if reply:
                    self.append_text(reply, tag=label)
                    self.log_to_file(f"{label}: {reply}\n\n")
                    self.history_1.append({"role": "assistant" if self.speaker == 0 else "user", "content": reply})
                    self.history_2.append({"role": "assistant" if self.speaker == 1 else "user", "content": reply})
                    print(f"Calling speak for {label}: exag={exaggeration}, cfg={cfg_weight}, speaker={self.speaker}")
                    self.speak(reply, voice_path, label, exaggeration, cfg_weight, language_id)
                    time.sleep(2)
                    print(f"Wrap phase: {self.wrap_phase}, Speaker: {self.speaker}")  # Debug line
                else:
                    print(f"No reply from {label}, forcing speaker switch. Current speaker: {self.speaker}")
                    time.sleep(1)

            except Exception as e:
                self.append_text(f"\nError: {e}\n", "error")
                print(f"Error in {label} turn, speaker={self.speaker}, switching speakers")
                time.sleep(1)

            self.speaker = 1 - self.speaker
            time.sleep(0.7)

if __name__ == "__main__":
    root = tk.Tk()
    app = LLMDuetApp(root)
    root.mainloop()