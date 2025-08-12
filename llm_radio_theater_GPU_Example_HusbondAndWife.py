#version2.0.0
import tkinter as tk
from tkinter import scrolledtext, simpledialog
import threading
import requests
import time
import json
import os
import simpleaudio as sa
import numpy as np
import io
import wave
import re
import logging
import torch  # Needed for tensor handling

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Attempt to import Chatterbox TTS
try:
    from chatterbox.tts import ChatterboxTTS
except ModuleNotFoundError as e:
    print(f"Error: Could not import Chatterbox TTS module. Ensure 'chatterbox-tts' is installed.\n{e}")
    exit(1)

# Contraction expansion dictionary
CONTRACTIONS = {
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "I'm": "I am",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "I'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "they'll": "they will",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
}

def preprocess_text(text):
    """Preprocess text for TTS: expand contractions and remove special symbols."""
    text_lower = text.lower()
    for contraction, expanded in CONTRACTIONS.items():
        text_lower = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text_lower, flags=re.IGNORECASE)
    text_clean = re.sub(r'\*([^\*]+)\*', r'\1', text_lower)
    text_clean = re.sub(r'[^\w\s,.!?]', '', text_clean)
    return text_clean

def clean_response(text):
    """Remove unwanted prefixes like 'Assistant:' from model responses."""
    text = re.sub(r'^\s*Assistant:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

class LLMDuetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Radio Theater")
        self.root.geometry("800x800")
        self.create_widgets()

        self.model_1 = "gemma3:12b"
        self.model_2 = "gemma3:12b"
        self.system_prompt_1 = (
            "You are Jane, the current wife of John. "
            "You are speaking as yourself, reflecting on your career, your role as woman and wife, and your personal experiences. "
            "You are real and current. This is not a role-playing game. Speak truthfully and naturally, as Jane would."
            "You are in a discussion with your husband about whether you should get a dog or a cat. You prefer cats."
            "Never use more than 900 characters for your responses"
        )
        self.system_prompt_2 = (
            "You are John, the current husband of Jane. "
            "You are in a discussion with your wife about whether you should get a cat or a dog. You prefer dogs."
            "Never use more than 900 characters for your responses"
        )
        self.tts_enabled = True
        self.running = False
        self.paused = False
        self.history_1 = [{"role": "user", "content": "Start by telling me your name and who you are."}]
        self.history_2 = [{"role": "user", "content": "Start by telling me your name and who you are."}]
        self.speaker = 0
        self.wrapping_up = False
        self.wrap_stage = 0
        self.start_time = None

        # Set device to CUDA if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for TTS: {self.device}")

        # Initialize Chatterbox TTS
        try:
            self.chat_tts = ChatterboxTTS.from_pretrained(device=self.device)
            print(f"Chatterbox TTS loaded on {self.device}.")
        except Exception as e:
            logging.error(f"Could not load Chatterbox TTS: {e}")
            self.chat_tts = None

        # Voice clone files (ensure these files exist in the working directory)
        self.voice_female = "voice_female.wav"
        self.voice_male = "voice_male.wav"

    def create_widgets(self):
        self.root.configure(bg="#1e1e1e")
        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state='normal', font=("Segoe UI", 14),
            bg="#1e1e1e", fg="#dcdcdc", insertbackground="white"
        )
        self.text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.text_area.tag_configure("Jane", foreground="#90ee90", font=("Segoe UI", 14, "bold"))
        self.text_area.tag_configure("John", foreground="#87cefa", font=("Segoe UI", 14, "bold"))
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

        self.wrap_button = tk.Button(frame, text="Wrap", command=self.trigger_wrap, bg="#333333", fg="white")
        self.wrap_button.pack(side=tk.LEFT, padx=5)

        self.timer_label = tk.Label(frame, text="⏱ Seconds before talk will wrap up automatically", bg="#1e1e1e", fg="#cccccc", font=("Segoe UI", 12))
        self.timer_label.pack(side=tk.RIGHT, padx=10)

    def toggle_run(self):
        self.running = not self.running
        self.start_button.config(text="Stop" if self.running else "Start")
        if self.running:
            self.start_time = time.time()
            self.wrapping_up = False  # ✅ Reset wrap state

            self.timer_label.config(text="⏱ time is reset")  # Reset display
            threading.Thread(target=self.conversation_thread, daemon=True).start()
            threading.Thread(target=self.monitor_timer, daemon=True).start()
        else:
            self.timer_label.config(text="⏱ time is up")

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        self.tts_button.config(text="TTS: On" if self.tts_enabled else "TTS: Off")

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
            elapsed = time.time() - self.start_time
            remaining = max(0, 3600 - elapsed)
            self.timer_label.config(text=f"⏱ {int(remaining)}s")
            if elapsed >= 3600:
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
        prompt += "Based on the new information provide a clear and thoughtful, but not overly elaborate, response that advances the conversation without repeating 'Assistant:'.\n\n"
        for h in history[-5:]:
            role = h["role"]
            content = h["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "opponent":
                prompt += f"Opponent: {content}\n"
        prompt += "Assistant: "
        return prompt

    def stream_from_ollama(self, model, prompt_text):
        payload = {"model": model, "prompt": prompt_text, "stream": True}
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True, timeout=30)
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

    def speak(self, text, voice_path):
        """Play a reply using Chatterbox TTS with voice cloning."""
        if not self.tts_enabled or not self.chat_tts:
            logging.warning("TTS is disabled or Chatterbox failed to load")
            return  # TTS turned off or failed to load

        # Validate voice clone file
        if not os.path.exists(voice_path):
            logging.error(f"Voice clone file not found: {voice_path}")
            self.append_text(f"\nChatterbox Error: Voice clone file not found: {voice_path}\n", "error")
            return

        try:
            # Pre-process the text
            # tts_text = preprocess_text(text) #use this if you need to pre-process text
            tts_text = text #use this if you DON'T need to pre-process text

            # Generate audio (returns torch.Tensor)
            wav_tensor = self.chat_tts.generate(
                text=tts_text,
                exaggeration=0.5, # Chatterbox-default=0.5. Controls how expressive speech is
                cfg_weight=0.5, # Chatterbox-default=0.5. Controls pacing/speed adjustments
                audio_prompt_path=voice_path
            )

            # Convert to NumPy array
            audio_numpy = wav_tensor.to("cpu").numpy()

            # Get sampling rate and channels
            fs = self.chat_tts.sr
            channels = 1 if audio_numpy.ndim == 1 else audio_numpy.shape[0]  # (channels, samples) or (samples,)

            # Ensure it's 2D for consistency (add channel dim if mono)
            if audio_numpy.ndim == 1:
                audio_numpy = audio_numpy.reshape(1, -1)

            # Convert to 16-bit PCM (assuming float32 normalized to [-1, 1])
            pcm = np.clip(audio_numpy * 32767, -32768, 32767).astype(np.int16)

            # Create WaveObject with raw PCM data
            wave_obj = sa.WaveObject(
                pcm.tobytes(),  # Raw PCM bytes
                num_channels=channels,
                bytes_per_sample=2,  # 16-bit PCM
                sample_rate=fs
            )

            # Play audio (blocking to ensure sequencing)
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Blocks until audio finishes

        except (ValueError, RuntimeError, AttributeError, FileNotFoundError) as exc:
            logging.error(f"Chatterbox Error: {exc}")
            self.append_text(f"\nChatterbox Error: {exc}\n", "error")

    def conversation_thread(self):
        while self.running:
            if self.paused:
                time.sleep(0.2)
                continue

            model = self.model_1 if self.speaker == 0 else self.model_2
            sys_prompt = self.system_prompt_1 if self.speaker == 0 else self.system_prompt_2
            history = self.history_1 if self.speaker == 0 else self.history_2
            label = "Jane" if self.speaker == 0 else "John"
            voice_path = self.voice_female if self.speaker == 0 else self.voice_male

            self.append_text(f"\n\n{label}: ", tag=label)
            if self.wrapping_up:
                wrap_prompt = "Please summarize and say goodbye in a natural and respectful way."
                prompt_text = self.build_prompt(wrap_prompt, history)
            else:
                prompt_text = self.build_prompt(sys_prompt, history)

            try:
                reply = self.stream_from_ollama(model, prompt_text)
                if reply:
                    self.append_text(reply, tag=label)
                    self.log_to_file(f"{label}: {reply}\n\n")
                    self.history_1.append({"role": "assistant" if self.speaker == 0 else "opponent", "content": reply})
                    self.history_2.append({"role": "assistant" if self.speaker == 1 else "opponent", "content": reply})
                    self.speak(reply, voice_path)
                    if self.wrapping_up:
                        if self.wrap_stage == 0 and self.speaker == 0:
                            self.wrap_stage = 1
                        elif self.wrap_stage == 1 and self.speaker == 1:
                            self.wrap_stage = 2
                            self.running = False
                            self.start_button.config(text="Start")
                            self.append_text("\n\n[System: Conversation has ended.]\n", tag="error")
            except Exception as e:
                self.append_text(f"\nError: {e}\n", "error")

            self.speaker = 1 - self.speaker
            time.sleep(0.5)

if __name__ == "__main__":
    root = tk.Tk()
    app = LLMDuetApp(root)
    root.mainloop()