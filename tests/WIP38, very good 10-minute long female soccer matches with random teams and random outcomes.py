import tkinter as tk
from tkinter import scrolledtext, simpledialog
import threading
import requests
import time
import json
import os
import simpleaudio as sa
import torch
import torch.serialization
import re

# Allowlist safe globals for VITS model loading
torch.serialization.add_safe_globals([
    'TTS.tts.configs.vits_config.VitsConfig',
    'TTS.utils.audio.audio_processor.AudioProcessor',
    'TTS.tts.models.vits.Vits'
])

# Attempt to import Coqui TTS
try:
    from TTS.api import TTS as CoquiTTS
except ModuleNotFoundError as e:
    print(f"Error: Could not import TTS module. Ensure 'coqui-tts==0.26.2' is installed.\n{e}")
    print("Run: pip install coqui-tts==0.26.2")
    exit(1)

# Use CPU explicitly
device = "cpu"
print(f"Using device for TTS: {device}")

# Load Coqui TTS with VITS multi-speaker English model on CPU
try:
    coqui_tts = CoquiTTS(model_name="tts_models/en/vctk/vits").to(device)
    # Print available speaker IDs
    print("Available VITS speaker IDs:", coqui_tts.speakers)
except Exception as e:
    print(f"Error loading TTS model: {e}")
    exit(1)

# Define speaker IDs from the model
SPEAKER_1 = "p225"  # Female
SPEAKER_2 = "p231"  # Male

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_1 = "gemma3:12b"
MODEL_2 = "gemma3:12b"
SYSTEM_PROMPT_1 = "You are commenting on a female soccer-match. Keep it brief and to the point. Always end your talk by updating the previous score (Only if it has changed, which is something you decide based on how the match is progressing) to a new one so the match advances, and remember to mention which team has which score. Strictly adhere to real soccer-rules. Scores do not always change during your speak, in which case you just repeat the last known score. The first time you talk the match is about to begin and the score is therefore zero to both teams."
SYSTEM_PROMPT_2 = "You are commenting on a female soccer-match. Keep it brief and to the point. Always end your talk by updating the previous score (Only if it has changed, which is something you decide based on how the match is progressing) to a new one so the match advances, and remember to mention which team has which score. Strictly adhere to real soccer-rules. Scores do not always change during your speak, in which case you just repeat the last known score. The first time you talk the match is about to begin and the score is therefore zero to both teams."
INITIAL_TOPIC = "Start by choosing the 2 random female teams we will see in this match and introduce them and what we might expect from them."
SAVE_FILE = "llm_conversation.txt"

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

        self.model_1 = MODEL_1
        self.model_2 = MODEL_2
        self.system_prompt_1 = SYSTEM_PROMPT_1
        self.system_prompt_2 = SYSTEM_PROMPT_2
        self.tts_enabled = True
        self.running = False
        self.paused = False
        self.history_1 = [{"role": "user", "content": INITIAL_TOPIC}]
        self.history_2 = [{"role": "user", "content": INITIAL_TOPIC}]
        self.speaker = 0
        self.wrapping_up = False
        self.wrap_stage = 0  # 0 = not started, 1 = speaker 0 finished, 2 = both finished
        self.start_time = None


    def create_widgets(self):
        self.root.configure(bg="#1e1e1e")
        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state='normal', font=("Segoe UI", 14),
            bg="#1e1e1e", fg="#dcdcdc", insertbackground="white"
        )
        self.text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.text_area.tag_configure("Woman", foreground="#90ee90", font=("Segoe UI", 14, "bold"))
        self.text_area.tag_configure("Man", foreground="#87cefa", font=("Segoe UI", 14, "bold"))
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

        self.timer_label = tk.Label(frame, text="⏱ 600s", bg="#1e1e1e", fg="#cccccc", font=("Segoe UI", 12))
        self.timer_label.pack(side=tk.RIGHT, padx=10)


    def toggle_run(self):
        self.running = not self.running
        self.start_button.config(text="Stop" if self.running else "Start")
        if self.running:
            self.start_time = time.time()
            self.timer_label.config(text="⏱ 600s")  # Reset display
            threading.Thread(target=self.conversation_thread, daemon=True).start()
            threading.Thread(target=self.monitor_timer, daemon=True).start()
        else:
            self.timer_label.config(text="⏱ --")

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
            elapsed = int(time.time() - self.start_time)
            remaining = max(0, 600 - elapsed)
            self.timer_label.config(text=f"⏱ {remaining}s")
            if elapsed >= 600:
                self.append_text("\n\n[System: 600 seconds elapsed. Triggering graceful wrap...]\n", tag="error")
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
        prompt += " Based on the new information provide a clear and thoughtful, but not overly elaborate, response that advances the conversation without repeating 'Assistant:'.\n\n"
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
            response = requests.post(OLLAMA_API, json=payload, stream=True, timeout=30)
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
        with open(SAVE_FILE, "a", encoding="utf-8") as f:
            f.write(text)

    def speak(self, text, speaker_id):
        if self.tts_enabled:
            try:
                tts_text = preprocess_text(text)
                coqui_tts.tts_to_file(
                    text=tts_text,
                    speaker=speaker_id,
                    file_path="output.wav"
                )
                wave_obj = sa.WaveObject.from_wave_file("output.wav")
                play_obj = wave_obj.play()
                play_obj.wait_done()
                os.remove("output.wav")
            except Exception as e:
                self.append_text(f"\nTTS Error: {e}\n", "error")

    def conversation_thread(self):
        while self.running:
            if self.paused:
                time.sleep(0.2)
                continue

            model = self.model_1 if self.speaker == 0 else self.model_2
            sys_prompt = self.system_prompt_1 if self.speaker == 0 else self.system_prompt_2
            history = self.history_1 if self.speaker == 0 else self.history_2
            label = "Woman" if self.speaker == 0 else "Man"
            speaker_id = SPEAKER_1 if self.speaker == 0 else SPEAKER_2

            self.append_text(f"\n\n{label}: ", tag=label)
            # If wrapping, override prompt
            if self.wrapping_up:
                wrap_prompt = "Please summarize the sporting-event we just saw and say goodbye to the viewers in a natural and respectful way. Remember to mention who was the final winner of the match."
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
                    self.speak(reply, speaker_id)
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