# JEL LLM Radio Theater

JEL LLM Radio Theater is a Python application that creates a dynamic, spoken dialogue between two AI characters using the Ollama language model (`gemma3:4b`) and either Coqui TTS (`tts_models/en/vctk/vits`) or Chatterbox TTS with voice-cloning. The characters engage in a conversational debate, with one embodying a female perspective (And female voice) and the other a male perspective (And male voice). If you use Chatterbox you can use wave-files to have two of your own voices used. The dialogue is displayed in a Tkinter GUI, spoken aloud via text-to-speech, and saved to a text file (`llm_conversation.txt`).

## New in v2.2.0 (The lastest version)
- Chatterbox is now fully articulated by the script, making for more expressive speech.
- Plays background-music while waiting for the LLM to generate text (If you don't want that, just save a short silent wav-file with this name: `pictureambience_JEL1.wav`. The current song used is made by JEL and is inspired by one composed by Guy Whitmore for the 1995 PC-game named: `Shivers`.) and plays a low tone at every LLM turn-switch.
- Chatterbox' speech-output is saved as time- and name-stamped wav-files in the sub-folder `audio_output` (One file for each turn, to avoid the silence while waiting for text to be generated. Join them in an audio-editor if you want to re-create the full spoken conversation).

## Features
- Two AI characters with distinct personalities, powered by `gemma3:4b` via Ollama.
- Text-to-speech using Coqui TTS with VITS multi-speaker model (~110 voices).
- Text-to-speech alternative using Chatterbox TTS with voice-cloning via two wave-files (This way you can have any voice you want).
- Tkinter GUI for real-time dialogue display.
- Saves conversation to `llm_conversation.txt`.
- For Coqui TTS; Handles contractions (e.g., “we’re” → “we are”) for accurate pronunciation (This is switched OFF by default when using Chatterbox, but can be enabled if needed/wanted by editing the .py file).
- For Coqui TTS; CPU-only execution for broad compatibility.
- For Chatterbox TTS; Defaults to using GPU but will use CPU if GPU is unavailable.
- Customizable system prompts, topics, and speaker voices (Primarily by editing the .py file, but topics can be modified during runtime via the GUI)

## Prerequisites
Before installing, ensure you have the following software installed on your Windows system:

1. **Python 3.11.9**:
   - Download from [python.org](https://www.python.org/downloads/release/python-3119/).
   - During installation, check "Add Python 3.11 to PATH" and install for all users.
   - Verify with: `python --version` (should output `Python 3.11.9`).
   - it MUST be Python 3.11 (Python 3.13 is not compatible, nor is 3.10)

2. **Ollama**:
   - Download and install from [ollama.com](https://ollama.com/download).
   - Install the `gemma3:4b` model by running: `ollama pull gemma3:4b`.
   - Ensure Ollama is running (`ollama serve`) and accessible at `http://localhost:11434`.
   - You can use a different model if you wish, but the default script is set to use the `gemma3:4b` model.
   - And use a separate CMD console window, which doesn't need to have a VENV active, to run the `ollama serve` command (As it doesn't allow further input to that CMD window when running. So run the main Python script in its own VENV activated CMD console window)

3. **Git** (optional, for cloning the repository):
   - Download from [git-scm.com](https://git-scm.com/download/win).
   - Install with default settings.

4. **System Requirements**:
   - Windows 10 or 11 (tested on Windows 11).
   - 16GB+ RAM recommended (Coqui TTS and LLM require ~4–8GB. Unsure about minimum RAM when using Chatterbox, but definitely works on 16GB VRAM).
   - ~10GB free disk space for Python, dependencies, and models (Maybe more if using Chatterbox)

## Installation
Follow these steps in order to set up the project:

1. **Clone the Repository** (or download as a ZIP):
   ```bash
   git clone https://github.com/JELSTUDIO/JEL_LLMradiotheater_Ollama.git
   cd JEL_LLMradiotheater_Ollama
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   ```bash
   venv\Scripts\activate
   ```

4. **Install Dependencies**:
   - The repository includes `requirements_CPUonly.txt`, which you also need to install even if you only want to use the new GPU/Chatterbox-version. Install with:
     ```bash
     pip install -r requirements_CPUonly.txt
     ```
   - Contents of `requirements_CPUonly.txt` (You can delete the 'coqui-tts' line if you only want to use Chatterbox):
     ```
     coqui-tts==0.26.2
     torch==2.7.1
     numpy==2.2.6
     gruut==2.4.0
     coqpit-config==0.2.0
     networkx==3.5
     simpleaudio==1.0.4
     requests==2.32.4
     ```
   - Then install Chatterbox (Keep the VENV, that you installed the requirements into, active, so Chatterbox is installed into it) with:
     ```bash
     pip3 install chatterbox-tts
     ```
   - To use version 2.2.0 (Or the specific NASA version, which is version 2.1.0) you must also install this new sound-engine (Still with the VENV active). This switches the audio playback system from simpleaudio to pygame, which is required for the new sound-effects to work. If you ONLY want to use version 2.0.0, which is the Coqui-TTS version (Without Chatterbox support), then you can skip this step:
     ```bash
     pip3 install pygame
     ```

4.1 **Install Dependencies for RXT50 series GPU support** (Skip this 4.1 section if you only want to use CPU):
   - First uninstall CPU-based torch. Uninstall with:
     ```bash
     pip3 uninstall torch torchvision torchaudio
     ```
   - Then install GPU-based torch. For CUDA 12.8 install with:
     ```bash
     pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
     ```

5. **Verify Ollama Setup**:
   - Start Ollama in a separate (Not in the VENV activated terminal, but its own terminal) terminal:
     ```bash
     ollama serve
     ```
   - Confirm the `gemma3:4b` model (The new version 2.2.0 GPU-based script uses `gemma3:12b` and `llama3.1:8b`, but you can change model in the .py file if you don't have those 2 installed. You can actually use any model you have installed, just make sure you enter its correct name in the script) is installed (Depending on how your Ollama is installed, you may need to close the CMD console window you ran ollama serve in, to run Ollama in the system-tray instead. Then after installing a new model to Ollama close the system-tray version of Ollama again and open a CMD console window and run the Ollama Serve command. You don't need a VENV for the Ollama Serve command. Sorry for this complication, but I'm new to Python and had help from AI-LLMs (Microsoft CoPilot and xAI's Grok) to complete this project, so it may be a bit messy):
     ```bash
     ollama list
     ```
   - If not installed, run:
     ```bash
     ollama pull gemma3:4b
     ollama pull gemma3:12b
     ollama pull llama3.1:8b
     ```

6. **Download Coqui TTS Model**:
   - The script automatically downloads `tts_models/en/vctk/vits` to `C:\Users\<YourUsername>\.cache\tts` on first run (Keep an eye on the CMD console window as you may have to press [Y] to accept the license-term for the TTS on the very first run). Ensure ~2GB free space.
   - If you haven't already got Chatterbox, then it will download its needed models to `C:\Users\<YourUsername>\.cache\huggingface\hub\models--ResembleAI--chatterbox` on first run.

## Running the Application
1. **Activate the Virtual Environment** (if not already active):
   ```bash
   cd JEL_LLMradiotheater_Ollama
   venv\Scripts\activate
   ```

2. **Start Ollama** (if not running. And, again, this must be done in a different CMD console window and not the one with the VENV activated. To be clear; this is NOT the Ollama-program that opens its own Ollama chat-box GUI and places a lama-icon in the windows task-bar or notification-area! If you have that open you must close it, or else the Ollama server won't be available to serve this conversation script):
   ```bash
   ollama serve
   ```

3. **Run the Script** (This is done in the VENV activated CMD console window. And be patient as it may take some time to load the LLM and, when using Chatterbox, generate the speeches. On first run you should check the CMD console window to see if you need to accept the TTS license):
   - This is for the CPU-based Coqui TTS, version 2.0.0, run:
     ```bash
     python llm_radio_theater_CPUonly_Example_HusbondAndWife.py
     ```
   - For the version 2.1.0 GPU-based Chatterbox TTS, run:
     ```bash
     python llm_radio_theater_GPU_Example_HusbondAndWife.py
     ```
   - For the version 2.2.0 GPU-based Chatterbox TTS, run:
     ```bash
     python llm_radio_theater_v2.2.0_Chatterbox_GPUandCPU_Example_HusbondAndWife.py
     ```

4. **Interact with the GUI**:
   - **Start**: Begins the dialogue between the Woman (pro-living room) and Man (pro-garage).
   - **Pause/Resume**: Pauses or resumes the conversation.
   - **Change Topic**: Enter a new topic for the characters to discuss.
   - **TTS: On/Off**: Toggles text-to-speech.
   - **WRAP**: Click this when you want the conversation to begin to wrap up with a natural ending.
   - **Timer Enforce**: If ON, the script will auto-wrap when the countdown reaches zero. If OFF the script will continue indefinitely (So if a conversation heads in an interesting direction and you want it to continue, click this to OFF to ignore the countdown timer)
   - **TIMER**: This will count down in seconds to an automatic wrap-up of the conversation so it doesn't continue forever (In the v2.2.0 script it defaults to 3600 seconds, but you can modify this in the .py file)
   - **Chatterbox sliders**: These will adjust how the 2 Chatterbox-voices speaks (How dramatic they sound, and how quick the talk) Adjust them to taste (And if you want you can update the defaults inside the script to your preferred values, as different voices may need different values to sound normal. Chatterbox' internal default values are 0.5 for both sliders, which is the setting that will be the safest against mis-spoken words or 'hallucination-talk')
   - Output appears in the GUI, is spoken aloud, and is saved to `llm_conversation.txt` (In version 2.2.0 the text-file is cleared/emptied when you first run the script so only the latest conversation-transcript is stored. So if you want to keep older conversation-transcripts, make sure to copy this file or its contents before running the script).

**Expected Output**:
- Console: Lists `Using device for TTS: cpu` and ~110 VITS speaker IDs (e.g., `p225`, `p231`).
- GUI: Displays dialogue with proper spacing (e.g., “Darling, it’s utterly absurd…”).
- Coqui-TTS: Pronounces (Not always, for some reason) contractions correctly (e.g., “it’s” as “it is”), ~10–20 seconds per sentence.
- Text File: Saves dialogue to `llm_conversation.txt` with correct formatting.
- In version 2.2.0 the console-output is much more verbose (Prints a lot of debug-info) than earlier versions. All that can be ignored (Including some warning-messages that pop-up talking about soon-to-be-deprecated libraries, as all those are outside of my control since they're not part of my script but some of the dependencies), at least at the time of writing (20250816) where the script works fine on Windows11.

## Customization
- **Change Voices**:
  - For Coqui TTS; Edit `llm_radio_theater_CPUonly_Example_HusbondAndWife.py` to use different VITS speaker IDs (e.g., `p226` for female, `p240` for male):
    ```python
    SPEAKER_1 = "p226"  # Female
    SPEAKER_2 = "p240"  # Male
    ```
  - Test voices:
    ```bash
    python -c "from TTS.api import TTS; tts = TTS(model_name='tts_models/en/vctk/vits').to('cpu'); tts.tts_to_file(text='This is a test.', speaker='p226', file_path='test_p226.wav')"
    ```
  - For Chatterbox TTS; Change file-names in the script so they match the name of the wave-files with the voice-snippets you want to clone, or replace the default wav-files with your own voice-recordings:
    ```python
    self.voice_female = "voice_female.wav"
    self.voice_male = "voice_male.wav"
    ```

- **Change Topic or Prompts**:
  - Modify `INITIAL_TOPIC` in the script, e.g.:
    ```python
    INITIAL_TOPIC = "What do you want to talk about today?"
    ```
  - Update system prompts for different perspectives, e.g.:
    ```python
    SYSTEM_PROMPT_1 = "You are the wife in the relationship."
    SYSTEM_PROMPT_2 = "You are the husband in the relationship."
    ```
  - In the new GPU-script it's moved to `self.history` 1 (Female, first speaker) and 2 (Male, second speaker), e.g.:
    ```python
    self.history_1 = [{"role": "user", "content": "Start by telling me your name and who you are."}]
    ```
  - In the new GPU-script it's moved to `self.system_prompt` 1 (Female, first speaker) and 2 (Male, second speaker), e.g.:
    ```python
    self.system_prompt_1 = ("You are Jane, the current wife of John. ")
    ```

## Troubleshooting
- **GUI Spacing Issues**:
  - Ensure you’re using the latest `llm_radio_theater_CPUonly_Example_HusbondAndWife.py`.
  - Verify Python 3.11.9: `python --version`.
- **TTS Errors**:
  - Test audio playback:
    ```bash
    python -c "import simpleaudio; wave_obj = simpleaudio.WaveObject.from_wave_file('output.wav'); play_obj = wave_obj.play(); play_obj.wait_done()"
    ```
  - Check `coqui-tts==0.26.2` and `torch==2.7.1`: `pip list`.
  - Make sure pygame is installed (See higher up in this text)
- **Ollama Errors**:
  - Ensure Ollama is running: `curl http://localhost:11434/api/tags`.
  - Reinstall `gemma3:4b`: `ollama pull gemma3:4b`.
- **Performance**:
  - Expect ~10–20 seconds per spoken sentence on CPU.
  - Ensure 16GB+ RAM to avoid slowdowns.
- **Logs**:
  - Check `llm_conversation.txt` for dialogue history.
  - Share console output, GUI text, or errors via GitHub Issues if problems occur (Hopefully somebody smarter than me can then help. Or you can show the problem to an LLM and see if it can guide you further)

## Contributing
Feel free to fork the repository or make suggestions for improvements. Report issues via [GitHub Issues](https://github.com/JELSTUDIO/JEL_LLMradiotheater_Ollama/issues).

## License
MIT License. See [LICENSE](LICENSE) for details.
(Be aware the NASA-version has its own license. See the NASA folder for that)
