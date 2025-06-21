# JEL LLM Radio Theater

JEL LLM Radio Theater is a Python application that creates a dynamic, spoken dialogue between two AI characters using the Ollama language model (`gemma3:4b`) and Coqui TTS (`tts_models/en/vctk/vits`). The characters engage in a conversational debate, with one embodying a female perspective (And female voice) and the other a male perspective (And male voice). The dialogue is displayed in a Tkinter GUI, spoken aloud via text-to-speech, and saved to a text file (`llm_conversation.txt`).

## Features
- Two AI characters with distinct personalities, powered by `gemma3:4b` via Ollama.
- Text-to-speech using Coqui TTS with VITS multi-speaker model (~110 voices).
- Tkinter GUI for real-time dialogue display.
- Saves conversation to `llm_conversation.txt`.
- Handles contractions (e.g., “we’re” → “we are”) for accurate pronunciation.
- CPU-only execution for broad compatibility.
- Customizable system prompts, topics, and speaker voices.

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
   - 16GB+ RAM recommended (TTS and LLM require ~4–8GB).
   - ~10GB free disk space for Python, dependencies, and models.

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
   - The repository includes `requirements_CPUonly.txt`. Install with:
     ```bash
     pip install -r requirements_CPUonly.txt
     ```
   - Contents of `requirements_CPUonly.txt`:
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

5. **Verify Ollama Setup**:
   - Start Ollama in a separate (Not in the VENV activated terminal, but its own terminal) terminal:
     ```bash
     ollama serve
     ```
   - Confirm the `gemma3:4b` model is installed (Depending on how your Ollama is installed, you may need to close the CMD console window you ran ollama serve in, to run Ollama in the system-tray instead. Then after installing a new model to Ollama close the system-tray version of Ollama again and open a CMD console window and run the Ollama Serve command. You don't need a VENV for the Ollama Serve command. Sorry for this complication, but I'm new to Python and had help from an AI-LLM to complete this project, so it may be a bit messy):
     ```bash
     ollama list
     ```
   - If not installed, run:
     ```bash
     ollama pull gemma3:4b
     ```

6. **Download Coqui TTS Model**:
   - The script automatically downloads `tts_models/en/vctk/vits` to `C:\Users\<YourUsername>\.cache\tts` on first run (Keep an eye on the CMD console window as you may have to press [Y] to accept the license-term for the TTS on the very first run). Ensure ~2GB free space.

## Running the Application
1. **Activate the Virtual Environment** (if not already active):
   ```bash
   cd JEL_LLMradiotheater_Ollama
   venv\Scripts\activate
   ```

2. **Start Ollama** (if not running. And, again, this must be done in a different CMD console window and not the one with the VENV activated):
   ```bash
   ollama serve
   ```

3. **Run the Script** (This is done in the VENV activated CMD console window. And be patient as it may take some time to load the LLM. On first run you should check the CMD console window to see if you need to accept the TTS license):
   ```bash
   python llm_radio_theater_CPUonly_Example_HusbondAndWife.py
   ```

4. **Interact with the GUI**:
   - **Start**: Begins the dialogue between the Woman (pro-living room) and Man (pro-garage).
   - **Pause/Resume**: Pauses or resumes the conversation.
   - **Change Topic**: Enter a new topic for the characters to discuss.
   - **TTS: On/Off**: Toggles text-to-speech.
   - Output appears in the GUI, is spoken aloud, and is saved to `llm_conversation.txt`.

**Expected Output**:
- Console: Lists `Using device for TTS: cpu` and ~110 VITS speaker IDs (e.g., `p225`, `p231`).
- GUI: Displays dialogue with proper spacing (e.g., “Darling, it’s utterly absurd…”).
- TTS: Pronounces (Not always, for some reason) contractions correctly (e.g., “it’s” as “it is”), ~10–20 seconds per sentence.
- Text File: Saves dialogue to `llm_conversation.txt` with correct formatting.

## Customization
- **Change Voices**:
  - Edit `llm_radio_theater_CPUonly_Example_HusbondAndWife.py` to use different VITS speaker IDs (e.g., `p226` for female, `p240` for male):
    ```python
    SPEAKER_1 = "p226"  # Female
    SPEAKER_2 = "p240"  # Male
    ```
  - Test voices:
    ```bash
    python -c "from TTS.api import TTS; tts = TTS(model_name='tts_models/en/vctk/vits').to('cpu'); tts.tts_to_file(text='This is a test.', speaker='p226', file_path='test_p226.wav')"
    ```

- **Change Topic or Prompts**:
  - Modify `INITIAL_TOPIC` in the script, e.g.:
    ```python
    INITIAL_TOPIC = "What do you want to talk about today?"
    ```
  - Update system prompts for different perspectives, e.g.:
    ```python
    SYSTEM_PROMPT_1 = "You are the wife in the relationship."
    SYSTEM_PROMPT_2 = "You are the husbond in the relationship."
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
Feel free to fork the repository, make improvements, and submit pull requests. Report issues via [GitHub Issues](https://github.com/JELSTUDIO/JEL_LLMradiotheater_Ollama/issues).

## License
MIT License. See [LICENSE](LICENSE) for details.
