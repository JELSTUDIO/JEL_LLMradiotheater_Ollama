# JEL LLM Radio Theater, special NASA fictional mission edition with extra sound-effects

Demo video on Youtube:

[![Watch demo video](https://img.youtube.com/vi/SdbZNV4STAE/0.jpg)](https://www.youtube.com/watch?v=SdbZNV4STAE)

First follow the installation-instructions for version 2.0.0 (The Chatterbox-GPU version on the main page)

Then, when you have that set-up, follow the installation-instructions below to get version 2.1.0 which is required to run the fictional NASA radio-play.

## Installation
Follow these steps in order to update v2.0.0 to v2.1.0 (This will not break v2.0.0 so those scripts will still work):

1. **Copy or move files into main folder**:
   - Copy or move all files (Except the README.md file) from the NASA folder into the main radio-theater folder.

2. **Activate the Virtual Environment**:
   ```bash
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   - This switches the audio playback system from simpleaudio to pygame, which is required to run the NASA radio-play correctly with the extra sound-effects:
     ```bash
     pip3 install pygame
     ```

4. **Verify Ollama Setup**:
   - Start Ollama in a separate (Not in the VENV activated terminal, but its own terminal) terminal:
     ```bash
     ollama serve
     ```
   - Confirm the `gemma3:12b` model is installed, or change model in the .py file (Depending on how your Ollama is installed, you may need to close the CMD console window you ran ollama serve in, to run Ollama in the system-tray instead. Then after installing a new model to Ollama close the system-tray version of Ollama again and open a CMD console window and run the Ollama Serve command. You don't need a VENV for the Ollama Serve command. Sorry for this complication, but I'm new to Python and had help from an AI-LLM to complete this project, so it may be a bit messy):
     ```bash
     ollama list
     ```
   - If not installed, run:
     ```bash
     ollama pull gemma3:12b
     ```

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

3. **Run the Script** (This is done in the VENV activated CMD console window. And be patient as it may take some time to load the LLM and, when using Chatterbox, generate the speeches. On first run you should check the CMD console window to see if you need to accept the TTS license):
   - For the the GPU-based Chatterbox TTS running the special NASA fictional mission, run:
     ```bash
     python Chatterbox_GPU_NASAmission.py
     ```

4. **Interact with the GUI**:
   - **Start**: Begins the dialogue between the Woman (pro-living room) and Man (pro-garage).
   - **Pause/Resume**: Pauses or resumes the conversation.
   - **Change Topic**: Enter a new topic for the characters to discuss.
   - **TTS: On/Off**: Toggles text-to-speech.
   - **WRAP**: Click this when you want the conversation to begin to wrap up with a natural ending.
   - **TIMER**: This will count down in seconds to an automatic wrap-up of the conversation so it doesn't continue forever (In the new GPU-script it defaults to 3600 seconds, but you can modify this in the .py file)
   - Output appears in the GUI, is spoken aloud, and is saved to `llm_conversation.txt`.

**Expected Output**:
- Text File: Saves dialogue to `llm_conversation.txt` with correct formatting.

## Troubleshooting
- **TTS Errors**:
  - Make sure pygame is installed (See higher up in this text)

## License
NASA sound-files are public-domain but NOT for commercial purposes.

