# JEL LLM Radio Theater - Special EnglishOnly-Chatterbox version

This version of the RadioTheater script is hard-coded to specifically use ONLY the English-Only version of Chatterbox (Provided you have it installed), which I personally think sounds better than the new multi-lingual version (Even if using the English-Only version in the new multi-Lingual update)

To use this you need to edit 2 files in the VENV-folder (So of course you still need to have gone through the normal install-procedure first)

First, make sure you have the English-Only Chatterbox model installed. You will find it here:

   ```
   %USERPROFILE%\.cache\huggingface\hub\models--ResembleAI--chatterbox\snapshots\1b475dffa71fb191cb6d5901215eb6f55635a9b6
   ```

That link will expand to something like: C:\Users\YourUserName\.cache\huggingface\hub\models--ResembleAI--chatterbox\snapshots\1b475dffa71fb191cb6d5901215eb6f55635a9b6

The number: `1b475dffa71fb191cb6d5901215eb6f55635a9b6` is the hash for the correct Chatterbox-model. If you only have other numbers in the snapshots-folder, then they are different versions of Chatterbox and won't work.

If you have the correct version installed, go on to manually edit the (Or copy/paste the 2 included) Chatterbox-files in the RadioTheater VENV folder (This will not affect any other installation you may have of Chatterbox, as the VENV only pertains to this particular JEL-repo)

Look for this folder: `JEL_LLMradiotheater_Ollama\venv\Lib\site-packages\chatterbox`

In there you will find 2 files: `tts.py` and `vc.py`.

Replace them with the 2 updated files included here in `SpecialEnglishOnlyChatterboxVersion`, or edit them manually (See below how to do that)



## Manually editing the 2 VENV Chatterbox files
Follow these steps if you edit the 2 files manually:

0. **Make sure you only edit files in this folder**:
   - The local VENV of JEL's LLM-RadioTheater (This will not affect other Chatterbox-installations):
     ```
     `JEL_LLMradiotheater_Ollama\venv\Lib\site-packages\chatterbox`
     ```
1a. **In tts.py**:
   - Replace this line:
     ```
     def from_pretrained(cls, device) -> 'ChatterboxTTS':
     ```
   - with this line:
     ```
     def from_pretrained(cls, device, revision=None) -> 'ChatterboxTTS':
     ```
1b. **In tts.py**:
   - Replace this line:
     ```
     local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)
     ```
   - with this line:
     ```
     local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath, revision=revision or "main")
     ```

2a. **In vc.py**:
   - Replace this line:
     ```
     def from_pretrained(cls, device) -> 'ChatterboxVC':
     ```
   - with this line:
     ```
     def from_pretrained(cls, device, revision=None) -> 'ChatterboxVC':
     ```
2b. **In vc.py**:
   - Replace this line:
     ```
     local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)
     ```
   - with this line:
     ```
     local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath, revision=revision or "main")
     ```


## v2.4.3 (The English-Only Chatterbox version)
Once you've done that you can use this script, which you should edit to define your own conversation-topics and LLM-models used etc:`v2.4.3_test26.py`

You run it by entering this command in the CMD console window:
     ```
     python v2.4.3_test26.py
     ```


## License
MIT License. See [LICENSE](LICENSE) for details.
(Be aware the NASA-version has its own license. See the NASA folder for that)
