# KeyboardVideoToText
*Final project for CPSC 185.*

Fine-tuning a [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) model to predict typed text from overhead video, with audio, of typing on a keyboard.

## Environment
- The Conda environment is given at `environment.yml`. Create the environment with `conda env create -f environment.yml`.

## Dataset Creation
- `0training/collect_gui.py` is a GUI Python program that prompts sentences to type and records video and keystroke data. 
- Our keyboard video dataset, with full videos and keystroke timing data, is available at this [HuggingFace dataset](https://huggingface.co/datasets/andrewt28/keystroke-typing-videos).

<details>
<summary>Sample Data</summary>

Text: `The source said if approved, the authority would allow a transaction to be carried out.`

<video src="https://github.com/user-attachments/assets/ed61f0a7-8292-4608-a4a4-7ba5b6c469af.mp4" width=120></video>
</details>


## Training
- Install `llamafactory` via `pip install -e ".[torch,metrics]"` in the `1training/LLaMA-Factory` directory.
- Use `1training/0train.ipynb` to generate the augmented dataset and ensure that it's at `1training/LLaMA-Factory/data/keyboard_videos`. Look at the relative paths in `keyboard.json` to understand the directory structure for the .mp4 and .wav files.
- Run training via `1training/train.sh` which uses the configuration at `1training/train_keyboard.yml`.
