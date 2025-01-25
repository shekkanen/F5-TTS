import os
import numpy as np
import soundfile as sf
from zipfile import ZipFile
from flask import Flask, request, send_file
import torch
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT
from omegaconf import OmegaConf

app = Flask(__name__)

# Load the pretrained model (do this globally when the app starts)
model_cfg_path = "src/f5_tts/configs/F5TTS_Base_train.yaml"
model_cfg = OmegaConf.load(model_cfg_path).model.arch
ckpt_path = "ckpts/model_commonvoice_fi_librivox_fi_vox_populi_fi_20241217/model_last.pt" # Replace with your actual ckpt path if needed
vocab_path = "ckpts/model_commonvoice_fi_librivox_fi_vox_populi_fi_20241217/vocab.txt" # Use Emilia vocab for wider character support

vocoder = load_vocoder(vocoder_name="vocos", is_local=False)
ema_model = load_model(DiT, model_cfg, ckpt_path, mel_spec_type="vocos", vocab_file=vocab_path)

# Define the Finnish alphabet phrases
alphabet_sentences = [
    f"{chr(97 + i)} niin kuin {['aurinko', 'banaani', 'celcius', 'delfiini', 'elefantti', 'fanta', 'golf', 'haukka', 'ilta', 'juusto', 'kissa', 'lamppu', 'meri', 'nalle', 'omena', 'puu', 'quark', 'ruusu', 'sydän', 'talo', 'ukko', 'vene', 'watti', 'xenon', 'yksi', 'zebra'][i]}"
    for i in range(26)
]

@app.route("/generate", methods=["POST"])
def generate_audio():
    if "audio" not in request.files:
        return "No audio file provided", 400

    # Save the reference audio file temporarily
    ref_audio_file = request.files["audio"]
    ref_audio_path = "ref_audio.wav" # Fixed filename
    ref_audio_file.save(ref_audio_path)

    # Process the reference audio
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, "")
    except Exception as e:
        os.remove(ref_audio_path) # Clean up even if preprocessing fails
        return f"Error preprocessing reference audio: {e}", 500

    # Generate audio files
    os.makedirs("generated_audio", exist_ok=True)
    generated_files = []
    try:
        for i, sentence in enumerate(alphabet_sentences):
            output_path = f"generated_audio/{chr(97 + i)}.wav"
            audio_segment, sample_rate, _ = infer_process(
                ref_audio, ref_text, sentence, ema_model, vocoder, mel_spec_type="vocos"
            )
            sf.write(output_path, audio_segment, sample_rate)
            generated_files.append(output_path)

        # Zip the generated files
        zip_filename = "alphabet_audio.zip"
        with ZipFile(zip_filename, "w") as zipf:
            for file_name in os.listdir("generated_audio"):
                zipf.write(os.path.join("generated_audio", file_name), file_name)

        return send_file(zip_filename, as_attachment=True)

    except Exception as e:
        # Clean up generated audio directory and ref_audio if generation fails
        os.system("rm -rf generated_audio") # Use os.system for simplicity in this case
        os.remove(ref_audio_path)
        return f"Error during TTS generation: {e}", 500

    finally:
        # Clean up temporary files and directory after success or failure
        os.remove(ref_audio_path)
        os.system("rm -rf generated_audio") # Use os.system for simplicity in this case
        # Clean up zip file as well
        if os.path.exists("alphabet_audio.zip"):
            os.remove("alphabet_audio.zip")


@app.route("/")
def index():
    return "F5-TTS Alphabet Voice Cloning Service is running!"

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")