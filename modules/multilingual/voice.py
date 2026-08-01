import torch
import torchaudio
from transformers import AutoModel
from huggingface_hub import login
import whisper
import soundfile as sf

# Optional: HuggingFace Login (if required for private repos)
login("")

_whisper_model = None

def _load():
    if _whisper_model is None:
        print("Loading Whisper model...")
        return whisper.load_model("base")


_whisper_model=_load()

def voice_to_text(audio_path: str) -> str:
    result = _whisper_model.transcribe(audio_path, fp16=False)
    if result["language"]=="en":
        return result["text"].strip()
    else:
        # ==========================================
        # 3. INDIC LANGUAGES ASR (Using AI4Bharat Conformer)
        # ==========================================

        print("\n--- Running Indic Language Transcription (AI4Bharat) ---")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device being used: {device}")

        # Load AI4Bharat Indic Conformer
        indic_model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
            token="",
            from_tf=True,
        ).to(device)

        # Model strictly eval mode मध्ये टाका
        indic_model.eval()

        # Load and preprocess audio
        data, sr = sf.read(audio_path)
        wav = torch.from_numpy(data).float()

        # If soundfile returns shape (samples, channels), transpose to (channels, samples)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2:
            wav = wav.T

        # Mono channel convert करा
        wav = torch.mean(wav, dim=0, keepdim=True)

        # 16kHz resample
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)

        wav = wav.to(device)
        with torch.inference_mode():
            # print("\n[CTC Decoding] (Faster on CPU):")
            # Language Code: Marathi = 'mr', Hindi = 'hi', Gujrati = 'gu', etc.
            # ctc_result = indic_model(wav, "te", "ctc")
            # print(ctc_result)

            # Note: RNNT CPU वर खूप वेळ घेतो, गरज असेल तरच वापरा
            print("\n[RNNT Decoding]:")
            rnnt_result = indic_model(wav, "te", "rnnt")
            return rnnt_result




