# transformer.py
import io
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import SpeakerCNN


class SpeakerIdentificationTransformer:
    def __init__(self, model_path, target_label="Rowney Downey"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SpeakerCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.target_label = target_label

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

    def _audio_to_mel(self, audio_path, sr=22050):
        y, _ = librosa.load(audio_path, sr=sr)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig = plt.figure(figsize=(3, 3))
        librosa.display.specshow(mel_db, sr=sr, cmap="magma")
        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf).convert("RGB")

    def predict(self, audio_path):
        img = self._audio_to_mel(audio_path)
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(img)
            prob = torch.sigmoid(logit).item()

        label = self.target_label if logit.item() >= 0 else "Non-Target Speaker"

        return {
            "status": "success",
            "predicted_speaker": label,
            "probability_of_target_speaker": round(prob, 4)
        }
