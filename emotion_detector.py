import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import threading
import queue
import time
import logging

# Setup logging to see exactly what's happening in the background
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base emotions from DeepFace
BASE_EMOTIONS = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# Advanced fused moods
FUSED_MOODS = {
    ("happy", "surprise"): ("Excited", "🤩"),
    ("sad", "fear"): ("Anxious", "😰"),
    ("angry", "disgust"): ("Irritated", "😠"),
    ("happy", "neutral"): ("Content", "😌"),
    ("sad", "neutral"): ("Melancholy", "😔"),
    ("fear", "surprise"): ("Shocked", "😱"),
    ("angry", "fear"): ("Panic", "😫"),
    ("happy", "fear"): ("Nervous", "😬"),
}

class EmotionAI:
    def __init__(self):
        self.current_mood = "Initializing..."
        self.current_emoji = "⏳"
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()
        self.running = True

        # UI State
        self.hud_alpha = 0.6
        self.pulse_val = 0
        self.pulse_dir = 1

        logger.info("Warming up AI models... Please wait.")
        try:
            # Force load models by analyzing a blank image
            # We use a real-looking blank image to avoid some internal errors
            dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

        # Start worker thread
        self.worker = threading.Thread(target=self._analysis_worker, daemon=True)
        self.worker.start()

    def _get_fused_mood(self, emotion_dict):
        """Combines top 2 emotions to create a more nuanced mood."""
        sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
        top1, prob1 = sorted_emotions[0]
        top2, prob2 = sorted_emotions[1]

        if prob1 - prob2 < 0.2:
            for key, value in FUSED_MOODS.items():
                if set(key) == {top1, top2}:
                    return value[0], value[1]

        return top1.capitalize(), BASE_EMOTIONS.get(top1, "❓")

    def _analysis_worker(self):
        """Background thread for heavy DeepFace analysis."""
        logger.info("Worker thread started.")
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None: break

                # Pre-process frame: DeepFace works better if the image isn't too huge
                # Resize to a standard height of 720p if it's larger
                h, w = frame.shape[:2]
                if h > 720:
                    scale = 720 / h
                    frame = cv2.resize(frame, (int(w * scale), 720))

                # Analyze
                # We use 'opencv' as it's the fastest. If it fails, user may need 'retinaface'.
                results = DeepFace.analyze(frame, actions=['emotion'],
                                           enforce_detection=False,
                                           detector_backend='opencv')

                if results and len(results) > 0:
                    # DeepFace returns a list of results (one for each face)
                    res = results[0]

                    # Check if a face was actually detected in the result
                    # When enforce_detection=False, DeepFace might still return a result
                    # but without region coordinates if no face is found.
                    if 'region' in res and res['region'] is not None:
                        emotion_probs = res.get('emotion', {})
                        if emotion_probs:
                            mood_name, emoji = self._get_fused_mood(emotion_probs)
                            self.result_queue.put((mood_name, emoji))
                        else:
                            self.result_queue.put(("Unknown", "❓"))
                    else:
                        self.result_queue.put(("Searching for face...", "😶"))
                else:
                    self.result_queue.put(("No face detected", "😶"))

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Analysis Error: {e}")
                self.result_queue.put(("AI Error", "⚠️"))

    def draw_hud(self, frame, mood, emoji):
        """Renders a modern, attractive HUD over the frame."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        bg_color = (20, 20, 20, int(255 * self.hud_alpha))
        accent_color = (0, 255, 255, 255)
        text_color = (255, 255, 255, 255)

        hud_w, hud_h = 320, 110
        margin = 20

        draw.rounded_rectangle(
            [margin, margin, margin + hud_w, margin + hud_h],
            radius=15, fill=bg_color
        )
        draw.line([margin, margin + hud_h, margin + hud_w, margin + hud_h], fill=accent_color, width=3)

        try:
            font_header = ImageFont.truetype("arial.ttf", 18)
            font_main = ImageFont.truetype("seguiemj.ttf", 36)
        except:
            font_header = ImageFont.load_default()
            font_main = ImageFont.load_default()

        draw.text((margin + 20, margin + 15), "AI EMOTION ANALYSIS", font=font_header, fill=accent_color)

        emotion_text = f"{emoji} {mood.upper()}"
        w = draw.textlength(emotion_text, font=font_main)
        draw.text((margin + (hud_w - w)//2, margin + 40), emotion_text, font=font_main, fill=text_color)

        self.pulse_val += self.pulse_dir * 2
        if self.pulse_val > 10 or self.pulse_val < 0:
            self.pulse_dir *= -1

        is_valid_mood = mood not in ["Searching for face...", "AI Error", "Unknown", "Loading AI Models...", "No face detected"]
        indicator_color = (0, 255, 0, 150 + self.pulse_val * 10) if is_valid_mood else (255, 0, 0, 150)
        draw.ellipse([margin + hud_w - 30, margin + 20, margin + hud_w - 20, margin + 30], fill=indicator_color)

        combined = Image.alpha_composite(pil_img, overlay)
        return cv2.cvtColor(np.array(combined.convert("RGB")), cv2.COLOR_RGB2BGR)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam.")
            return

        logger.info("Webcam opened. Starting loop.")

        frame_count = 0
        mood_name = "Loading AI Models..."
        mood_emoji = "⏳"

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # Analyze every 10th frame to keep it fast
            if frame_count % 10 == 0:
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            try:
                while not self.result_queue.empty():
                    mood_name, mood_emoji = self.result_queue.get_nowait()
            except queue.Empty:
                pass

            frame = self.draw_hud(frame, mood_name, mood_emoji)
            cv2.imshow('Emotion AI - Modern HUD', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.running = False
        self.frame_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ai = EmotionAI()
    ai.run()
