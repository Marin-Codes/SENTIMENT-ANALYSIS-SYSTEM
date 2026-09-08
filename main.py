import os
import sys
import warnings

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import cv2
import numpy as np
import sounddevice as sd
import keyboard
import threading
import queue
import tempfile
from scipy.io.wavfile import write

try:
    from fer.fer import FER
except ImportError:
    from fer import FER

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError as e:
    WHISPER_AVAILABLE = False
    WhisperModel = None
    _whisper_import_error = e

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import transformers
    transformers.logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    _transformers_import_error = e

EMOTION_COLORS = {
    "happy":    (  0, 240, 120),
    "sad":      (240, 120,  40),
    "angry":    ( 30,  30, 255),
    "fear":     (220,  40, 200),
    "disgust":  ( 20, 180,  80),
    "surprise": (  0, 220, 255),
    "neutral":  (180, 180, 220),
}

SENTIMENT_COLORS = {
    "Positive": (  0, 240, 120),
    "Neutral":  (  0, 220, 255),
    "Negative": ( 30,  30, 255),
}

DEFAULT_COLOR = (180, 180, 220)
SARA_CYAN   = (255, 220,   0)
SARA_PURPLE = (220,  60, 160)

print("=" * 60)
print("  S.A.R.A. -- Sentiment Analysis & Response AI")
print("=" * 60)
print("Loading AI Models... (This may take a moment on first run)")

face_detector = FER(mtcnn=True)

if not WHISPER_AVAILABLE:
    raise ImportError(
        f"faster-whisper is not installed. Install it with:\n"
        f"  pip install faster-whisper\n"
        f"Original error: {_whisper_import_error}"
    )
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

if not TRANSFORMERS_AVAILABLE:
    raise ImportError(
        f"transformers is not installed. Install it with:\n"
        f"  pip install transformers\n"
        f"Original error: {_transformers_import_error}"
    )

def _load_sentiment_analyzer(model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_analyzer = _load_sentiment_analyzer()

print("Models loaded successfully!")

SAMPLE_RATE = 16000
audio_queue = queue.Queue()
is_recording = False
audio_frames = []
_audio_lock = threading.Lock()

current_text          = "Hold SPACEBAR to speak..."
current_sentiment     = "Waiting for input"
current_sentiment_key = "Neutral"
current_face_emotion  = "Neutral"
current_face_color    = DEFAULT_COLOR


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO STATUS] {status}")
    if is_recording:
        audio_queue.put(indata.copy())


def process_audio():
    global audio_frames

    with _audio_lock:
        frames_to_process = list(audio_frames)
        audio_frames.clear()

    if not frames_to_process:
        return None, None, None

    audio_data = np.concatenate(frames_to_process, axis=0)
    duration = len(audio_data) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    print(f"[DEBUG] Audio RMS: {rms:.4f}, duration: {duration:.2f}s")

    if duration < 0.4:
        return "Audio too short. Hold SPACEBAR longer.", "Neutral", "Neutral"
    if rms < 0.01:
        return "No speech detected (silence).", "Neutral", "Neutral"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, SAMPLE_RATE,
          np.clip(audio_data * 32767, -32768, 32767).astype(np.int16))
    temp_file.close()

    try:
        segments, _ = whisper_model.transcribe(temp_file.name, beam_size=5)
        transcribed_text = "".join([seg.text for seg in segments]).strip()
        print(f"[DEBUG] Transcription: '{transcribed_text}'")

        if not transcribed_text:
            return "No speech detected. Try again.", "Neutral", "Neutral"

        result = sentiment_analyzer(transcribed_text[:512])[0]
        stars  = int(result["label"].split()[0])

        if stars <= 2:
            sentiment_key = "Negative"
        elif stars == 3:
            sentiment_key = "Neutral"
        else:
            sentiment_key = "Positive"

        label = f"{sentiment_key} ({result['score']:.2f})"
        return transcribed_text, label, sentiment_key

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return "Error processing audio.", "Neutral", "Neutral"
    finally:
        try:
            os.remove(temp_file.name)
        except OSError:
            pass


def process_audio_and_update_ui():
    global current_text, current_sentiment, current_sentiment_key
    try:
        text, sentiment, key = process_audio()
    except Exception as e:
        print(f"[ERROR] Audio processing failed: {e}")
        return
    if text:
        current_text = text
    if sentiment:
        current_sentiment = sentiment
    if key:
        current_sentiment_key = key


_face_result_lock    = threading.Lock()
_latest_face_data    = []
_face_frame_queue    = queue.Queue(maxsize=1)
_face_thread_running = True


def face_detection_worker():
    global _face_thread_running
    while _face_thread_running:
        try:
            small_frame = _face_frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        faces = face_detector.detect_emotions(small_frame)
        parsed = []
        if faces:
            main_face = max(faces, key=lambda f: (f["box"][2] * f["box"][3]))
            x, y, w, h = main_face["box"]
            emotions   = main_face["emotions"]
            dominant   = max(emotions, key=emotions.get)
            confidence = emotions[dominant]
            color      = EMOTION_COLORS.get(dominant.lower(), DEFAULT_COLOR)
            label      = f"{dominant.capitalize()} ({confidence:.0%})"
            parsed.append({"box": (x, y, w, h), "color": color, "label": label})

        with _face_result_lock:
            global _latest_face_data
            _latest_face_data = parsed


def glass_overlay(img, pt1, pt2, tint=(20, 20, 40), alpha=0.62, border_color=None,
                  border_alpha=0.85, border_thickness=1, radius=14):
    x1, y1 = pt1
    x2, y2 = pt2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    panel = np.full(roi.shape, tint, dtype=np.uint8)
    blended = cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0)
    img[y1:y2, x1:x2] = blended
    if border_color:
        draw_rounded_rect(img, (x1, y1), (x2, y2), border_color,
                          thickness=border_thickness, radius=radius)


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness)


def draw_corner_brackets(img, pt1, pt2, color, length=20, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    ln = length
    cv2.line(img, (x1, y1), (x1 + ln, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + ln), color, thickness)
    cv2.line(img, (x2, y1), (x2 - ln, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + ln), color, thickness)
    cv2.line(img, (x1, y2), (x1 + ln, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - ln), color, thickness)
    cv2.line(img, (x2, y2), (x2 - ln, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 -                                                                                               ln), color, thickness)


def draw_sara_logo(img, x, y, frame_count):
    pulse = 0.75 + 0.25 * abs(np.sin(frame_count * 0.04))
    halo_alpha = 0.18 + 0.10 * abs(np.sin(frame_count * 0.04))
    glass_overlay(img, (x - 6, y - 28), (x + 198, y + 36),
                  tint=(40, 10, 60), alpha=halo_alpha + 0.55, border_color=None)
    glow_c = tuple(int(c * pulse) for c in SARA_PURPLE)
    draw_rounded_rect(img, (x - 6, y - 28), (x + 198, y + 36),
                      glow_c, thickness=1, radius=8)
    sara_color = tuple(int(c * pulse) for c in SARA_CYAN)
    cv2.putText(img, "S.A.R.A.", (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.80, sara_color, 2, cv2.LINE_AA)
    sub_color = tuple(int(c * pulse) for c in (160, 160, 220))
    cv2.putText(img, "Sentiment Analysis & Response AI", (x, y + 22),
                cv2.FONT_HERSHEY_PLAIN, 0.72, sub_color, 1, cv2.LINE_AA)
    dot_r = 4
    dot_x = x + 192
    dot_y = y - 14
    dot_alpha = 0.5 + 0.5 * abs(np.sin(frame_count * 0.08))
    dot_color = tuple(int(c * dot_alpha) for c in (0, 255, 120))
    cv2.circle(img, (dot_x, dot_y), dot_r + 2, tuple(int(c * 0.3) for c in dot_color), -1)
    cv2.circle(img, (dot_x, dot_y), dot_r, dot_color, -1)


def main():
    global is_recording, audio_frames, current_text, current_sentiment
    global current_sentiment_key, current_face_emotion, current_face_color
    global _face_thread_running

    face_thread = threading.Thread(target=face_detection_worker, daemon=True)
    face_thread.start()

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                             dtype="float32", callback=audio_callback)
    stream.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    cv2.namedWindow("__probe__", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("__probe__", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("__probe__", np.zeros((10, 10, 3), dtype=np.uint8))
    cv2.waitKey(1)
    rect = cv2.getWindowImageRect("__probe__")
    screen_w = rect[2] if rect[2] > 0 else 1280
    screen_h = rect[3] if rect[3] > 0 else 720
    cv2.destroyWindow("__probe__")

    aspect = cam_w / cam_h if cam_h > 0 else 16 / 9
    target_w = int(screen_w * 0.90)
    target_h = int(screen_h * 0.90)

    if target_w / aspect <= target_h:
        win_w, win_h = target_w, int(target_w / aspect)
    else:
        win_w, win_h = int(target_h * aspect), target_h

    WIN_NAME = "S.A.R.A. -- Sentiment Analysis & Response AI"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(WIN_NAME, win_w, win_h)
    cv2.waitKey(1)

    print("\n=== S.A.R.A. SYSTEM ONLINE ===")
    print("Look at the camera for facial emotion analysis.")
    print("HOLD the SPACEBAR to record your voice (Multilingual).")
    print("Press ESC to quit.\n")

    DETECT_EVERY = 3
    frame_count  = 0
    SMALL_W      = 320

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        frame_h, frame_w = frame.shape[:2]
        win_aspect   = win_w / win_h
        frame_aspect = frame_w / frame_h

        if frame_aspect > win_aspect:
            new_w = win_w
            new_h = max(1, int(win_w / frame_aspect))
        else:
            new_h = win_h
            new_w = max(1, int(win_h * frame_aspect))

        frame = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((win_h, win_w, 3), (12, 8, 20), dtype=np.uint8)
        y_off = (win_h - new_h) // 2
        x_off = (win_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = frame
        frame = canvas
        h, w = frame.shape[:2]

        scan_y = int((frame_count * 2) % h)
        scan_alpha = 0.06 + 0.04 * abs(np.sin(frame_count * 0.05))
        scan_overlay = frame.copy()
        cv2.line(scan_overlay, (0, scan_y), (w, scan_y), (80, 160, 220), 1)
        cv2.addWeighted(scan_overlay, scan_alpha, frame, 1 - scan_alpha, 0, frame)

        if frame_count % DETECT_EVERY == 0:
            small_h = max(1, int(SMALL_W * h / w))
            small   = cv2.resize(frame, (SMALL_W, small_h))
            try:
                _face_frame_queue.put_nowait(small)
            except queue.Full:
                pass

        with _face_result_lock:
            faces_snapshot = list(_latest_face_data)

        scale_x = w / SMALL_W
        scale_y = h / max(1, int(SMALL_W * h / w))

        for face in faces_snapshot:
            fx, fy, fw, fh = face["box"]
            fx = int(fx * scale_x)
            fy = int(fy * scale_y)
            fw = int(fw * scale_x)
            fh = int(fh * scale_y)

            color = face["color"]
            label = face["label"]
            current_face_emotion = label
            current_face_color   = color

            glow_soft = tuple(max(0, int(c * 0.20)) for c in color)
            glow_mid  = tuple(max(0, int(c * 0.50)) for c in color)
            draw_rounded_rect(frame, (fx - 6, fy - 6),
                              (fx + fw + 6, fy + fh + 6),
                              glow_soft, thickness=8, radius=16)
            draw_rounded_rect(frame, (fx - 2, fy - 2),
                              (fx + fw + 2, fy + fh + 2),
                              glow_mid, thickness=3, radius=13)
            draw_rounded_rect(frame, (fx, fy), (fx + fw, fy + fh),
                              color, thickness=1, radius=10)
            draw_corner_brackets(frame, (fx, fy), (fx + fw, fy + fh),
                                 color, length=18, thickness=2)

            lx = fx
            ly = max(22, fy - 10)
            (lw_px, lh_px), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
            glass_overlay(frame, (lx - 4, ly - lh_px - 6), (lx + lw_px + 8, ly + 4),
                          tint=(20, 10, 40), alpha=0.72,
                          border_color=color, border_thickness=1, radius=6)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

        if keyboard.is_pressed("space"):
            if not is_recording:
                with _audio_lock:
                    audio_frames.clear()
                is_recording = True
                print("[REC] Recording... (Release Spacebar to process)")
            while not audio_queue.empty():
                with _audio_lock:
                    audio_frames.append(audio_queue.get())

            pulse_r = int(12 + 5 * abs(np.sin(frame_count * 0.25)))
            mic_x, mic_y = w - 50, 46
            glow_alpha = 0.25 + 0.20 * abs(np.sin(frame_count * 0.25))
            mic_overlay = frame.copy()
            cv2.circle(mic_overlay, (mic_x, mic_y), pulse_r + 10, (30, 30, 220), -1)
            cv2.addWeighted(mic_overlay, glow_alpha, frame, 1 - glow_alpha, 0, frame)
            cv2.circle(frame, (mic_x, mic_y), pulse_r + 2, (0, 0, 160), -1)
            cv2.circle(frame, (mic_x, mic_y), pulse_r, (0, 60, 255), -1)
            cv2.circle(frame, (mic_x, mic_y), max(4, pulse_r - 5), (80, 120, 255), -1)
            glass_overlay(frame, (mic_x - 90, mic_y - 20), (mic_x - 14, mic_y + 12),
                          tint=(10, 5, 40), alpha=0.78)
            cv2.putText(frame, "LISTENING", (mic_x - 88, mic_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 120, 255), 1, cv2.LINE_AA)
        else:
            if is_recording:
                is_recording = False
                while not audio_queue.empty():
                    with _audio_lock:
                        audio_frames.append(audio_queue.get())
                current_text = "Processing..."
                print("[STOP] Processing audio...")
                threading.Thread(target=process_audio_and_update_ui,
                                 daemon=True).start()

        hud_h  = max(100, int(h * 0.16))
        hud_y0 = h - hud_h

        glass_overlay(frame, (0, hud_y0), (w, h),
                      tint=(18, 10, 36), alpha=0.82, border_color=None)

        edge_alpha = 0.55 + 0.20 * abs(np.sin(frame_count * 0.03))
        edge_overlay = frame.copy()
        cv2.line(edge_overlay, (0, hud_y0), (w, hud_y0), SARA_CYAN, 2)
        cv2.addWeighted(edge_overlay, edge_alpha, frame, 1 - edge_alpha, 0, frame)
        cv2.line(frame, (0, hud_y0 + 3), (w, hud_y0 + 3), (60, 40, 100), 1)

        divider_x = w // 2
        cv2.line(frame, (divider_x, hud_y0 + 8), (divider_x, h - 8), (50, 30, 80), 1)

        pad    = 18
        line1  = hud_y0 + int(hud_h * 0.28)
        line2  = hud_y0 + int(hud_h * 0.58)
        line3  = hud_y0 + int(hud_h * 0.86)
        fs_big = max(0.44, min(0.66, w / 950))
        fs_sm  = max(0.36, min(0.54, w / 1150))

        cv2.putText(frame, "VOICE INPUT", (pad, hud_y0 + 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.85,
                    tuple(int(c * 0.65) for c in SARA_CYAN), 1, cv2.LINE_AA)
        cv2.putText(frame, "FACE ANALYSIS", (divider_x + pad, hud_y0 + 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.85,
                    tuple(int(c * 0.65) for c in SARA_PURPLE), 1, cv2.LINE_AA)

        max_chars    = max(20, int((w // 2) / 10))
        display_text = (current_text if len(current_text) <= max_chars
                        else current_text[:max_chars - 3] + "...")
        cv2.putText(frame, display_text, (pad, line1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_big, (220, 220, 240), 1, cv2.LINE_AA)

        s_color = SENTIMENT_COLORS.get(current_sentiment_key, DEFAULT_COLOR)
        s_label = "SENTIMENT"
        sv_text = current_sentiment
        cv2.putText(frame, s_label, (pad, line2 - 2),
                    cv2.FONT_HERSHEY_PLAIN, 0.72,
                    tuple(int(c * 0.55) for c in s_color), 1, cv2.LINE_AA)
        s_lw = int(cv2.getTextSize(s_label, cv2.FONT_HERSHEY_PLAIN, 0.72, 1)[0][0])
        cv2.putText(frame, sv_text, (pad + s_lw + 8, line2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_sm, s_color, 1, cv2.LINE_AA)
        val_w = int(cv2.getTextSize(sv_text, cv2.FONT_HERSHEY_SIMPLEX, fs_sm, 1)[0][0])
        ux = pad + s_lw + 8
        uy = line2 + 3
        cv2.line(frame, (ux, uy), (ux + val_w, uy), s_color, 1)
        cv2.putText(frame, "Hold SPACE to speak", (pad, line3),
                    cv2.FONT_HERSHEY_PLAIN, 0.78, (80, 70, 120), 1, cv2.LINE_AA)

        f_color = current_face_color
        f_label = "EMOTION"
        f_text  = current_face_emotion
        fpad    = divider_x + pad
        cv2.putText(frame, f_label, (fpad, line1 - 2),
                    cv2.FONT_HERSHEY_PLAIN, 0.72,
                    tuple(int(c * 0.55) for c in f_color), 1, cv2.LINE_AA)
        f_lw = int(cv2.getTextSize(f_label, cv2.FONT_HERSHEY_PLAIN, 0.72, 1)[0][0])
        cv2.putText(frame, f_text, (fpad + f_lw + 8, line1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_sm, f_color, 1, cv2.LINE_AA)

        bar_y    = line2 + 6
        bar_maxw = w - fpad - pad
        pulse_w  = int(bar_maxw * (0.50 + 0.50 * abs(np.sin(frame_count * 0.06))))
        glass_overlay(frame, (fpad, bar_y - 6), (fpad + bar_maxw, bar_y + 6),
                      tint=(30, 20, 50), alpha=0.75)
        if pulse_w > 0:
            bar_overlay = frame.copy()
            cv2.rectangle(bar_overlay, (fpad, bar_y - 5),
                          (fpad + pulse_w, bar_y + 5), f_color, -1)
            cv2.addWeighted(bar_overlay, 0.65, frame, 0.35, 0, frame)
        draw_rounded_rect(frame, (fpad, bar_y - 6), (fpad + bar_maxw, bar_y + 6),
                          tuple(int(c * 0.4) for c in f_color), thickness=1, radius=4)

        draw_sara_logo(frame, 14, 38, frame_count)

        bracket_color = tuple(int(c * (0.3 + 0.15 * abs(np.sin(frame_count * 0.03))))
                              for c in SARA_CYAN)
        draw_corner_brackets(frame, (4, 4), (w - 4, hud_y0 - 4),
                             bracket_color, length=30, thickness=1)

        cv2.imshow(WIN_NAME, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    _face_thread_running = False
    cap.release()
    stream.stop()
    stream.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
