import os
import re
import sys
import threading
import wave

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

SAMPLE_RATE = 44100
OUTPUT_DIR = "generated_tones"

NOTE_OFFSETS = {
    "C": -9, "C#": -8, "DB": -8,
    "D": -7, "D#": -6, "EB": -6,
    "E": -5,
    "F": -4, "F#": -3, "GB": -3,
    "G": -2, "G#": -1, "AB": -1,
    "A": 0,  "A#": 1,  "BB": 1,
    "B": 2,
}


def parse_pitch(raw_value: str) -> tuple[float, str]:
    text = raw_value.strip().upper().replace("HZ", "").strip()
    try:
        hz = float(text)
        if hz <= 0 or hz > 20000:
            raise ValueError("Frequency out of audible range.")
        clean_hz = int(hz) if hz.is_integer() else round(hz, 2)
        return float(hz), f"{clean_hz}Hz"
    except ValueError:
        match = re.fullmatch(r"([A-G][#B]?)(-?\d+)", text)
        if not match:
            raise ValueError(
                f"Invalid tone '{raw_value}'. Use Hz (e.g. 196) or a note (e.g. G3)."
            )
        note, octave = match.group(1), int(match.group(2))
        semitones_from_a4 = (octave - 4) * 12 + NOTE_OFFSETS[note]
        hz = 440.0 * (2.0 ** (semitones_from_a4 / 12.0))
        return float(hz), f"{note}{octave}"


def render_relaxing_chime(
    base_hz: float,
    sample_rate: int = SAMPLE_RATE,
    chime_duration_sec: float = 2.2,
) -> np.ndarray:
    """
    Synthesizes a warm, slow-decaying singing-bowl-style chime with a gentle
    attack and subtle chorus warmth.
    """
    num_samples = max(1, int(sample_rate * chime_duration_sec))
    t = np.linspace(0.0, chime_duration_sec, num_samples, endpoint=False)

    root = base_hz if base_hz <= 440.0 else base_hz * 0.5
    octave = root * 2.0

    attack_len = min(int(0.04 * sample_rate), num_samples // 4)
    env_main = np.exp(-2.5 * (t / chime_duration_sec))
    env_warm = np.exp(-2.8 * (t / chime_duration_sec))
    env_oct = np.exp(-4.2 * (t / chime_duration_sec))

    if attack_len > 0:
        ramp = np.linspace(0.0, 1.0, attack_len)
        env_main[:attack_len] *= ramp
        env_warm[:attack_len] *= ramp
        env_oct[:attack_len] *= ramp

    tail_len = min(int(0.1 * sample_rate), num_samples // 4)
    if tail_len > 0:
        fade_out = np.linspace(1.0, 0.0, tail_len)
        env_main[-tail_len:] *= fade_out
        env_warm[-tail_len:] *= fade_out
        env_oct[-tail_len:] *= fade_out

    chime = (
        0.16 * np.sin(2.0 * np.pi * root * t) * env_main
        + 0.09 * np.sin(2.0 * np.pi * (root * 1.003) * t) * env_warm
        + 0.03 * np.sin(2.0 * np.pi * octave * t) * env_oct
    )
    return chime.astype(np.float32)


def synthesize_breath_tones(
    tone_a_hz: float,
    tone_b_hz: float,
    direction: str,
    duration_sec: float,
    hold_at_end: bool = True,
    hold_duration_sec: float = 1.0,
    add_chime: bool = False,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    1. Synthesizes the main breath cue over `duration_sec` (sweep + optional hold).
    2. Fades out the main cue cleanly at its end.
    3. Appends the relaxing chime at the end if `add_chime` is True.
    """
    num_samples = max(1, int(round(sample_rate * duration_sec)))
    t = np.linspace(0.0, duration_sec, num_samples, endpoint=False)

    low_hz = min(tone_a_hz, tone_b_hz)
    high_hz = max(tone_a_hz, tone_b_hz)

    if direction == "rising":
        target_hz = high_hz
        f0, f1 = low_hz, high_hz
    else:
        target_hz = low_hz
        f0, f1 = high_hz, low_hz

    effective_hold_sec = 0.0
    if hold_at_end and hold_duration_sec > 0.0:
        effective_hold_sec = min(hold_duration_sec, max(0.0, duration_sec - 0.1))

    sweep_end_sec = max(0.05, duration_sec - effective_hold_sec)

    sweep_freq = np.empty_like(t)
    rate_of_change = (f1 - f0) / sweep_end_sec
    for i, ti in enumerate(t):
        if ti <= sweep_end_sec:
            sweep_freq[i] = f0 + rate_of_change * ti
        else:
            sweep_freq[i] = f1

    dt = 1.0 / sample_rate
    moving_phase = 2.0 * np.pi * np.cumsum(sweep_freq) * dt
    target_phase = 2.0 * np.pi * target_hz * t

    main_audio = (0.35 * np.sin(moving_phase) + 0.22 * np.sin(target_phase)).astype(np.float32)

    # Smooth fade-in at start and fade-out at the end of the main tone cue
    fade_in_samples = min(int(0.03 * sample_rate), num_samples // 4)
    if fade_in_samples > 0:
        main_audio[:fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples)

    fade_out_samples = min(int(0.06 * sample_rate), num_samples // 4)
    if fade_out_samples > 0:
        main_audio[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples)

    if add_chime:
        chime = render_relaxing_chime(target_hz, sample_rate=sample_rate, chime_duration_sec=2.2)
        combined = np.concatenate([main_audio, chime])
    else:
        combined = main_audio

    return np.clip(combined, -1.0, 1.0).astype(np.float32)


def save_wav(filepath: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pcm16 = (audio * 32767.0).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


class BreathToneWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UX Experiment 001: Non-visual Progress Bar")
        self.resize(560, 580)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.lock = threading.Lock()
        self.audio: np.ndarray | None = None
        self.is_playing = False
        self.play_index = 0
        self.scrubbing = False
        self.stream: sd.OutputStream | None = None

        self._build_ui()
        self._init_audio_stream()
        self._refresh_history()
        if self.history_list.count() > 0:
            self._load_file(os.path.join(OUTPUT_DIR, self.history_list.item(0).text()))

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    def _init_audio_stream(self):
        try:
            self.stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as err:
            self.stream = None
            QMessageBox.warning(self, "Audio Output Warning", f"Could not open audio stream: {err}")

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        form_box = QGroupBox("Tone Parameters (Press Enter to Generate)")
        form_layout = QFormLayout(form_box)

        self.tone1_input = QLineEdit("196")
        self.tone2_input = QLineEdit("392")
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["rising", "falling"])
        self.duration_input = QLineEdit("30")

        hold_row = QHBoxLayout()
        self.hold_checkbox = QCheckBox("Hold at end")
        self.hold_checkbox.setChecked(True)
        self.hold_duration_input = QLineEdit("1")
        self.hold_duration_input.setFixedWidth(70)
        self.hold_duration_label = QLabel("sec")
        self.hold_checkbox.toggled.connect(self._on_hold_toggled)
        hold_row.addWidget(self.hold_checkbox)
        hold_row.addWidget(self.hold_duration_input)
        hold_row.addWidget(self.hold_duration_label)
        hold_row.addStretch()

        self.chime_checkbox = QCheckBox("Append relaxing chime at end")
        self.chime_checkbox.setChecked(True)
        self.generate_btn = QPushButton("Generate & Save .WAV")

        form_layout.addRow("Tone 1 (Hz or Note, e.g. 196 or G3):", self.tone1_input)
        form_layout.addRow("Tone 2 (Hz or Note, e.g. 392 or G4):", self.tone2_input)
        form_layout.addRow("Direction:", self.direction_combo)
        form_layout.addRow("Duration (seconds, e.g. 30):", self.duration_input)
        form_layout.addRow("Hold on Target:", hold_row)
        form_layout.addRow("End Cue:", self.chime_checkbox)
        form_layout.addRow("", self.generate_btn)

        for widget in (self.tone1_input, self.tone2_input, self.duration_input, self.hold_duration_input):
            widget.returnPressed.connect(self.generate_tone)
        self.generate_btn.clicked.connect(self.generate_tone)
        layout.addWidget(form_box)

        player_box = QGroupBox("Playback & Scrubbing")
        player_layout = QVBoxLayout(player_box)

        self.file_label = QLabel("No file loaded")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderPressed.connect(self._on_scrub_start)
        self.slider.sliderMoved.connect(self._on_scrub_move)
        self.slider.sliderReleased.connect(self._on_scrub_end)

        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.stop_btn = QPushButton("Stop")
        self.time_label = QLabel("0.0s / 0.0s")
        self.play_btn.clicked.connect(self.toggle_play)
        self.stop_btn.clicked.connect(self.stop_playback)

        btn_row.addWidget(self.play_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.time_label)

        player_layout.addWidget(self.file_label)
        player_layout.addWidget(self.slider)
        player_layout.addLayout(btn_row)
        layout.addWidget(player_box)

        hist_box = QGroupBox("Generated History (Double-click to load)")
        hist_layout = QVBoxLayout(hist_box)
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self._on_history_double_click)
        hist_layout.addWidget(self.history_list)
        layout.addWidget(hist_box)

    def _on_hold_toggled(self, checked: bool):
        self.hold_duration_input.setEnabled(checked)
        self.hold_duration_label.setEnabled(checked)

    def _duration_sec(self) -> float:
        with self.lock:
            if self.audio is None:
                return 0.0
            return len(self.audio) / SAMPLE_RATE

    def _audio_callback(self, outdata, frames, time_info, status):
        outdata.fill(0.0)
        with self.lock:
            if not self.is_playing or self.audio is None:
                return
            total_len = len(self.audio)
            if self.play_index >= total_len:
                self.is_playing = False
                return
            end_idx = min(self.play_index + frames, total_len)
            chunk = self.audio[self.play_index:end_idx]
            outdata[:len(chunk), 0] = chunk
            self.play_index = end_idx
            if self.play_index >= total_len:
                self.is_playing = False

    def generate_tone(self):
        try:
            hz1, label1 = parse_pitch(self.tone1_input.text())
            hz2, label2 = parse_pitch(self.tone2_input.text())
            direction = self.direction_combo.currentText()
            duration = float(self.duration_input.text().strip())
            if duration <= 0.1 or duration > 3600:
                raise ValueError("Duration must be between 0.1 and 3600 seconds.")

            hold_at_end = self.hold_checkbox.isChecked()
            hold_sec = float(self.hold_duration_input.text().strip() or "0")
            if hold_at_end and (hold_sec < 0.0 or hold_sec >= duration):
                raise ValueError("Hold duration must be 0 or greater and less than total duration.")
        except ValueError as err:
            QMessageBox.critical(self, "Invalid Input", str(err))
            return

        self.stop_playback()
        add_chime = self.chime_checkbox.isChecked()
        dur_label = f"{int(duration)}s" if duration.is_integer() else f"{duration}s"
        hold_label = f"{int(hold_sec)}s" if hold_sec.is_integer() else f"{hold_sec}s"
        hold_suffix = f"_hold{hold_label}" if (hold_at_end and hold_sec > 0) else ""
        chime_suffix = "_chime" if add_chime else ""
        filename = f"{label1}-{label2}_{direction}_{dur_label}{hold_suffix}{chime_suffix}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        audio = synthesize_breath_tones(
            hz1,
            hz2,
            direction,
            duration,
            hold_at_end=hold_at_end,
            hold_duration_sec=hold_sec,
            add_chime=add_chime,
            sample_rate=SAMPLE_RATE,
        )
        save_wav(filepath, audio, SAMPLE_RATE)
        self._load_file(filepath, audio)
        self._refresh_history(select_name=filename)

    def _load_file(self, filepath: str, audio: np.ndarray | None = None):
        self.stop_playback()
        if audio is None:
            with wave.open(filepath, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        with self.lock:
            self.audio = audio
            self.play_index = 0
            self.is_playing = False
        self.file_label.setText(f"Loaded: {os.path.basename(filepath)}")
        self.slider.setValue(0)
        self.time_label.setText(f"0.0s / {self._duration_sec():.1f}s")

    def toggle_play(self):
        with self.lock:
            if self.audio is None:
                return
            if self.is_playing:
                self.is_playing = False
                self.play_btn.setText("Play")
            else:
                if self.play_index >= len(self.audio) - int(0.05 * SAMPLE_RATE):
                    self.play_index = 0
                self.is_playing = True
                self.play_btn.setText("Pause")

    def stop_playback(self):
        with self.lock:
            self.is_playing = False
            self.play_index = 0
        self.slider.setValue(0)
        self.play_btn.setText("Play")
        self.time_label.setText(f"0.0s / {self._duration_sec():.1f}s")

    def _on_scrub_start(self):
        self.scrubbing = True

    def _on_scrub_move(self, value: int):
        total = self._duration_sec()
        sec = (value / 1000.0) * total
        self.time_label.setText(f"{sec:.1f}s / {total:.1f}s")
        with self.lock:
            if self.audio is not None:
                self.play_index = min(int(sec * SAMPLE_RATE), len(self.audio))

    def _on_scrub_end(self):
        total = self._duration_sec()
        sec = (self.slider.value() / 1000.0) * total
        with self.lock:
            if self.audio is not None:
                self.play_index = min(int(sec * SAMPLE_RATE), len(self.audio))
        self.scrubbing = False

    def _on_tick(self):
        with self.lock:
            if self.audio is None:
                return
            total_len = len(self.audio)
            idx = self.play_index
            playing = self.is_playing

        total = total_len / SAMPLE_RATE
        if total <= 0:
            return

        if not playing:
            self.play_btn.setText("Play")

        if playing and not self.scrubbing:
            pos_sec = min(idx / SAMPLE_RATE, total)
            self.slider.blockSignals(True)
            self.slider.setValue(int((pos_sec / total) * 1000))
            self.slider.blockSignals(False)
            self.time_label.setText(f"{pos_sec:.1f}s / {total:.1f}s")

    def _refresh_history(self, select_name: str | None = None):
        self.history_list.clear()
        if not os.path.isdir(OUTPUT_DIR):
            return
        wavs = sorted(
            (f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".wav")),
            key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
            reverse=True,
        )
        for name in wavs:
            self.history_list.addItem(name)
            if select_name and name == select_name:
                self.history_list.setCurrentRow(self.history_list.count() - 1)

    def _on_history_double_click(self, item):
        filepath = os.path.join(OUTPUT_DIR, item.text())
        if os.path.isfile(filepath):
            self._load_file(filepath)
            with self.lock:
                self.play_index = 0
                self.is_playing = True
            self.play_btn.setText("Pause")

    def closeEvent(self, event):
        with self.lock:
            self.is_playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BreathToneWindow()
    win.show()
    sys.exit(app.exec())
