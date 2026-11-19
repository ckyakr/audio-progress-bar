# UX Experiment 001: Non-visual Progress Bar

An eyes-free audio progress indicator designed to communicate elapsed time, remaining duration, and phase completion without looking at a screen.

## The Hypothesis

Visual progress bars require active visual attention—which breaks focus during breathwork, meditation, or any eyes-closed / eyes-busy task. 

**Can a closing musical interval, target pitch hold, and warm completion chime give the brain an intuitive sense of progress and arrival purely by ear?**

## How the Audio Cues Work

Instead of relying on harsh beeps or countdown voiceovers, progress is mapped to a sequential three-stage harmonic cue:

1. **Anchor & Sweep (Progress):** A stationary **anchor tone** marks the destination pitch, while a **moving tone** glides steadily toward it. As the pitch interval narrows toward unison, you intuitively feel how close the interval is to finishing.
   * **Rising cues** (low to high) map naturally to inhales or building energy.
   * **Falling cues** (high to low) map naturally to exhales or settling down.
2. **Target Hold (Arrival):** Once the moving tone meets the anchor pitch, both tones optionally hold steady in unison (default: `1s`) so you experience a clear moment of stillness and arrival before the main tone cue fades out cleanly.
3. **Appended Completion Chime (Transition):** When enabled, a warm, `2.2s` singing-bowl-style chime is **appended after** the main tone cue finishes (`total length = duration + 2.2s`), marking the transition without overlapping the held note.

## Features

* **Flexible Pitch Input:** Enter frequencies in Hertz (`196`, `392`) or musical note names (`G3`, `G4`).
* **Rising & Falling Modes:** Configure directional cues for inhale/exhale cycles or general timers.
* **Configurable End Behavior:** Toggleable unison hold at the target pitch (customizable duration, default `1s`) and toggleable appended completion chime.
* **Real-Time Scrubbing:** Thread-safe audio playback with a live timeline scrubber to audition specific moments (e.g., jumping straight to the convergence, hold, or appended chime).
* **`.WAV` Export & History:** Automatically saves generated cues into `generated_tones/` with descriptive filenames (such as `196Hz-392Hz_rising_30s_hold1s_chime.wav`) and lets you reload past generations with a double-click.

## Requirements

* **Python 3.10+**
* **Python Packages** (listed in `requirements.txt`):
  * `numpy` — waveform, pitch-sweep, and chime synthesis
  * `sounddevice` — real-time audio playback
  * `PySide6` — desktop GUI (`UX Experiment 001: Non-visual Progress Bar`) and timeline scrubber
* **System Audio Library (Linux):** `sounddevice` requires PortAudio.
  * **Arch / CachyOS:** `sudo pacman -S portaudio`
  * **Ubuntu / Debian:** `sudo apt install libportaudio2`
  * *(Windows and macOS typically bundle PortAudio automatically via `pip`.)*

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/ux-001-non-visual-progress-bar.git
   cd ux-001-non-visual-progress-bar
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python tone.py
   ```

4. **Generate & audition:**
   * Set your start/target tones, direction (`rising` or `falling`), main cue duration, hold toggle/duration, and chime toggle.
   * Press **Enter** or click **Generate & Save .WAV**.
   * Use **Play**, **Stop**, or drag the scrubber slider to test how the progression, unison hold, and appended chime feel.
