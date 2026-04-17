# midi_mover

`midi_mover` is a webcam rhythm game that combines pose tracking with MIDI-driven gameplay.

The player sees themself live on screen. Five target circles are drawn around the head, positioned relative to the detected eye center. Notes from a target MIDI song are mapped to lanes and fall down a timeline on the right side of the screen, similar to a compact Guitar Hero / waterfall view. To hit a note, the player must place the correct hand (`L` or `R`) into the correct circle (`1`..`5`) at the correct time.

At the end of each song, the game evaluates the score and hit percentage. A persistent highscore board is shown after every song. If a new highscore entry is achieved, the game announces a headshot countdown, captures a cropped portrait from the camera, and stores it with the leaderboard entry.

This README is intended to be the main project entry point for implementation.

---

## Core gameplay summary

### Tracking model
Use:
- **left eye**
- **right eye**
- **left wrist**
- **right wrist**

Definitions:
- **head center** = midpoint between left eye and right eye
- **left hand** = left wrist keypoint
- **right hand** = right wrist keypoint

### Circle layout
There are exactly **5 circles** around the head.

Fixed circle numbering:
- **1** = far left
- **2** = upper-left
- **3** = above head
- **4** = upper-right
- **5** = far right

The numbering must be consistent across:
- live overlay
- timeline lanes
- note logic
- scoring
- logs/debug output
- highscore metadata if needed

### Note identity
Each target note is one of:
- `L1`, `L2`, `L3`, `L4`, `L5`
- `R1`, `R2`, `R3`, `R4`, `R5`

Examples:
- `R4` = right hand, circle 4
- `L1` = left hand, circle 1

### Gameplay screen layout
The game window is split into two permanent regions:

- **Left panel: 60% width**
  - live camera panel
  - person-centered crop
  - stretched to full panel height
  - horizontal padding on both sides if needed
  - overlay circles and wrist markers

- **Right panel: 40% width**
  - 5-lane vertical timeline / waterfall
  - notes move from top to bottom
  - horizontal "now" line is fixed at **25% panel height**
  - judged notes remain visible after crossing the line
  - judged notes turn:
    - **green** = hit
    - **red** = miss

### Song flow
1. Startup / initialization
2. Random MIDI selected from configured folder
3. Song title shown before start
4. Gameplay
5. Song result summary
6. If new highscore: headshot countdown `3 -> 2 -> 1 -> 0`, then capture
7. Highscore board shown after every song
8. Loop to next random song

---

## Project goals

The implementation should satisfy these product goals:

1. Real-time webcam-based pose tracking using eye and wrist keypoints.
2. Stable five-circle interaction around the detected head.
3. Real-time playback of a randomly selected target MIDI file.
4. Visual note timeline with 5 fixed lanes and falling notes.
5. Hit / miss evaluation based on hand, lane, and timing.
6. Persistent highscore system with portrait capture for new entries.
7. All tunable values must live in a separate YAML configuration file.

---

## Main technology choices

Recommended stack:

- **Python**
- **Ultralytics** for pose/keypoint inference
- **OpenCV (`cv2`)** for camera input and frame preprocessing
- **Pygame** for windowing, drawing, timing, and audio playback
- **Mido** for MIDI parsing
- **PyYAML** for config loading
- **SQLite** for persistent highscores
- **Pillow** for portrait cropping and image saving
- **NumPy** for geometry / numeric operations

The implementation should prefer clear modular structure over premature optimization.

---

## Configuration requirements

All hyperparameters and runtime configuration must be moved into a **separate YAML config file** for easy tuning.

This is mandatory.

Do **not** hardcode gameplay tuning values across the codebase.

### Examples of values that belong in the YAML config
- camera id
- MIDI folder path
- window size
- left/right panel width ratio
- circle radius
- circle offsets relative to head center
- circle outline thickness
- wrist smoothing values
- player crop padding
- note speed
- hit window timing
- good/perfect thresholds if introduced
- colors
- fonts / font sizes
- intro screen duration
- post-song screen duration
- countdown duration
- leaderboard size
- portrait crop expansion factors
- debug flags
- logging flags

### Example config file location
- `config/config.yaml`

### Important rule
Every future implementation ticket should check whether new tunables belong in the YAML config. If yes, add them there instead of hardcoding them in source files.

---

## Suggested repository structure

A practical structure is:

```text
midi_mover/
├── README.md
├── TODO.html
├── environment.yml
├── config/
│   └── config.yaml
├── assets/
│   ├── fonts/
│   ├── sounds/
│   └── images/
├── midi/
│   └── *.mid
├── data/
│   ├── highscores.db
│   └── portraits/
├── src/
│   ├── main.py
│   ├── app.py
│   ├── config_loader.py
│   ├── state_machine.py
│   ├── camera/
│   │   ├── capture.py
│   │   ├── cropper.py
│   │   └── overlays.py
│   ├── vision/
│   │   ├── pose_tracker.py
│   │   ├── keypoints.py
│   │   └── smoothing.py
│   ├── gameplay/
│   │   ├── circle_layout.py
│   │   ├── note_mapping.py
│   │   ├── judge.py
│   │   ├── score.py
│   │   └── timeline.py
│   ├── midi/
│   │   ├── loader.py
│   │   ├── parser.py
│   │   └── scheduler.py
│   ├── ui/
│   │   ├── screens.py
│   │   ├── hud.py
│   │   ├── leaderboard_view.py
│   │   └── transitions.py
│   └── highscore/
│       ├── db.py
│       ├── portraits.py
│       └── models.py
└── tests/
```

This exact structure is not mandatory, but the implementation should keep responsibilities separated.

---

## Environment setup

Use the Conda environment named:

```bash
conda create -n midi_mover python=3.10 -y
conda activate midi_mover
```

Install dependencies inside that environment.

### 1) Install PyTorch (CUDA example)

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 2) Install Python packages (including pyfluidsynth)

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install ultralytics opencv-python pygame mido pyyaml pillow numpy pyfluidsynth tensorboard
```

### 3) Install native FluidSynth library (required)

`pyfluidsynth` is only a Python binding. The app uses `audio.backend: pyfluidsynth` by default, so the native FluidSynth shared library must also be installed.

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y fluidsynth libfluidsynth3 libfluidsynth-dev
```

If you see this runtime error:

```text
ImportError: Couldn't find the FluidSynth library.
```

it means the native library is missing or not visible to the current environment.

### 4) Verify pyfluidsynth can load the native library

```bash
python -c "import fluidsynth; s=fluidsynth.Synth(); s.delete(); print('pyfluidsynth OK')"
```

### 5) Start the app

Smoke test path:

```bash
python main.py --smoke-test
```

Normal run:

```bash
python main.py --camera-id 0 --midi-dir /home/alex/data/midi_mover/midis --config config/default.yaml
```

### Notes

- Target-song output is expected to run through **pyfluidsynth** (no pygame fallback for song output).
- Ensure these YAML paths exist and are correct for your machine:
  - `pose.stage2_hand_model_name`
  - `audio.fluidsynth.soundfont_path`
  - MIDI directory passed via `--midi-dir`
- On first run, Ultralytics may download `yolo11n-pose.pt` automatically.

If extra packages are required during implementation, document them and keep the environment reproducible.

### Required workflow rule
Use the Conda environment named exactly:

```text
midi_mover
```

---

## Runtime inputs

The game input is defined by:
- the **camera ID** used by OpenCV
- the **folder containing target MIDI files**

These should come from the YAML config, not from scattered hardcoded constants.

At startup the application must:
1. open the configured camera
2. scan the configured MIDI folder
3. load or initialize the highscore database
4. randomly choose one MIDI file
5. display the song title extracted from the filename
6. begin the game loop

---

## Screen-by-screen behavior

### 1. Startup / initialization
Responsibilities:
- initialize config
- create window
- open camera
- initialize pose model
- scan MIDI folder
- initialize audio
- open or create highscore storage

Expected result:
- the application reaches a stable ready state
- the full pipeline can render and run

### 2. Song title intro
Responsibilities:
- select one random MIDI
- extract title from filename
- display title before gameplay
- create time buffer before first playable note

Expected result:
- player can read the title and prepare

### 3. Gameplay
Responsibilities:
- render the split layout
- track eyes and wrists
- compute head center
- place circles around the head
- render timeline
- play the song
- judge notes
- update score and hit percentage in real time

Expected result:
- complete playable loop

### 4. Song summary
Responsibilities:
- compute final score
- compute percentage hit
- decide whether score qualifies for leaderboard

Expected result:
- always show end-of-song result data

### 5. New highscore countdown
Responsibilities:
- show `3 -> 2 -> 1 -> 0`
- capture a headshot after countdown
- save portrait

Expected result:
- portrait is attached to new highscore entry

### 6. Highscore board
Responsibilities:
- display persistent leaderboard after every song
- display hit percentage
- highlight new entry if applicable

Expected result:
- leaderboard is visible after every completed song

---

## Implementation epics

The work is organized into three epics:

### Epic 1
**Liveview with circles and keypoint detection and visualization and also music playing**

Includes:
- camera pipeline
- pose/keypoint detection
- person-centered crop
- circle overlay rendering
- live markers and labels
- music playback plumbing

### Epic 2
**Loading of target MIDI files and timeline**

Includes:
- MIDI loading from folder
- random song selection
- filename-to-title extraction
- note mapping into `Lx` / `Rx`
- waterfall timeline rendering
- hit/miss judgment and note history coloring

### Epic 3
**Highscore system**

Includes:
- score persistence
- percentage hit storage
- leaderboard rendering
- new highscore detection
- headshot countdown and portrait capture
- updated leaderboard display after each song

See `TODO.html` for the task / subtask backlog and completion state.

---

## Required workflow for the implementation agent

These rules are mandatory.

### 1. Always fully complete each subtask
Do not partially complete a subtask and move on.

A subtask is only done when:
- the code is implemented
- the affected pipeline runs
- the subtask acceptance criteria are satisfied

### 2. After each subtask, the complete pipeline must run
This is a strict requirement.

After every subtask:
- run the app
- confirm the full project still starts and runs end-to-end
- fix regressions before moving on

No subtask should leave the project in a broken or non-runnable state.

### 3. Update `TODO.html` continuously
After finishing a subtask:
- mark that subtask as `DONE`

If all subtasks under a task are complete:
- mark the task as `DONE`

If all tasks under an epic are complete:
- mark the epic as `DONE`

`TODO.html` is the current truth source for execution status.

### 4. Do not use Git
Do not create branches.
Do not require Git operations.
Do not depend on commit-based workflow instructions.

### 5. Keep tuning values in YAML
Whenever new magic numbers or gameplay parameters appear, move them into the YAML config if they are tunable.

### 6. Preserve runnable integration
Do not build isolated features that only work later.
Prefer incremental vertical slices that keep the whole application executable.

---

## Acceptance standards

Each implemented feature should meet these standards:

### Functional
- behavior matches the product specification
- errors are handled clearly where practical
- startup is deterministic and understandable

### Visual
- overlays are readable
- timeline lanes are visually stable
- circle placement is consistent
- live crop is coherent and not distracting

### Timing
- note travel is deterministic
- "now" line behavior is fixed
- judgment is based on explicit timing windows from config

### Persistence
- highscores survive restart
- portraits are saved and reload correctly
- leaderboard remains readable even when no new score is added

### Maintainability
- modules have focused responsibilities
- config values are centralized
- implementation decisions are discoverable from code and comments

---

## Notes on MIDI interpretation

The project uses MIDI files as target songs.

Implementation expectations:
- load MIDI files from a configured folder
- pick one randomly on startup / next-song transition
- extract display title from filename
- convert MIDI note events into gameplay target notes
- schedule note visualization and audio playback from the same timing source where possible

The exact note-to-lane mapping strategy may evolve, but it must be:
- deterministic
- documented
- consistent across runs

---

## Notes on scoring

The scoring system should at minimum track:
- total notes
- notes hit
- hit percentage
- final score
- combo / max combo if implemented

Highscore comparisons should be explicit and stable. If multiple fields matter for ranking, define that ranking rule clearly in code and documentation.

---

## Notes on portrait capture

Portrait capture is only triggered when a run qualifies as a new highscore entry.

Requirements:
- visible countdown from `3` to `0`
- capture occurs after countdown
- crop should be a headshot, not the full-body gameplay crop
- saved portrait must be linked to the leaderboard entry
- portrait should still load after restarting the application

---

## How to use this repository

Typical workflow:

1. Activate the Conda environment:
   ```bash
   conda activate midi_mover
   ```

2. Make sure the config file points to:
   - a valid camera id
   - a valid MIDI folder

3. Start the application from the main entry point.

4. Use `TODO.html` to drive implementation progress and status tracking.

---

## Backlog file

The implementation backlog is stored in:

- `TODO.html`

That file is the working task board for the project and should be kept up to date during development.

---

## Current status

At the current stage, this repository should be treated as:
- product specification
- backlog definition
- implementation starting point

The implementation agent should use:
- this `README.md`
- the `TODO.html` backlog
- the YAML config requirement

as the primary guidance for building the project.

