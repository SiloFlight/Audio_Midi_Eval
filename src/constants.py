from pathlib import Path


#------------- Audio Parameters
DEFAULT_SAMPLE_RATE = 22050

#------------- File Locations
MAESTRO_Folder = Path("data/MAESTRO")
MAPS_Folder = Path("data/MAPS")
Synthetic_Folder = Path("data/Synthetic")
Index_Folder = Path("data/Indices")
Models_Folder = Path("models")

#Model Types
INPUT_DURATIONS = [1,2,4]
ACCEPTED_DURATIONS = [.1,.25,.5]

#CQT Params
MIN_NOTE = 21 #MIDI code for A0
MAX_NOTE = 108 #MIDI code for C8

HOP_LENGTH = int(DEFAULT_SAMPLE_RATE / 30) # Ideally 30 bins per second.


#HCQT Params
HARMONICS = sorted([.5,1,2])

#Derived Constants
NOTE_BINS = max(0,MAX_NOTE-MIN_NOTE+1)

#Label Layout
ONSET_ROW = 0
OFFSET_ROW = 1

#Dataset Pipeline
SHUFFLE_BUFFER_SIZE = 10_000
MAX_EXAMPLES_PER_TRACK = SHUFFLE_BUFFER_SIZE // 10
MAX_TRACK_DURATION = 2 * 60  # seconds
SAMPLE_PER_EVENT = 5
NEGATIVE_PERCENTAGE = 2.5
TRACK_LOAD_WORKERS = 4  # concurrent load_track_info/context calls per _example_generator

#Training
LOGIT_THRESHOLD = 0.0
POS_WEIGHT = 3
INITIAL_LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 25
LR_PLATEAU_PATIENCE = 10
LR_PLATEAU_FACTOR = 0.5

#Curriculum Stages
CURRICULUM_SPLIT_SEED = 42  # fixed regardless of training seed, so the split itself never moves
CURRICULUM_VAL_TRACKS = {"ISOL": 140, "Chords": 414}  # ~10K validation examples each, see diagnose_maps_volume.py
CURRICULUM_NEGATIVE_PERCENTAGE = {"ISOL": 0.75, "Chords": 0.4}  # ~10% negative rate, see diagnose_negative_percentage_sweep.py
CURRICULUM_MAX_TRACK_DURATION = {"Full": 720}  # seconds - see diagnose_stage3_duration_sweep.py
CURRICULUM_MAX_EXAMPLES_PER_TRACK = {"Full": 2000}  # see diagnose_stage3_duration_sweep.py
CURRICULUM_LEARNING_RATE = {
    "ISOL": INITIAL_LEARNING_RATE,  # same as default until tuned per stage
    "Chords": INITIAL_LEARNING_RATE,  # same as default until tuned per stage
    "Full": 5e-6,  # 1/4 of INITIAL_LEARNING_RATE - starting point, reactionary to how stage 3 training goes
}