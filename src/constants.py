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

#Training
LOGIT_THRESHOLD = 0.0
POS_WEIGHT = 3
INITIAL_LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 25
LR_PLATEAU_PATIENCE = 10
LR_PLATEAU_FACTOR = 0.5