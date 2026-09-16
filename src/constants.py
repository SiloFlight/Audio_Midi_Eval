from pathlib import Path


#------------- Audio Parameters
DEFAULT_SAMPLE_RATE = 22050

#------------- File Locations
MAESTRO_Folder = Path("data/MAESTRO")
MAPS_Folder = Path("data/MAPS")
Synthetic_Folder = Path("data/Synthetic")
Index_Folder = Path("data/Indices")

#Model Types
INPUT_DURATIONS = [1,2,4]
ACCEPTED_DURATIONS = [.1,.25,.5]

#CQT Params
MIN_NOTE = 21 #MIDI code for A0
MAX_NOTE = 108 #MIDI code for C8

HOP_LENGTH = int(DEFAULT_SAMPLE_RATE / 30) # Ideally 30 bins per second.


#HCQT Params
HARMONICS = sorted([.5,1,2,3])

#Derived Constants
NOTE_BINS = max(0,MAX_NOTE-MIN_NOTE+1)