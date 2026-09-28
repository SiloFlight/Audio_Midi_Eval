1. Project Task
    - Model Goal: The goal of this project was to train a neural network to detect the presence of piano note onset / offset from audio for automated evaluation of piano playing.
    - How to Run (all commands run from repo root)
        - Index Creation :
          ```
          python3 -m scripts.build_track_indices
          ```
          Builds both the MAESTRO and MAPS track indices as CSVs under `data/Indices/`. Run once before anything else. Use `-i MAESTRO` or `-i MAPS` to build just one, `-o` to overwrite an index that already exists.
        - Model Training (By Curriculum) :
          ```
          python3 -m scripts.train_curriculum_stage -s <stage> -i <input_type> -d <audio_duration> -a <accepted_duration>
          ```
          - `<stage>` : `ISOL`, `Chords`, `Full` — run in this order; each stage after the first automatically loads and continues from the previous stage's saved checkpoint.
          - `<input_type>` : `Waveform`, `CQT`, `HCQT`
          - `<audio_duration>` : `1`, `2`, `4` (seconds)
          - `<accepted_duration>` : `0.1`, `0.25`, `0.5` (seconds)
        - Model Evaluation :
          ```
          python3 -m scripts.evaluate_curriculum_stage -s <stage> -i <input_type> -d <audio_duration> -a <accepted_duration>
          ```
          - `<stage>` : `ISOL`, `Chords`, `Full` — picks which checkpoint to load; run all three to build the full cross-stage matrix below. Each run evaluates that one checkpoint against *every* stage's own validation set, not just its own.
          - `<input_type>`, `<audio_duration>`, `<accepted_duration>` : same choices as above — must match the checkpoint's own training config.
1. Model Arch
    - The HCQT model is a CNN approach with two event heads that are concatenated into a singular tensor inspired from Spotify's Basic Pitch model. Specifically, there are two independent heads feeding an onset/ offset layer which are concatenated into a single vector for model output. Each head is a `Conv2D(32 filters, 5x5, same padding) → BatchNorm → ReLU → Conv2D(1 filter, 3x3, same padding) → GlobalAveragePooling1D → (1, NOTE_BINS)`. 
    - Model predictions are done in raw logit space with a threshold of 0.0 with a mean reduced Weighted Binary Cross Entropy loss function.
    - Model checkpoints are identified by Input Type, Input Duration, Accepted Duration.
        - Models are trained on either a HCQT,CQT, or waveform representation of audio.
        - The duration of the audio representation is also varied between 1,2, or 4 second audio inputs.
        - The accepted duration determines the radius around the onset and offset s.t. labels have positive labels.
1. Training
    - Curriculum Approach : The training of the model was done in stages. The initial model was trained on monophonic notes, then further trained on polyphonic chords until being trained on actual piano pieces.
        - Isolated monophonic notes : MAPS (ISOL)
        - Simple polyphonic chords : MAPS (UCHO, RAND)
        - Piano Pieces : MAPS(MUS) + MAESTRO
1. Results
    - Cross-stage evaluation matrix on HCQT, 4 second audio input, .5 second accepted duration. Each row in the result of evaluating the model checkpoint on 'Evaluation stage''s validation set.

      | Checkpoint | Evaluation stage | Precision | Recall | F1 |
      |---|---|---|---|---|
      | ISOL | **ISOL** | 0.5793 | 0.5958 | **0.5874** |
      | ISOL | Chords | 0.4501 | 0.0155 | 0.0299 |
      | ISOL | Full | 0.3949 | 0.0801 | 0.1332 |
      | Chords | ISOL | 0.3205 | 0.6329 | 0.4256 |
      | Chords | **Chords** | 0.3887 | 0.2860 | **0.3296** |
      | Chords | Full | 0.4058 | 0.1510 | 0.2201 |
      | Full | ISOL | 0.1299 | 0.5999 | 0.2136 |
      | Full | Chords | 0.3442 | 0.0215 | 0.0404 |
      | Full | **Full** | 0.4012 | 0.5647 | **0.4691** |
1. Citations:
    - The raw audio + midi recordings originates from the MAPS (MIDI-Aligned Piano Sounds) and the MAESTRO dataset. These two datasets are cited below.
        - MAPS : V. Emiya, R. Badeau, and B. David, "Multipitch estimation of piano sounds using a new probabilistic spectral smoothness principle," IEEE Transactions on Audio, Speech and Language Processing, 2010.
        - MAESTRO : Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset." In International Conference on Learning Representations (ICLR), 2019.
    - The architecture of my model is heavily inspired by Spotify's Basic Pitch model discussed in the paper: "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation".
        - The publication of the paper can be found at: [arXiv](https://arxiv.org/abs/2203.09893).