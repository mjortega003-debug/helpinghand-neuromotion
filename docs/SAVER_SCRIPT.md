# Saver Script (DataSaver) - What it does and how to use it

## What is the saver script?

The saver script is the component responsible for **persisting streaming data to disk** in a reliable, consistent format.
It makes sure that once EEG data is being read (mock or UltraCortex), you actually end up with files in `logs/` that can later be:

* cleaned (data cleaning step)
* visualized (Unity or Python visualization step)
* reused for training/debugging

In our repo, the saver script is implemented as:

* `src/core/data_saver.py`

## What it saves

It produces two CSV files (one run per file by default):

1. **Raw EEG data**

* File: `logs/neuromotion_raw_eeg_YYYYMMDD_HHMMSS.csv`
* Format (wide rows):

  * `timestamp, input_source, ch_0, ch_1, ... ch_{N-1}`

This is the "ground truth" stream data and is useful for debugging and reprocessing later.

2. **Processed outputs**

* File: `logs/neuromotion_processed_YYYYMMDD_HHMMSS.csv`
* Format:

  * `timestamp, input_source, intent, command, feat_0, feat_1, ... (optional)`

This is useful for quickly driving Unity visualizations (intent + command) and tracking the feature vectors.

## Why we want it separate from cleaning and visualization

We keep saving separate because we want data to be recorded even if:

* the cleaner crashes
* the classifier changes
* the visualization fails

The saver should be "dumb and reliable" - it only writes, it does not make ML decisions.

## How to wire it into UltraCortexSimulation

1. Add the import:

* `from src.core.data_saver import DataSaver, DataSaverConfig`

2. In `UltraCortexSimulation.__init__`, create the saver after config is loaded:

* `self.saver = DataSaver(DataSaverConfig(
    input_source=self.mode,
    sample_rate=self.config.get("sampling_rate", None),
    channel_count=self.config.get("channels", None),
    save_raw=True,
    save_processed=True,
    unique_per_run=True
  ))`

3. At the beginning of `start_simulation()`:

* `self.saver.start()`

4. Inside the real-time loop, after retrieving `eeg_arr`:

* `self.saver.save_raw_chunk(eeg_arr)`

5. After features + intent + command are computed:

* `self.saver.save_processed(intent=intent, command=command, features=features)`

6. In the `finally:` block (shutdown cleanup):

* `self.saver.close()`

## Notes / defaults

* Filenames are unique per run (timestamp suffix) so you don’t overwrite previous sessions.
* If you want a single consistent filename each time, set:

  * `unique_per_run=False`
* If you only want raw or only processed outputs:

  * `save_raw=False` or `save_processed=False`
