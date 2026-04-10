
Key features
------------
- Core feature 1 (e.g., real-time motion analysis)
- Core feature 2 (e.g., model inference & visualization)
- Core feature 3 (e.g., data import/export and logging)

Prerequisites
-------------
- Git
- Node.js (>= 14) or Python (3.8+) 



Quick start
-----------
1. Clone the repository

2. run these commands to install dependancies:
         python -m venv .venv
         source .venv\Scripts\activate on Windows
         pip install numpy pandas scipy scikit-learn joblib pylsl opencv-python mediapipe brainflow pyyaml gymnasium stable-baselines3
         pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3. load OpenBCI_GUI application 
        begin an 8 channel stream either synthetic(no headset) or cyton(with headset) 
        click start data stream
        open the networking tab under one of the 3 Windows
        switch the protocol to lsl 
        name stream one EMG of type obci_emg1 with the data type EMG
        then stream 2 as EEG of type obci_eeg1 with the data type TimeSeriesFilt
        click start lsl stream

3. quick start go to data processor and run it (press ctrl + c at any time to stop recording)





Project structure
-----------------
- /src or /app — application source code
- /models — trained models or model definitions
- /data — sample data, fixtures
- /docs — documentation (this file)
- /tests — automated tests
- package.json / requirements.txt — dependency manifest



