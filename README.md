🧭 Multilingual Speech → Emotion → Location Suggestions → Navigation System
Offline ASR (Faster-Whisper) • Offline Translation (Seamless M4T) • Emotion Analysis • Realistic Chennai Place Suggestions • OSRM Offline Routing
📦 1. Project Overview

This system allows users to:

Speak in multiple Indian languages

Convert that speech to text using Faster-Whisper ASR

Translate to English using Seamless M4T (offline)

Detect emotion

Suggest places near the user's location

Provide turn-by-turn navigation using OSRM (Offline Routing Engine)

Show full routes and maps using Folium

Everything runs offline except optional geocoding.

📁 2. Folder Structure
project/
│
├── app.py                      # Streamlit UI (main application)
├── speech_input.py            # Faster-Whisper + Seamless pipeline
├── emotion_model.py           # Emotion classifier
├── osrm_route.py              # OSRM routing helpers
│
├── seamless-m4t-v2-large/     # Seamless model folder (local)
├── final_emotion_model/       # Emotion model folder
└── india-latest.osm.pbf       # OSRM map file (downloaded)

⚙️ 3. Installation
Step 1 — Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   (Windows)

Step 2 — Install Required Packages
pip install streamlit folium streamlit-folium
pip install torch torchaudio transformers
pip install faster-whisper
pip install pyaudio webrtcvad noisereduce soundfile
pip install requests

🌏 4. Download OSRM Map File (PBF)

Download India map (.osm.pbf):

Official Geofabrik Link:

🔗 https://download.geofabrik.de/asia/india.html

(or choose Tamil Nadu, South India, Asia → depending on your need)

Download file:

india-latest.osm.pbf


Place it in your project directory.

🛠️ 5. Build OSRM Routing Backend (Windows Guide)
➤ Install OSRM Backend

Download Windows binaries:
🔗 https://github.com/Project-OSRM/osrm-backend/releases

Download osrm-backend-win64.zip
Extract anywhere.

📌 Step-by-Step Setup

Assume your file is:

C:\osrm\india-latest.osm.pbf

1. Extract the map
cd C:\osrm
osrm-extract india-latest.osm.pbf -p profiles/car.lua

2. Partition the map
osrm-partition india-latest.osrm

3. Customize
osrm-customize india-latest.osrm

4. Start OSRM Routing Server
osrm-routed india-latest.osrm


You should see:

[info] running and waiting for requests on 0.0.0.0:5000

🗺️ 6. Test OSRM Is Running

Open browser:

http://127.0.0.1:5000/route/v1/driving/80.28,13.05;80.23,12.59


If you get JSON, OSRM is working.

🚀 7. Run the Streamlit App

Inside project folder:

streamlit run app.py


The app will open automatically in your browser.

🎤 8. Using the App
Tab 1 — Speech

Speak in Hindi/Tamil/Telugu/etc → ASR → English Translation

Tab 2 — Emotion

Emotion detection from text:

happy

sad

angry

excited

neutral

Tab 3 — Suggestions

Shows nearby Chennai landmarks based on mood.

Tab 4 — Navigation

Offline OSRM routing with map and turn-by-turn route.

❗ 9. Troubleshooting
❌ OSRM not reachable

Check:

OSRM server is running

Correct URL (default: http://127.0.0.1:5000)

Correct .osm.pbf file

❌ PyAudio error

Install PyAudio binary for Windows:

pip install pipwin
pipwin install pyaudio

❌ Seamless model not found

Set correct folder:

SEAMLESS_DIR = r"C:\Users\sinth\seamless-m4t-v2-large"

❌ Emotion model issues

Verify folder:

final_emotion_model/
    config.json
    pytorch_model.bin
    tokenizer.json
    vocab.txt
    merges.txt
