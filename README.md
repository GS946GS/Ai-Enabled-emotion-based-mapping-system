# 🧭 Multilingual Speech → Emotion → Location Suggestions → Navigation System

**Offline ASR (Faster-Whisper) • Offline Translation (Seamless M4T) • Emotion Analysis • Location Suggestions • Offline Navigation (OSRM)**

---

# 📦 1. Project Overview

This system allows users to:

- 🎤 Speak in **multiple Indian languages**
- 📝 Convert speech to text using **Faster-Whisper ASR**
- 🌍 Translate to English using **Seamless M4T (offline)**
- 😊 Detect **emotion from the text**
- 📍 Suggest **places near the user's location**
- 🗺️ Provide **turn-by-turn navigation using OSRM (offline routing engine)**
- 🧭 Display **interactive maps and routes using Folium**

✅ The system works **fully offline** except optional geocoding.

---

# 📁 2. Folder Structure

```
project/
│
├── app.py                      # Streamlit UI (main application)
├── speech_input.py             # Faster-Whisper + Seamless pipeline
├── emotion_model.py            # Emotion classifier
├── osrm_route.py               # OSRM routing helpers
│
├── seamless-m4t-v2-large/      # Seamless translation model (local)
├── final_emotion_model/        # Emotion model files
└── india-latest.osm.pbf        # OSRM map file
```

---

# ⚙️ 3. Installation

## Step 1 — Create Virtual Environment

```
python -m venv venv
```

Activate environment (Windows):

```
venv\Scripts\activate
```

---

## Step 2 — Install Required Packages

```
pip install streamlit folium streamlit-folium
pip install torch torchaudio transformers
pip install faster-whisper
pip install pyaudio webrtcvad noisereduce soundfile
pip install requests
```

---

# 🌏 4. Download OSRM Map File (PBF)

Download India map (.osm.pbf):

Official Geofabrik link:

https://download.geofabrik.de/asia/india.html

You may also download:

- Tamil Nadu map
- South India map
- Asia map

Download file:

```
india-latest.osm.pbf
```

Place the file in your **project directory**.

---

# 🛠️ 5. Build OSRM Routing Backend (Windows Guide)

## Install OSRM Backend

Download Windows binaries:

https://github.com/Project-OSRM/osrm-backend/releases

Download:

```
osrm-backend-win64.zip
```

Extract anywhere on your system.

---

## Step-by-Step Setup

Assume your map file location is:

```
C:\osrm\india-latest.osm.pbf
```

### 1️⃣ Extract the map

```
cd C:\osrm
osrm-extract india-latest.osm.pbf -p profiles/car.lua
```

---

### 2️⃣ Partition the map

```
osrm-partition india-latest.osrm
```

---

### 3️⃣ Customize

```
osrm-customize india-latest.osrm
```

---

### 4️⃣ Start OSRM Routing Server

```
osrm-routed india-latest.osrm
```

You should see:

```
[info] running and waiting for requests on 0.0.0.0:5000
```

---

# 🗺️ 6. Test OSRM Server

Open browser:

```
http://127.0.0.1:5000/route/v1/driving/80.28,13.05;80.23,12.59
```

If you see **JSON output**, OSRM is working correctly.

---

# 🚀 7. Run the Streamlit App

Inside your project folder:

```
streamlit run app.py
```

The web app will automatically open in your browser.

---

# 🎤 8. Using the Application

## Tab 1 — Speech

Speak in languages such as:

- Hindi
- Tamil
- Telugu
- Malayalam
- Kannada

The system performs:

```
Speech → Text → English Translation
```

---

## Tab 2 — Emotion Detection

The system analyzes the text and detects emotion:

- happy
- sad
- angry
- excited
- neutral

---

## Tab 3 — Place Suggestions

Based on detected emotion, the system suggests **nearby landmarks in Chennai**.

Example:

| Emotion | Suggested Places |
|-------|-------|
| Happy | Marina Beach |
| Sad | Elliot's Beach |
| Excited | VGP Universal Kingdom |
| Neutral | Phoenix Mall |

---

## Tab 4 — Navigation

Uses **OSRM offline routing** to provide:

- Route path
- Distance
- Turn-by-turn directions
- Interactive map display

---

# 🧠 Technologies Used

- **Python**
- **Streamlit**
- **Faster-Whisper (Offline Speech Recognition)**
- **Seamless M4T (Offline Translation)**
- **Transformers**
- **PyTorch**
- **OSRM (Offline Routing Engine)**
- **Folium (Interactive Maps)**

---

# ❗ 9. Troubleshooting

## OSRM Not Reachable

Check:

- OSRM server is running
- Correct URL:

```
http://127.0.0.1:5000
```

- Correct `.osm.pbf` map file

---

## PyAudio Installation Error

Install PyAudio using:

```
pip install pipwin
pipwin install pyaudio
```

---

## Seamless Model Not Found

Set the correct model path in your code:

```
SEAMLESS_DIR = r"C:\Users\sinth\seamless-m4t-v2-large"
```

---

## Emotion Model Issues

Verify folder structure:

```
final_emotion_model/
    config.json
    pytorch_model.bin
    tokenizer.json
    vocab.txt
    merges.txt
```

---

# 📜 License

This project is released under the **MIT License**.

---

# 👨‍💻 Author

**Sinthanaiselvan G**

GitHub  
https://github.com/GS946GS
