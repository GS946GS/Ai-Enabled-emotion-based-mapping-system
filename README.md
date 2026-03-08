# 🧭 AI-Enabled Emotion Based Mapping System

**Multilingual Speech → Emotion Detection → Smart Location Suggestions → Navigation**

🚀 **This project was developed within 2 days during the _GenAI NxtWave Buildathon_.**

The system demonstrates how **Generative AI, speech recognition, emotion analysis, and geospatial routing** can be combined to create an intelligent navigation assistant.

---

# 📦 Project Overview

This project is an **AI-powered navigation assistant** that understands **user speech in multiple languages**, detects the **user's emotional state**, and suggests **nearby places with navigation routes**.

The system works mostly **offline** using modern AI models for speech recognition, translation, and emotion analysis.

---

# 🚀 Features

🎤 **Multilingual Speech Input**

Supports speech in Indian languages such as:

- Tamil  
- Hindi  
- Telugu  
- Malayalam  
- Kannada  

Speech is converted to text using **Faster-Whisper (offline ASR)**.

---

🌍 **Offline Translation**

Speech text is translated to English using **Seamless M4T** running locally.

---

😊 **Emotion Detection**

The translated text is analyzed to detect user emotions such as:

- Happy
- Sad
- Angry
- Excited
- Neutral

Emotion detection uses a **fine-tuned Transformer model**.

---

📍 **Emotion-Based Place Suggestions**

Based on the detected emotion, the system suggests places such as:

| Emotion | Suggested Places |
|-------|-------|
| Happy | Beaches, Parks |
| Sad | Calm relaxing areas |
| Excited | Amusement parks |
| Neutral | Shopping malls |

---

🗺 **Offline Navigation**

The system provides:

- Route distance
- Navigation path
- Interactive map

Routing is done using **OSRM (Open Source Routing Machine)**.

---

📊 **Interactive Map Visualization**

Maps and routes are displayed using:

- **Folium**
- **Streamlit**

---

# 🧠 System Architecture

```
User Speech
     │
     ▼
Faster-Whisper ASR
     │
     ▼
Seamless M4T Translation
     │
     ▼
Emotion Detection Model
     │
     ▼
Emotion-Based Place Suggestions
     │
     ▼
OSRM Routing Engine
     │
     ▼
Map + Navigation (Folium)
```

---

# 📂 Project Structure

```
Ai-Enabled-emotion-based-mapping-system
│
├── app.py
│   Streamlit main application
│
├── speech_input.py
│   Handles speech recording, Faster-Whisper ASR and translation
│
├── emotion_model.py
│   Emotion classification using transformer model
│
├── osrm_route.py
│   Handles route requests from OSRM server
│
├── seamless-m4t-v2-large/
│   Local translation model
│
├── final_emotion_model/
│   Pretrained emotion classification model
│
└── india-latest.osm.pbf
    Map file for OSRM routing
```

---

# ⚙️ Installation

## Create Virtual Environment

```
python -m venv venv
```

Activate:

```
venv\Scripts\activate
```

---

## Install Dependencies

```
pip install streamlit
pip install folium streamlit-folium
pip install torch torchaudio transformers
pip install faster-whisper
pip install pyaudio webrtcvad
pip install noisereduce soundfile
pip install requests
```

---

# 🌏 Download Map Data

Download the **India map file (.osm.pbf)**:

https://download.geofabrik.de/asia/india.html

Place the file inside the project directory.

```
india-latest.osm.pbf
```

---

# 🛠 Setup OSRM Routing Server

Download OSRM backend:

https://github.com/Project-OSRM/osrm-backend/releases

Extract the binaries.

---

### Step 1 — Extract Map

```
cd C:\osrm
osrm-extract india-latest.osm.pbf -p profiles/car.lua
```

---

### Step 2 — Partition

```
osrm-partition india-latest.osrm
```

---

### Step 3 — Customize

```
osrm-customize india-latest.osrm
```

---

### Step 4 — Start Routing Server

```
osrm-routed india-latest.osrm
```

Server runs at:

```
http://127.0.0.1:5000
```

---

# 🧪 Test OSRM

Open in browser:

```
http://127.0.0.1:5000/route/v1/driving/80.28,13.05;80.23,12.59
```

If JSON appears, the routing server works correctly.

---

# 🚀 Run the Application

Inside the project directory:

```
streamlit run app.py
```

The web interface will open automatically.

---

# 🎤 Application Workflow

### 1️⃣ Speech Input
User speaks in native language.

### 2️⃣ Speech Recognition
Faster-Whisper converts speech → text.

### 3️⃣ Translation
Seamless M4T converts text → English.

### 4️⃣ Emotion Detection
Emotion classifier detects emotional state.

### 5️⃣ Location Suggestions
System recommends nearby places.

### 6️⃣ Navigation
OSRM calculates route and displays map.

---

# 🧰 Technologies Used

- Python
- Streamlit
- PyTorch
- Transformers
- Faster-Whisper
- Seamless M4T
- OSRM Routing Engine
- Folium Maps

---

# ⚠ Troubleshooting

## PyAudio installation error

```
pip install pipwin
pipwin install pyaudio
```

---

## OSRM server not reachable

Ensure:

- OSRM server is running
- Correct URL:

```
http://127.0.0.1:5000
```

---

## Missing models

Verify these folders exist:

```
seamless-m4t-v2-large/
final_emotion_model/
```

---

# 🏆 Buildathon

This project was **designed and developed within 2 days during the _GenAI NxtWave Buildathon_**, demonstrating rapid prototyping using modern AI and geospatial technologies.

---

# 👨‍💻 Author

**Sinthanaiselvan G**

GitHub  
https://github.com/GS946GS

---

⭐ If you find this project useful, consider giving it a star.
