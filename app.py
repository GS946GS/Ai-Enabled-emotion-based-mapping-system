# app.py
import streamlit as st
from streamlit_folium import st_folium
import math
import folium
import requests
import importlib

st.set_page_config(page_title="Speech → Emotion → Suggestions → Navigation", layout="wide")

# ---------------------------
# Robust imports for emotion model
# ---------------------------
try:
    from emotion_model import predict_emotion as emo_predict
    st.sidebar.success("Loaded emotion_model.predict_emotion")
except Exception:
    try:
        from speech_input import predict_emotion as emo_predict
        st.sidebar.success("Loaded predict_emotion from speech_input")
    except Exception:
        emo_predict = None
        st.sidebar.warning("Could not import predict_emotion; emotion detection will be disabled.")

# ---------------------------
# Robust imports for speech pipeline
# ---------------------------
process_input = None
process_audio = None
record_audio = None
indian_langs = None

speech_modules_checked = []
try:
    mod = importlib.import_module("speech_pipeline")
    speech_modules_checked.append("speech_pipeline")
    if hasattr(mod, "process_input"):
        process_input = mod.process_input
    if hasattr(mod, "indian_langs"):
        indian_langs = mod.indian_langs
except Exception:
    pass

if process_input is None:
    try:
        mod = importlib.import_module("input_pipeline_fasterwhisper")
        speech_modules_checked.append("input_pipeline_fasterwhisper")
        if hasattr(mod, "process_input"):
            process_input = mod.process_input
        if hasattr(mod, "indian_langs"):
            indian_langs = indian_langs or getattr(mod, "indian_langs", None)
    except Exception:
        pass

if process_input is None:
    try:
        mod = importlib.import_module("speech_input")
        speech_modules_checked.append("speech_input")
        process_audio = getattr(mod, "process_audio", None)
        record_audio = getattr(mod, "record_audio", None)
        if process_audio and not process_input:
            def _process_input(lang):
                aud = None
                if record_audio:
                    aud = record_audio()
                else:
                    aud = getattr(mod, "record_audio", lambda *a, **k: None)()
                return process_audio(aud, src_lang=lang)
            process_input = _process_input
    except Exception:
        pass

if process_input is None and process_audio is None:
    st.warning("No speech pipeline functions found. Expected process_input() or process_audio().")
    st.info(f"Checked modules: {speech_modules_checked}")

# ---------------------------
# Robust imports for OSRM/navigation
# ---------------------------
nav_modules_checked = []
osrm_route = None
pretty_route_summary = None
route_to_folium = None
show_map = None

try:
    mod = importlib.import_module("navigation_route")
    nav_modules_checked.append("navigation_route")
    osrm_route = getattr(mod, "osrm_route", None)
    pretty_route_summary = getattr(mod, "pretty_route_summary", None)
    route_to_folium = getattr(mod, "route_to_folium", None)
    show_map = getattr(mod, "show_map", None)
except Exception:
    pass

if osrm_route is None:
    try:
        mod = importlib.import_module("osrm_route")
        nav_modules_checked.append("osrm_route")
        osrm_route = getattr(mod, "osrm_route", None)
        pretty_route_summary = getattr(mod, "pretty_route_summary", None)
        route_to_folium = getattr(mod, "route_to_folium", None)
        show_map = getattr(mod, "show_map", None)
    except Exception:
        pass

if osrm_route is None:
    st.warning("osrm_route() not found. Checked modules: " + ", ".join(nav_modules_checked))

# ---------------------------
# Local OSRM checker
# ---------------------------
def is_osrm_running(url="http://127.0.0.1:5000", timeout=2.0):
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        if r.status_code == 200:
            return True
    except:
        pass
    try:
        r = requests.get(f"{url}/route/v1/driving/0,0;0,0", timeout=timeout)
        return r.status_code == 200
    except:
        return False

# ---------------------------
# Suggestion DB (Chennai realistic)
# ---------------------------
SUGGESTION_DB = {
    "happy": [
        {"name": "Marina Beach", "lat": 13.0500, "lon": 80.2820},
        {"name": "Besant Nagar (Elliot's Beach)", "lat": 12.9831, "lon": 80.2698},
        {"name": "Semmozhi Poonga", "lat": 13.0346, "lon": 80.2467},
    ],
    "sad": [
        {"name": "Kapaleeswarar Temple", "lat": 13.0389, "lon": 80.2655},
        {"name": "Nageswara Park", "lat": 13.0410, "lon": 80.2578},
        {"name": "Quiet Garden", "lat": 13.0410, "lon": 80.2578},
    ],
    "angry": [
        {"name": "Chembarambakkam Lake", "lat": 12.9542, "lon": 80.0681},
        {"name": "Koyambedu Ground", "lat": 13.0695, "lon": 80.1940},
    ],
    "excited": [
        {"name": "Phoenix Marketcity", "lat": 12.9941, "lon": 80.2268},
        {"name": "Express Avenue Mall", "lat": 13.0827, "lon": 80.2707},
    ],
    "neutral": [
        {"name": "Anna Nagar Tower Park", "lat": 13.0651, "lon": 80.2109},
        {"name": "Forum Vijaya Mall", "lat": 13.0232, "lon": 80.2669},
    ],
    "default": [
        {"name": "Local Park", "lat": 13.0350, "lon": 80.2500},
        {"name": "Popular Cafe", "lat": 13.0300, "lon": 80.2600},
    ]
}

# ---------------------------
# Helpers
# ---------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def sort_and_attach_distance(sources, base_lat, base_lon):
    out = []
    for s in sources:
        d = haversine_km(base_lat, base_lon, s["lat"], s["lon"])
        s2 = s.copy()
        s2["dist_km"] = round(d, 2)
        out.append(s2)
    return sorted(out, key=lambda x: x["dist_km"])

# ---------------------------
# UI Sidebar
# ---------------------------
st.sidebar.title("Settings")
CURRENT_LAT = st.sidebar.number_input("Current latitude", value=12.5942, format="%.6f")
CURRENT_LON = st.sidebar.number_input("Current longitude", value=80.2354, format="%.6f")
st.sidebar.markdown("---")
OSRM_URL = st.sidebar.text_input("OSRM URL", value="http://127.0.0.1:5000")

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Speech", "Emotion", "Suggestions", "Navigation"])

# ---------------------------
# TAB 1 — Speech
# ---------------------------
with tab1:
    st.header("1 — Speech (ASR & Translation)")
    if indian_langs:
        langs = list(indian_langs.keys())
        selected_src_lang = st.selectbox(
            "Input language code",
            options=langs,
            index=0,
            format_func=lambda x: f"{x} — {indian_langs[x]}"
        )
    else:
        selected_src_lang = st.text_input("Source language code:", value="tam")

    if st.button("🎙 Capture Speech"):
        with st.spinner("Running your speech pipeline..."):
            try:
                if process_input:
                    pipeline_out = process_input(selected_src_lang)
                elif process_audio:
                    aud = record_audio() if record_audio else None
                    pipeline_out = process_audio(aud, src_lang=selected_src_lang)
                else:
                    pipeline_out = {"error": "No speech pipeline found."}
            except Exception as e:
                pipeline_out = {"error": str(e)}
                st.error(f"Pipeline error: {e}")

            st.session_state["pipeline_output"] = pipeline_out

    po = st.session_state.get("pipeline_output")
    if po:
        st.subheader("Pipeline output")
        st.json(po)
    else:
        st.info("No pipeline output yet.")

# ---------------------------
# TAB 2 — Emotion
# ---------------------------
with tab2:
    st.header("2 — Emotion Detection")

    pipeline_output = st.session_state.get("pipeline_output") or {}
    default_text = pipeline_output.get("english_text") or pipeline_output.get("recognized_text") or ""

    text = st.text_area("Text to analyze:", value=default_text)

    if st.button("Detect Emotion"):
        if not text.strip():
            st.warning("Enter some text.")
        elif emo_predict:
            try:
                res = emo_predict(text)
                if isinstance(res, dict):
                    st.session_state["detected_emotion"] = res
                    st.success(f"Emotion: {res['emotion']} ({res['confidence']:.2f})")
                else:
                    st.session_state["detected_emotion"] = {"emotion": str(res), "confidence": 1.0}
                    st.success(f"Emotion: {res}")
            except Exception as e:
                st.error(f"Emotion model error: {e}")

    if st.session_state.get("detected_emotion"):
        st.subheader("Last detected emotion")
        st.json(st.session_state["detected_emotion"])

# ---------------------------
# TAB 3 — Suggestions (MAP FIXED!)
# ---------------------------
with tab3:
    st.header("3 — Suggestions")

    detected = st.session_state.get("detected_emotion")
    if not detected:
        st.info("Run Speech → Emotion first.")
    else:
        mood = detected.get("emotion", "").lower()
        st.write(f"Detected mood: **{mood}**")

        candidates = SUGGESTION_DB.get(mood, SUGGESTION_DB["default"])
        sorted_places = sort_and_attach_distance(candidates, CURRENT_LAT, CURRENT_LON)

        for i, c in enumerate(sorted_places):
            st.write(f"**{c['name']}** — {c['dist_km']} km")
            if st.button(f"Navigate to {c['name']}", key=f"nav_btn_{i}"):
                if not is_osrm_running(OSRM_URL):
                    st.error("OSRM not running!")
                else:
                    with st.spinner("Fetching route..."):
                        try:
                            resp = osrm_route(OSRM_URL, (CURRENT_LON, CURRENT_LAT), (c["lon"], c["lat"]))
                        except Exception as e:
                            resp = None
                            st.error(f"OSRM error: {e}")

                        if resp:
                            st.session_state["last_route_resp"] = resp
                            st.session_state["last_route_name"] = c["name"]
                            st.success("Route loaded!")

        # --- Persistent MAP DISPLAY ---
        if "last_route_resp" in st.session_state:
            resp = st.session_state["last_route_resp"]
            name = st.session_state.get("last_route_name", "Location")

            st.subheader(f"Route to {name}")
            try:
                if pretty_route_summary:
                    st.write(pretty_route_summary(resp))
            except:
                pass

            try:
                m = route_to_folium(resp)
                st_folium(m, width=900, height=500)
            except Exception as e:
                st.error(f"Map render failed: {e}")

# ---------------------------
# TAB 4 — Manual Navigation
# ---------------------------
with tab4:
    st.header("4 — Navigation (manual)")

    running = is_osrm_running(OSRM_URL)
    if running:
        st.success("OSRM backend is running.")
    else:
        st.error("OSRM backend not reachable.")

    c1, c2 = st.columns(2)
    with c1:
        src_lat = st.number_input("Source lat", value=CURRENT_LAT)
        src_lon = st.number_input("Source lon", value=CURRENT_LON)
    with c2:
        dst_lat = st.number_input("Destination lat", value=13.0500)
        dst_lon = st.number_input("Destination lon", value=80.2820)

    if st.button("Get Route (manual)"):
        if not running:
            st.error("OSRM offline.")
        else:
            with st.spinner("Fetching route..."):
                try:
                    resp = osrm_route(OSRM_URL, (src_lon, src_lat), (dst_lon, dst_lat))
                except Exception as e:
                    resp = None
                    st.error(f"OSRM error: {e}")

                if resp:
                    try:
                        st.write(pretty_route_summary(resp))
                    except:
                        pass

                    try:
                        m = route_to_folium(resp)
                        st_folium(m, width=900, height=500)
                    except Exception as e:
                        st.error(f"Map error: {e}")

# Footer
st.markdown("---")
st.caption("Suggestions are realistic Chennai places. Map persistence enabled. The app calls your modules exactly as-is.")
