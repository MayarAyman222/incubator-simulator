# streamlit_incubator_simulator.py
# Incubator Simulator (Streamlit)
import streamlit as st
import numpy as np
import pandas as pd
import time
import math
import random
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

st.set_page_config(page_title="Incubator Simulator", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Utilities
# ---------------------------
def clamp(x, a, b):
    return max(a, min(b, x))

def percent_in_range(value, vmin, vmax):
    if vmax == vmin:
        return 100 if value >= vmax else 0
    p = (value - vmin) / (vmax - vmin) * 100.0
    return clamp(p, 0.0, 100.0)

# ---------------------------
# Session state defaults
# ---------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "alarm_paused" not in st.session_state:
    st.session_state.alarm_paused = False
if "log_rows" not in st.session_state:
    st.session_state.log_rows = []
if "sim_t" not in st.session_state:
    st.session_state.sim_t = 0.0
if "temp_buffer" not in st.session_state:
    st.session_state.temp_buffer = deque([37.0]*300, maxlen=300)
if "humid_buffer" not in st.session_state:
    st.session_state.humid_buffer = deque([60.0]*300, maxlen=300)
if "o2_buffer" not in st.session_state:
    st.session_state.o2_buffer = deque([21.0]*300, maxlen=300)
if "co2_buffer" not in st.session_state:
    st.session_state.co2_buffer = deque([0.04]*300, maxlen=300)

# ---------------------------
# Sidebar: thresholds & controls
# ---------------------------
st.sidebar.title("Incubator Controls & Thresholds")

col_ctrl = st.sidebar.container()
with col_ctrl:
    start_btn = st.button("Start", key="start")
    stop_btn = st.button("Stop", key="stop")
    pause_alarm_btn = st.button("Pause/Resume Alarm", key="pause_alarm")
    save_log_btn = st.button("Save Log CSV", key="save_log")

st.sidebar.markdown("---")
st.sidebar.subheader("Thresholds (safe range)")

default_thresholds = {
    'TEMP_min': 36.0, 'TEMP_max': 38.0,
    'HUMID_min': 50, 'HUMID_max': 70,
    'O2_min': 19, 'O2_max': 25,
    'CO2_min': 0.03, 'CO2_max': 0.06,
    'FAN_min': 0, 'FAN_max': 100
}

if "thresholds" not in st.session_state:
    st.session_state.thresholds = default_thresholds.copy()

thr = st.session_state.thresholds
thr['TEMP_min'] = st.sidebar.number_input("Temp min (°C)", value=float(thr['TEMP_min']), step=0.1, format="%.1f")
thr['TEMP_max'] = st.sidebar.number_input("Temp max (°C)", value=float(thr['TEMP_max']), step=0.1, format="%.1f")
thr['HUMID_min'] = st.sidebar.number_input("Humidity min (%)", value=float(thr['HUMID_min']), step=1.0)
thr['HUMID_max'] = st.sidebar.number_input("Humidity max (%)", value=float(thr['HUMID_max']), step=1.0)
thr['O2_min'] = st.sidebar.number_input("O₂ min (%)", value=float(thr['O2_min']), step=0.1)
thr['O2_max'] = st.sidebar.number_input("O₂ max (%)", value=float(thr['O2_max']), step=0.1)
thr['CO2_min'] = st.sidebar.number_input("CO₂ min (%)", value=float(thr['CO2_min']), step=0.01)
thr['CO2_max'] = st.sidebar.number_input("CO₂ max (%)", value=float(thr['CO2_max']), step=0.01)
thr['FAN_min'] = st.sidebar.number_input("Fan speed min", value=float(thr['FAN_min']), step=1.0)
thr['FAN_max'] = st.sidebar.number_input("Fan speed max", value=float(thr['FAN_max']), step=1.0)

st.sidebar.markdown("---")
st.sidebar.write("Simulation speed & options")
sim_speed = st.sidebar.slider("Update interval (ms)", 50, 1000, 200, step=50)
simulate_noise = st.sidebar.checkbox("Simulate noise/jitter", value=True)

# manual overrides
st.sidebar.markdown("---")
st.sidebar.subheader("Manual overrides (optional)")
manual_temp = st.sidebar.number_input("Manual Temp (°C, -1 to disable)", value=-1.0, step=0.1)
manual_humid = st.sidebar.number_input("Manual Humidity (% -1 to disable)", value=-1.0, step=1.0)

# ---------------------------
# Top: Title + status
# ---------------------------
st.title("Incubator Simulator")
cols = st.columns([1,2,1])
with cols[0]:
    st.metric("Status", "Running" if st.session_state.running else "Stopped")
with cols[1]:
    last_saved = st.session_state.log_rows[-1][0] if st.session_state.log_rows else "no data"
    st.write("**Last update:** " + (last_saved if isinstance(last_saved, str) else str(last_saved)))
with cols[2]:
    st.write("Alarm: ")
    if st.session_state.alarm_paused:
        st.success("Paused")
    else:
        st.write("Active")

# ---------------------------
# Layout: left plots, right vitals
# ---------------------------
left_col, right_col = st.columns([2,1])
temp_container = left_col.container()
humid_container = left_col.container()
o2_container = left_col.container()
co2_container = left_col.container()
vitals_container = right_col.container()
controls_container = right_col.container()

# ---------------------------
# Controls behavior
# ---------------------------
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
if pause_alarm_btn:
    st.session_state.alarm_paused = not st.session_state.alarm_paused
if save_log_btn:
    if st.session_state.log_rows:
        df = pd.DataFrame(st.session_state.log_rows,
                          columns=["timestamp","TEMP","HUMID","O2","CO2","FAN","alarm"])
        filename = f"incubator_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        st.success(f"Saved log to {filename}")
    else:
        st.info("No data to save yet.")

# ---------------------------
# Simulation step
# ---------------------------
def simulate_step(dt=0.2):
    st.session_state.sim_t += dt
    t = st.session_state.sim_t

    temp = 37.0 + 0.3*math.sin(0.05*t) + (random.uniform(-0.2,0.2) if simulate_noise else 0)
    humid = 60 + 5*math.sin(0.03*t) + (random.uniform(-2,2) if simulate_noise else 0)
    o2 = 21 + 0.5*math.sin(0.02*t) + (random.uniform(-0.3,0.3) if simulate_noise else 0)
    co2 = 0.04 + 0.01*math.sin(0.01*t) + (random.uniform(-0.005,0.005) if simulate_noise else 0)
    fan = 50 + 10*math.sin(0.02*t) + (random.uniform(-5,5) if simulate_noise else 0)

    # manual overrides
    if manual_temp >= 0:
        temp = manual_temp
    if manual_humid >= 0:
        humid = manual_humid

    st.session_state.temp_buffer.append(temp)
    st.session_state.humid_buffer.append(humid)
    st.session_state.o2_buffer.append(o2)
    st.session_state.co2_buffer.append(co2)

    rec = {
        "timestamp": datetime.now().isoformat(),
        "TEMP": round(temp,2),
        "HUMID": round(humid,1),
        "O2": round(o2,2),
        "CO2": round(co2,3),
        "FAN": int(round(fan))
    }

    alarms = []
    th = st.session_state.thresholds
    if rec['TEMP']<th['TEMP_min'] or rec['TEMP']>th['TEMP_max']:
        alarms.append(f"TEMP {rec['TEMP']}")
    if rec['HUMID']<th['HUMID_min'] or rec['HUMID']>th['HUMID_max']:
        alarms.append(f"HUMID {rec['HUMID']}")
    if rec['O2']<th['O2_min'] or rec['O2']>th['O2_max']:
        alarms.append(f"O2 {rec['O2']}")
    if rec['CO2']<th['CO2_min'] or rec['CO2']>th['CO2_max']:
        alarms.append(f"CO2 {rec['CO2']}")
    if rec['FAN']<th['FAN_min'] or rec['FAN']>th['FAN_max']:
        alarms.append(f"FAN {rec['FAN']}")

    rec['alarm'] = "; ".join(alarms) if alarms else ""
    st.session_state.log_rows.append([rec['timestamp'], rec['TEMP'], rec['HUMID'], rec['O2'], rec['CO2'], rec['FAN'], rec['alarm']])

    return rec

# ---------------------------
# Main loop
# ---------------------------
if st.session_state.running:
    rec = simulate_step(dt=sim_speed/1000.0)

    # Left plots
    def plot_trend(buffer, title, ylim):
        fig, ax = plt.subplots(figsize=(9,2))
        ax.plot(list(buffer), linewidth=1.2)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_facecolor("#0f1113")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with temp_container:
        st.subheader("Temperature (°C)")
        plot_trend(st.session_state.temp_buffer, "Temp", (30,40))
    with humid_container:
        st.subheader("Humidity (%)")
        plot_trend(st.session_state.humid_buffer, "Humidity", (40,80))
    with o2_container:
        st.subheader("O₂ (%)")
        plot_trend(st.session_state.o2_buffer, "O2", (15,25))
    with co2_container:
        st.subheader("CO₂ (%)")
        plot_trend(st.session_state.co2_buffer, "CO2", (0,0.1))

    # Right vitals
    with vitals_container:
        st.subheader("Parameters")
        def show_metric(name, value, vmin, vmax, unit=""):
            pct = percent_in_range(value, vmin, vmax)
            if value < vmin or value > vmax:
                st.markdown(f"**{name}**: <span style='color:red'>{value}{unit}</span>", unsafe_allow_html=True)
                st.progress(int(pct))
            else:
                st.markdown(f"**{name}**: <span style='color:#00b894'>{value}{unit}</span>", unsafe_allow_html=True)
                st.progress(int(pct))

        show_metric("TEMP (°C)", rec["TEMP"], thr['TEMP_min'], thr['TEMP_max'], " °C")
        show_metric("HUMID (%)", rec["HUMID"], thr['HUMID_min'], thr['HUMID_max'], " %")
        show_metric("O₂ (%)", rec["O2"], thr['O2_min'], thr['O2_max'], " %")
        show_metric("CO₂ (%)", rec["CO2"], thr['CO2_min'], thr['CO2_max'], " %")
        show_metric("FAN (%)", rec["FAN"], thr['FAN_min'], thr['FAN_max'], " %")

        st.markdown("### Alarm status")
        if rec['alarm'] and not st.session_state.alarm_paused:
            st.error(rec['alarm'])
        elif rec['alarm'] and st.session_state.alarm_paused:
            st.warning("Alarm paused: " + rec['alarm'])
        else:
            st.success("All parameters normal")

    # Controls
    with controls_container:
        st.subheader("Log & Export")
        st.write(f"Total log rows: {len(st.session_state.log_rows)}")
        if st.button("Clear Log"):
            st.session_state.log_rows = []
            st.success("Log cleared")

    time.sleep(sim_speed/1000.0)
    st.rerun()
