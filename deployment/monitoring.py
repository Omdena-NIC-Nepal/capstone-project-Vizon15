import streamlit as st
import psutil
import datetime

def show_dashboard():
    st.write("### System Resource Usage")
    st.metric("CPU Usage (%)", f"{psutil.cpu_percent()}%")
    st.metric("Memory Usage (%)", f"{psutil.virtual_memory().percent}%")
    st.metric("Disk Usage (%)", f"{psutil.disk_usage('/').percent}%")
    st.write(f"Checked at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Optionally, you can add more detailed charts/logs here.