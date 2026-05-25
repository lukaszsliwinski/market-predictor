# Temporary basic GUI made with Streamlit

import streamlit as st
from utils.predict import predict
from datetime import datetime
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=1000, key="ny_clock")
now_ny = datetime.now(ZoneInfo("America/New_York"))

st.markdown(
  f"""
    <div style="text-align:center; margin-top:20px;">
      <span>New York:</span><br>
      <span style="font-size:18px; font-weight:bold; line-height:1.5;">
        {now_ny.strftime('%Y-%m-%d')} | {now_ny.strftime('%H:%M:%S')}
      </span>
    </div>
  """,
  unsafe_allow_html=True
)

prediction = predict()

recommendation = "BUY" if (
  ((prediction["Day_dir_till_hour"].item() == 1.0) & (prediction["Pred"].item() == 1)) |
  ((prediction["Day_dir_till_hour"].item() == -1.0) & (prediction["Pred"].item() == 0))
) else "SELL"

st.markdown(
  f"""
    <div style="text-align:center; margin-top:20px;">
      <span>Recommendation for:</span><br>
      <span style="font-size:18px; font-weight:bold; line-height:1.5;">
        {prediction['Date_NY'].item()} | {prediction['Hour_NY'].item()}:00
      </span>
    </div>
  """,
  unsafe_allow_html=True
)

st.markdown(
  f"""
    <div style="display:flex; justify-content:center;">
      <div style="
        font-size:28px;
        font-weight:bold;
        color:white;
        background-color:{'green' if recommendation == 'BUY' else 'red'};
        padding: 10px 32px;
        margin-top: 10px;
        border-radius:16px;
        width:fit-content;
      ">
        {recommendation}
      </div>
    </div>
  """,
  unsafe_allow_html=True
)