# Temporary basic GUI made with Streamlit

import streamlit as st
from utils.predict import predict
from utils.history import history
from datetime import datetime
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh
import exchange_calendars as ecals

nyse = ecals.get_calendar("XNYS")

st_autorefresh(interval=1000, key="ny_clock")
now_ny = datetime.now(ZoneInfo("America/New_York"))
current_hour = now_ny.hour

def is_market_open(date: datetime) -> bool:
  return nyse.is_session(date.date())

market_open = is_market_open(now_ny)

# UI clock
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

if not market_open:
  st.markdown(
    """
      <div style="text-align:center; margin-top:20px; color:white; background-color:#444; padding:10px; border-radius:10px;">
        Market is closed (NYSE) – no prediction available
      </div>
    """,
    unsafe_allow_html=True
  )
  st.stop()

if current_hour in [17, 18]:
  st.markdown(
    """
      <div style="text-align:center; margin-top:20px; color:white; background-color:#444; padding:10px; border-radius:10px;">
        Market is open – prediction disabled for 17:00 and 18:00 NY time
      </div>
    """,
    unsafe_allow_html=True
  )
  st.stop()

# Session state init
if "prediction" not in st.session_state:
  st.session_state.prediction = predict()
  st.session_state.last_hour = st.session_state.prediction["Hour_NY"].item()

if "yf_history" not in st.session_state:
  st.session_state.yf_history = history()

if "last_predict_time" not in st.session_state:
  st.session_state.last_predict_time = now_ny

# if the hour does NOT match → run predict at most every 10 seconds
pred_hour = int(st.session_state.prediction["Hour_NY"].item())
time_since_last_predict = (now_ny - st.session_state.last_predict_time).total_seconds()
should_refresh_prediction = (
  pred_hour != current_hour and time_since_last_predict >= 10
)

if should_refresh_prediction:
  st.session_state.prediction = predict()
  st.session_state.last_hour = st.session_state.prediction["Hour_NY"].item()
  st.session_state.last_predict_time = now_ny

  st.session_state.yf_history = history()

prediction = st.session_state.prediction
recommendation = "BUY" if (
  ((prediction["Day_dir_till_hour"].item() == 1.0) & (prediction["Pred"].item() == 1)) |
  ((prediction["Day_dir_till_hour"].item() == -1.0) & (prediction["Pred"].item() == 0))
) else "SELL"

# UI prediction
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
 

yf_history = st.session_state.yf_history[["Date_NY", "Hour_NY"]].iloc[::-1].reset_index(drop=True).head(24)
yf_history["Hour_NY"] = yf_history["Hour_NY"].astype(str) + ":00"

st.markdown("<div style='margin-top:50px;'></div>", unsafe_allow_html=True)
st.write("Last provided data points:")
st.dataframe(yf_history)
