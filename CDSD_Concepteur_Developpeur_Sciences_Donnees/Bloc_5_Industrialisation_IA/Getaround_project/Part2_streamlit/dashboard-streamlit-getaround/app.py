
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset

import streamlit as st
import pandas as pd

# Load dataset
def load_data():
    try:
        df = pd.read_csv("processed_delay_getaround_data.csv")
        st.success("✅ CSV loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        st.stop()

    df = df.rename(columns={
        "checkin_type": "type",
        "delay_at_checkout_in_minutes": "delay",
        "previous_ended_rental_id": "prev_id",
        "time_delta_with_previous_rental_in_minutes": "time_delta"
    })
    return df

df = load_data()


# Sidebar with logo and author note
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/8e/Getaround_%28Europe%29.png", width=200)
st.sidebar.markdown("---")
st.sidebar.markdown("Made by **Loic Valentini**  \nMay 2025")

# Main title with color accents
st.markdown("<h1 style='color:#aa1ba3;'>🚗 Getaround Delay Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Section 1: Basic Metrics
st.markdown("<h2 style='color:#aa1ba3;'>📊 Basic Rental Metrics</h2>", unsafe_allow_html=True)

# Filter by check-in type
checkin_filter = st.radio("Select check-in type:", ["both", "connect", "mobile"])
if checkin_filter != "both":
    df_filtered = df[df["type"] == checkin_filter]
else:
    df_filtered = df.copy()

# Filter ended rentals for metrics
df_ended = df_filtered[df_filtered["state"] == "ended"]
df_positive_delay = df_ended[df_ended["delay"].notna() & (df_ended["delay"] > 0)]

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals", len(df_filtered))
col2.metric("Completed Rentals", len(df_ended))
col3.metric("Canceled Rentals", (df_filtered["state"] == "canceled").sum())

col4, col5, col6 = st.columns(3)
col4.metric("Rentals with Delay", len(df_positive_delay))
col5.metric("Avg Delay (min)", f"{df_positive_delay['delay'].mean():.1f}")
col6.metric("Median Delay (min)", f"{df_positive_delay['delay'].median():.0f}")

st.markdown("---")

# Section 2: Impact Simulation
st.markdown("<h2 style='color:#aa1ba3;'>📉 Buffer Threshold Impact Simulation</h2>", unsafe_allow_html=True)

threshold = st.slider("Choose buffer threshold (in minutes):", 0, 720, 60, step=10)

# Filter datasets for analysis
df_buffer = df_filtered[df_filtered["time_delta"].notna()]
df_critical = df_filtered.dropna(subset=["delay", "time_delta"])
df_critical = df_critical[df_critical["delay"] > df_critical["time_delta"]]
df_canceled = df_filtered[(df_filtered["state"] == "canceled") & (df_filtered["time_delta"].notna())]

# Compute metrics
total_rentals = len(df_filtered)
affected_pct = 100 * (df_buffer["time_delta"] < threshold).sum() / total_rentals

total_critical = len(df_critical)
solved_pct = 100 * (df_critical["time_delta"] < threshold).sum() / total_critical if total_critical else 0

total_canceled = len(df_canceled)
canceled_pct = 100 * (df_canceled["time_delta"] < threshold).sum() / total_canceled if total_canceled else 0

# Results text
st.markdown("---")
st.markdown(f"<h3 style='color:#aa1ba3;'>🎯 Results for a {threshold}-minute buffer</h3>", unsafe_allow_html=True)
st.write(f"- **{affected_pct:.1f}%** of all rentals would be affected (availability impact)")
st.write(f"- **{solved_pct:.1f}%** of critical overlap cases would be solved")
st.write(f"- **{canceled_pct:.1f}%** of cancellations happened with a shorter buffer (possible prevention)")

# Bar chart of the 3 rates
st.markdown("#### 📊 Visual Summary")

labels = ["% Rentals Affected", "% Critical Solved", "% Cancellations Concerned"]
values = [affected_pct, solved_pct, canceled_pct]
colors = ["#aa1ba3", "#cc66cc", "#e699e6"]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(labels, values, color=colors)
ax.set_ylim(0, 110)  # increase to leave space above bars
ax.set_ylabel("Percentage (%)")
ax.set_title(f"Impact of a {threshold}-min Buffer", color="#aa1ba3", pad=20)  # add padding below title

# Add percentage labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{height:.1f}%", ha="center", va="bottom", fontsize=10)

st.pyplot(fig)
