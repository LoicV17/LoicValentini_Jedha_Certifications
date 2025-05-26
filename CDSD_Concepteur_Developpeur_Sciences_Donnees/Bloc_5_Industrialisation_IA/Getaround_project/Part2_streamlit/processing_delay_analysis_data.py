import pandas as pd

# Reload the Excel file
excel_path = "../src/get_around_delay_analysis.xlsx"
df = pd.read_excel(excel_path, sheet_name="rentals_data")

# Rename columns for consistency with previous work
df = df.rename(columns={
    "checkin_type": "type",
    "delay_at_checkout_in_minutes": "delay",
    "previous_ended_rental_id": "prev_id",
    "time_delta_with_previous_rental_in_minutes": "time_delta"
})

# Save processed version as CSV for use in Streamlit
csv_path = "processed_delay_getaround_data.csv"
df.to_csv(csv_path, index=False)

csv_path
