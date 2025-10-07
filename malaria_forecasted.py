import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re
import io

st.set_page_config(page_title="Malaria Forecasting App", layout="wide")

st.title("🦟 Malaria Cases Forecasting App (SARIMA Model)")
st.write("Forecast malaria cases for the next 8 months for any selected location.")

# --- Upload dataset ---
uploaded_file = st.file_uploader("📂 Upload Malaria Dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv('malaria cases.csv')
    df.columns = df.columns.str.strip()

    required_cols = ['periodname', 'organisationunitname', 'Total Malaria Cases']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Your dataset must include the columns: {', '.join(required_cols)}")
    else:
        # --- Extract and clean dates ---
        def extract_start_date(text):
            match = re.search(r'(\d{4}-\d{2}-\d{2})', str(text))
            return match.group(1) if match else None
        
        df['start_date'] = df['periodname'].apply(extract_start_date)
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df = df.dropna(subset=['start_date'])

        # Convert malaria cases to numeric
        df['Total Malaria Cases'] = pd.to_numeric(df['Total Malaria Cases'], errors='coerce').fillna(0)

        # --- Select location ---
        locations = sorted(df['organisationunitname'].dropna().unique())
        selected_location = st.selectbox("🏥 Select Location", locations)

        # --- Filter by location ---
        location_data = df[df['organisationunitname'] == selected_location]

        if location_data.empty:
            st.warning(f"No data found for {selected_location}. Try another location.")
        else:
            # Aggregate by date
            location_data = (
                location_data.groupby('start_date')['Total Malaria Cases']
                .sum()
                .reset_index()
                .sort_values('start_date')
            )
            location_data = location_data.set_index('start_date')

            if len(location_data) < 3:
                st.warning(f"Not enough data points to forecast for {selected_location}.")
            else:
                # --- Plot historical data ---
                st.subheader(f"📊 Historical Malaria Cases in {selected_location}")
                st.line_chart(location_data['Total Malaria Cases'])

                # --- SARIMA Forecast ---
                try:
                    model = sm.tsa.statespace.SARIMAX(
                        location_data['Total Malaria Cases'],
                        order=(1,1,1),
                        seasonal_order=(1,1,1,12),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    results = model.fit(disp=False)

                    # --- Forecast next 8 months ---
                    forecast_steps = 8
                    forecast = results.get_forecast(steps=forecast_steps)
                    forecast_index = pd.date_range(location_data.index[-1], periods=forecast_steps+1, freq='M')[1:]

                    forecast_df = pd.DataFrame({
                        'Date': forecast_index,
                        'Forecasted Cases': forecast.predicted_mean.values,
                        'Lower CI': forecast.conf_int().iloc[:, 0].values,
                        'Upper CI': forecast.conf_int().iloc[:, 1].values
                    })

                    # --- Plot forecast ---
                    st.subheader(f"📈 8-Month Forecast for {selected_location}")
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.plot(location_data.index, location_data['Total Malaria Cases'], label="Historical Cases", color='blue')
                    ax.plot(forecast_df['Date'], forecast_df['Forecasted Cases'], label="Forecasted Cases", color='orange')
                    ax.fill_between(forecast_df['Date'], forecast_df['Lower CI'], forecast_df['Upper CI'], color='orange', alpha=0.2)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Malaria Cases")
                    ax.legend()
                    st.pyplot(fig)

                    st.success(f"Forecast completed for {selected_location} ✅")

                    # --- Show forecast table ---
                    st.dataframe(forecast_df)

                    # --- CSV download ---
                    csv_buffer = io.StringIO()
                    forecast_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Forecast as CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"malaria_forecast_{selected_location}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error fitting SARIMA model: {e}")
else:
    st.info("Please upload your malaria dataset (CSV format) to begin.")
