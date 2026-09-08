import datetime
import warnings
import numpy as np
import pandas as pd
import pulp as pl
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="LogiAgent: Autonomous Fleet Optimization Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- High-Contrast Dark Theme CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Title Header ---
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #161B22 0%, #21262D 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #58A6FF; margin-bottom: 20px;">
        <h1 style="color: #58A6FF; margin:0; font-size: 28px;">LogiAgent: Autonomous Fleet Optimization Dashboard</h1>
        <p style="color: #8B949E; margin:5px 0 0 0; font-size: 14px;">Decision Support System untuk Outbound Logistics (Senin - Sabtu | Libur Minggu)</p>
    </div>
""",
    unsafe_allow_html=True,
)


# --- Data Generation Engine ---
@st.cache_data
def load_historical_data():
    np.random.seed(42)
    date_rng = pd.date_range(start="2026-05-01", end="2026-09-07", freq="B")
    base_trend = np.linspace(7000, 9500, len(date_rng))
    seasonality = 1500 * np.sin(np.linspace(0, 3 * np.pi, len(date_rng)))
    noise = np.random.normal(0, 400, len(date_rng))
    demand = base_trend + seasonality + noise

    df = pd.DataFrame({"Demand": np.clip(demand, 4000, 14000)}, index=date_rng)
    return df


df_history = load_historical_data()

# --- Sidebar Configuration ---
st.sidebar.markdown("## ⚙️ Parameter Konfigurasi")

area_pilihan = st.sidebar.selectbox(
    "Pilih Area / Rute Pengiriman:",
    [
        "Total Semua Area",
        "Rute Jakarta - Bandung",
        "Rute Surabaya - Malang",
        "Rute Medan - Pekanbaru",
    ],
)

z_score_map = {
    "80% (Z=0.84)": 0.84,
    "90% (Z=1.28)": 1.28,
    "95% (Z=1.65)": 1.65,
    "99% (Z=2.33)": 2.33,
}
selected_sla = st.sidebar.selectbox(
    "Safety Capacity Buffer (Z-Score):",
    options=list(z_score_map.keys()),
    index=1,
)
z_val = z_score_map[selected_sla]

horizon_prediksi = st.sidebar.slider(
    "Horizon Prediksi (Hari Kerja):", min_value=1, max_value=30, value=6
)

st.sidebar.markdown("---")
st.sidebar.markdown("📅 **Simulasi Target Tanggal Muat**")
today_date = datetime.date(2026, 9, 8)

future_dates = pd.date_range(
    start=today_date, periods=horizon_prediksi + 5, freq="B"
)

selected_target_date = st.sidebar.selectbox(
    "Pilih Tanggal Pengiriman:",
    options=future_dates,
    format_func=lambda x: x.strftime("%A, %d %b %Y")
    + (
        " (Hari Ini)"
        if x.date() == today_date
        else (" (Besok)" if x.date() == today_date + datetime.timedelta(days=1) else "")
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<div style='background-color: #1f242d; padding: 10px; border-radius: 5px; font-size: 12px;'>
    <b>💡 Operasional Outbound</b><br>
    (Senin–Sabtu): Hari Minggu libur total dari jadwal pengiriman & alokasi truk.
</div>
""",
    unsafe_allow_html=True,
)

# --- Holt-Winters Forecasting Model ---
model = ExponentialSmoothing(
    df_history["Demand"],
    trend="add",
    seasonal="add",
    seasonal_periods=5,
    initialization_method="estimated",
).fit()

forecast_full = model.forecast(steps=30)
forecast_index = pd.date_range(
    start=df_history.index[-1] + pd.Timedelta(days=1), periods=30, freq="B"
)
forecast_series = pd.Series(forecast_full.values, index=forecast_index)

fitted_values = model.fittedvalues
rmse = np.sqrt(np.mean((df_history["Demand"] - fitted_values) ** 2))
safety_buffer_val = z_val * rmse

if selected_target_date in forecast_series.index:
    base_pred = forecast_series.loc[selected_target_date]
else:
    base_pred = forecast_series.iloc[0]

target_demand = base_pred + safety_buffer_val

# --- Integer Linear Programming (ILP) Fleet Optimization via PuLP ---
fleet_types = {
    "CDD 7 Ton": {"cap": 1500, "cost": 1800000, "max_unit": 8},
    "FUSO 10 Ton": {"cap": 2000, "cost": 2400000, "max_unit": 6},
    "FUSO 17 Ton": {"cap": 3800, "cost": 4800000, "max_unit": 4},
}

prob = pl.LpProblem("Fleet_Allocation_Optimization", pl.LpMinimize)

x_vars = {
    truck: pl.LpVariable(
        f"truck_{truck}", lowBound=0, upBound=data["max_unit"], cat="Integer"
    )
    for truck, data in fleet_types.items()
}

prob += pl.lpSum(
    x_vars[truck] * fleet_types[truck]["cost"] for truck in fleet_types
)

prob += (
    pl.lpSum(
        x_vars[truck] * fleet_types[truck]["cap"] for truck in fleet_types
    )
    >= target_demand
)

prob.solve(pl.PULP_CBC_CMD(msg=False))

allocated_fleet = {truck: int(x_vars[truck].varValue) for truck in fleet_types}
total_capacity = sum(
    allocated_fleet[truck] * fleet_types[truck]["cap"] for truck in fleet_types
)
total_cost = sum(
    allocated_fleet[truck] * fleet_types[truck]["cost"] for truck in fleet_types
)
utilization_rate = (
    (target_demand / total_capacity) * 100 if total_capacity > 0 else 0
)

# --- Top KPI Metrics Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <p style="color: #8B949E; margin:0; font-size:13px;">PREDIKSI DEMAND ({selected_target_date.strftime('%d %b')})</p>
            <h2 style="color: #58A6FF; margin:5px 0 0 0; font-size:24px;">{base_pred:,.0f} Ctn</h2>
            <span style="color: #3fb950; font-size:11px;">+ Tren & Musiman Terkini</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <p style="color: #8B949E; margin:0; font-size:13px;">SAFETY CAPACITY BUFFER</p>
            <h2 style="color: #D29922; margin:5px 0 0 0; font-size:24px;">+{safety_buffer_val:,.0f} Ctn</h2>
            <span style="color: #8B949E; font-size:11px;">SLA {selected_sla}</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    total_trucks_assigned = sum(allocated_fleet.values())
    st.markdown(
        f"""
        <div class="metric-card">
            <p style="color: #8B949E; margin:0; font-size:13px;">REKOMENDASI ARMADA</p>
            <h2 style="color: #3FB950; margin:5px 0 0 0; font-size:24px;">{total_trucks_assigned} Unit</h2>
            <span style="color: #8B949E; font-size:11px;">Cap: {total_capacity:,.0f} Ctn</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <p style="color: #8B949E; margin:0; font-size:13px;">ESTIMASI TOTAL BIAYA</p>
            <h2 style="color: #F85149; margin:5px 0 0 0; font-size:22px;">Rp {total_cost:,.0f}</h2>
            <span style="color: #3fb950; font-size:11px;">Optimum Cost (ILP)</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Layout Utama: Grafik Tren & Alokasi Truk ---
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.markdown("### 📈 Demand Trend & Forecast (Senin - Sabtu)")

    df_plot_hist = df_history.tail(45)
    df_plot_fc = forecast_series.head(horizon_prediksi)

    chart_data = pd.DataFrame(
        {
            "Histori Demand": df_plot_hist["Demand"],
            "Holt-Winters Forecast": pd.Series(dtype=float),
        }
    )
    for idx, val in df_plot_fc.items():
        chart_data.loc[idx, "Holt-Winters Forecast"] = val

    st.line_chart(
        chart_data,
        color=["#58A6FF", "#3FB950"],
        height=320,
    )
    st.caption(
        "*Garis biru menunjukkan data historis harian, garis hijau menyoroti proyeksi peramalan ke depan sesuai horizon terpilih.*"
    )

with col_right:
    st.markdown("### 🚚 Optimized Fleet Allocation")
    st.markdown(
        f"<span style='font-size: 12px; color: #8B949E;'>Target Tanggal: <b>{selected_target_date.strftime('%A, %d %B %Y')}</b></span>",
        unsafe_allow_html=True,
    )

    for truck, count in allocated_fleet.items():
        cap_single = fleet_types[truck]["cap"]
        st.markdown(
            f"""
            <div style="background-color: #161B22; border: 1px solid #30363D; padding: 10px 15px; border-radius: 6px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #C9D1D9;">{truck}</strong><br>
                    <span style="font-size: 11px; color: #8B949E;">Cap: {cap_single:,} Karton / unit</span>
                </div>
                <div style="font-size: 18px; font-weight: bold; color: #58A6FF;">
                    {count} Unit
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="background-color: #111418; padding: 12px; border-radius: 6px; border: 1px dashed #30363D; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px;">
                <span style="color: #8B949E;">Total Kapasitas Muat:</span>
                <strong style="color: #C9D1D9;">{total_capacity:,.0f} Karton</strong>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-top: 5px;">
                <span style="color: #8B949E;">Utilisasi Kapasitas Muatan:</span>
                <strong style="color: {'#3FB950' if utilization_rate <= 100 else '#F85149'};">{utilization_rate:.1f}%</strong>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    manifest_df = pd.DataFrame(
        [
            {
                "Tanggal Muat": selected_target_date.strftime("%Y-%m-%d"),
                "Jenis Truk": k,
                "Jumlah Unit": v,
                "Kapasitas Satuan": fleet_types[k]["cap"],
                "Total Biaya (IDR)": v * fleet_types[k]["cost"],
            }
            for k, v in allocated_fleet.items()
        ]
    )
    csv_data = manifest_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Pre-booking Manifest (CSV)",
        data=csv_data,
        file_name=f"logiagent_manifest_{selected_target_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )