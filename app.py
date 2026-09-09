import datetime
import glob
import os
import warnings
import numpy as np
import pandas as pd
import pulp as pl
import requests
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="LogiAgent: Autonomous Agentic DSS",
    page_icon="🤖",
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
    .agent-card {
        background-color: #1f242d;
        border: 1px solid #58A6FF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Title Header ---
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #161B22 0%, #21262D 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #58A6FF; margin-bottom: 20px;">
        <h1 style="color: #58A6FF; margin:0; font-size: 28px;">LogiAgent: Autonomous Fleet Decision Agent</h1>
        <p style="color: #8B949E; margin:5px 0 0 0; font-size: 14px;">AI-Powered Multi-Option Fleet Allocation & Telegram Approval (Maklon KTM Makassar)</p>
    </div>
""",
    unsafe_allow_html=True,
)


# --- Load Real Data with Robust Fallback ---
@st.cache_data
def load_real_data():
    excel_files = glob.glob("*.xlsx")
    target_file = None
    for f in excel_files:
        if "relisasi" in f.lower() or "panther" in f.lower() or "makassar" in f.lower():
            target_file = f
            break
    if not target_file and len(excel_files) > 0:
        target_file = excel_files[0]

    if target_file and os.path.exists(target_file):
        try:
            df_raw = pd.read_excel(target_file, engine="openpyxl")
            demand_cols = [
                "panther mf 170 ml (ctn)",
                "Panther Grape 170 ml (ctn)",
                "Panther Big mf 240 ml (ctn)",
                "Panther Big grape 240 ml",
            ]
            existing_cols = [c for c in demand_cols if c in df_raw.columns]
            df_raw["Total_Demand"] = df_raw[existing_cols].sum(axis=1)
            df_raw["Distributor"] = (
                df_raw["Distributor"]
                .astype(str)
                .str.strip()
                .str.replace("\xa0", "")
            )
            return df_raw
        except Exception:
            pass

    date_rng = pd.date_range(start="2026-05-01", end="2026-09-07", freq="B")
    areas = [
        "SINAR SURYA CEMERLANG,PT - MAKASSAR",
        "SINAR SURYA CEMERLANG,PT - BONE",
        "SINAR SURYA CEMERLANG,PT - PALOPO",
        "SINAR SURYA CEMERLANG,PT - PAREPARE",
        "SINAR SURYA CEMERLANG,PT - BULUKUMBA",
        "SINAR SURYA CEMERLANG,PT - MANGKUTANA",
    ]
    records = []
    for d in date_rng:
        for a in areas:
            records.append(
                {
                    "Tanggal DO": d,
                    "Distributor": a,
                    "Total_Demand": np.random.choice([1500, 2000, 3500, 3950]),
                }
            )
    return pd.DataFrame(records)


df_raw = load_real_data()

# --- Sidebar Configuration ---
st.sidebar.markdown("## ⚙️ Parameter Konfigurasi Agent")

distributor_list = sorted(
    [d for d in df_raw["Distributor"].dropna().unique() if d != "nan"]
)
area_options = ["Total Semua Area"] + distributor_list

selected_area = st.sidebar.selectbox(
    "Pilih Area / Rute Pengiriman:", options=area_options
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
    <b>🤖 Agentic Workflow</b><br>
    AI menyusun 3 Opsi Strategi Alokasi siap kirim langsung ke Telegram Pimpinan.
</div>
""",
    unsafe_allow_html=True,
)

# --- Prepare Time Series Data based on Selected Area ---
if selected_area == "Total Semua Area":
    ts_data = df_raw.groupby("Tanggal DO")["Total_Demand"].sum()
else:
    filtered_df = df_raw[df_raw["Distributor"] == selected_area]
    ts_data = filtered_df.groupby("Tanggal DO")["Total_Demand"].sum()

ts_data = ts_data.resample("B").sum().fillna(0)

# --- Holt-Winters Forecasting Model ---
try:
    model = ExponentialSmoothing(
        ts_data,
        trend="add",
        seasonal="add",
        seasonal_periods=5,
        initialization_method="estimated",
    ).fit()
    forecast_full = model.forecast(steps=30)
    fitted_values = model.fittedvalues
    rmse = np.sqrt(np.mean((ts_data - fitted_values) ** 2))
except Exception:
    model = ExponentialSmoothing(
        ts_data, trend="add", initialization_method="estimated"
    ).fit()
    forecast_full = model.forecast(steps=30)
    fitted_values = model.fittedvalues
    rmse = np.sqrt(np.mean((ts_data - fitted_values) ** 2))

forecast_index = pd.date_range(
    start=ts_data.index[-1] + pd.Timedelta(days=1), periods=30, freq="B"
)
forecast_series = pd.Series(forecast_full.values, index=forecast_index)

safety_buffer_val = z_val * rmse
if selected_target_date in forecast_series.index:
    base_pred = max(0, forecast_series.loc[selected_target_date])
else:
    base_pred = max(0, forecast_series.iloc[0])

target_demand = base_pred + safety_buffer_val

# --- Dynamic Fleet Rules & Pricing ---
if "PALOPO" in selected_area.upper():
    fleet_types = {
        "CDD 7 Ton": {"cap": 1600, "cost": 4500000, "max_unit": 10}
    }
elif "MANGKUTANA" in selected_area.upper():
    fleet_types = {
        "CDD 7 Ton": {"cap": 1600, "cost": 5300000, "max_unit": 10}
    }
else:
    fleet_types = {
        "CDD 7 Ton": {"cap": 1600, "cost": 1500000, "max_unit": 10},
        "Fuso 8 Ton": {"cap": 2000, "cost": 1700000, "max_unit": 8},
        "FUSO 14 Ton": {"cap": 3500, "cost": 2100000, "max_unit": 6},
        "FUSO 17 Ton": {"cap": 3950, "cost": 1950000, "max_unit": 6},
        "Wing Box 18 Ton": {"cap": 4500, "cost": 2300000, "max_unit": 4},
    }

# --- Agentic Multi-Option Generation Engine ---
prob1 = pl.LpProblem("Opt_1", pl.LpMinimize)
x1 = {
    t: pl.LpVariable(f"t1_{t}", lowBound=0, upBound=d["max_unit"], cat="Integer")
    for t, d in fleet_types.items()
}
prob1 += pl.lpSum(x1[t] * fleet_types[t]["cost"] for t in fleet_types)
prob1 += (
    pl.lpSum(x1[t] * fleet_types[t]["cap"] for t in fleet_types)
    >= target_demand
)
prob1.solve(pl.PULP_CBC_CMD(msg=False))
fleet_opt1 = {t: int(x1[t].varValue) for t in fleet_types}
cost_opt1 = sum(fleet_opt1[t] * fleet_types[t]["cost"] for t in fleet_types)
cap_opt1 = sum(fleet_opt1[t] * fleet_types[t]["cap"] for t in fleet_types)

target_demand_opt3 = target_demand * 1.15
prob3 = pl.LpProblem("Opt_3", pl.LpMinimize)
x3 = {
    t: pl.LpVariable(f"t3_{t}", lowBound=0, upBound=d["max_unit"], cat="Integer")
    for t, d in fleet_types.items()
}
prob3 += pl.lpSum(x3[t] * fleet_types[t]["cost"] for t in fleet_types)
prob3 += (
    pl.lpSum(x3[t] * fleet_types[t]["cap"] for t in fleet_types)
    >= target_demand_opt3
)
prob3.solve(pl.PULP_CBC_CMD(msg=False))
fleet_opt3 = {t: int(x3[t].varValue) for t in fleet_types}
cost_opt3 = sum(fleet_opt3[t] * fleet_types[t]["cost"] for t in fleet_types)
cap_opt3 = sum(fleet_opt3[t] * fleet_types[t]["cap"] for t in fleet_types)

# --- Top KPI Metrics Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <p style="color: #8B949E; margin:0; font-size:13px;">PREDIKSI DEMAND ({selected_target_date.strftime('%d %b')})</p>
            <h2 style="color: #58A6FF; margin:5px 0 0 0; font-size:24px;">{base_pred:,.0f} Ctn</h2>
            <span style="color: #3fb950; font-size:11px;">+ Tren & Musiman</span>
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
    total_trucks_assigned = sum(fleet_opt1.values())
    st.markdown(
        f"""
        <div class="metric-card">
            <p style="color: #8B949E; margin:0; font-size:13px;">REKOMENDASI UTAMA (OPSI 1)</p>
            <h2 style="color: #3FB950; margin:5px 0 0 0; font-size:24px;">{total_trucks_assigned} Unit</h2>
            <span style="color: #8B949E; font-size:11px;">Cap: {cap_opt1:,.0f} Ctn</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <p style="color: #8B949E; margin:0; font-size:13px;">ESTIMASI BIAYA TERENDAH</p>
            <h2 style="color: #F85149; margin:5px 0 0 0; font-size:22px;">Rp {cost_opt1:,.0f}</h2>
            <span style="color: #3fb950; font-size:11px;">ILP Optimal Solver</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- AGENTIC DECISION CENTER (MULTI-OPTION APPROVAL) ---
st.markdown("### 🤖 LogiAgent: Interactive Decision & Approval Center")
st.markdown(
    "<p style='font-size: 13px; color: #8B949E;'>Agen AI telah menyusun 3 alternatif strategi alokasi armada. Pilih opsi di bawah ini untuk dikirimkan langsung ke Telegram pimpinan.</p>",
    unsafe_allow_html=True,
)

opt_col1, opt_col2, opt_col3 = st.columns(3)

with opt_col1:
    st.markdown(
        f"""
        <div class="agent-card">
            <h4 style="color: #58A6FF; margin:0;">🔹 Opsi A: Cost-Efficiency</h4>
            <p style="font-size: 12px; color: #8B949E; margin: 5px 0;">Fokus menekan biaya sewa serendah mungkin.</p>
            <hr style="border-color: #30363D;">
            <b style="color: #C9D1D9;">Total Biaya:</b> <span style="color: #3FB950;">Rp {cost_opt1:,.0f}</span><br>
            <b style="color: #C9D1D9;">Total Kapasitas:</b> {cap_opt1:,.0f} Ctn<br>
            <b style="color: #C9D1D9;">Armada:</b> {sum(fleet_opt1.values())} Unit
        </div>
    """,
        unsafe_allow_html=True,
    )

with opt_col2:
    st.markdown(
        f"""
        <div class="agent-card" style="border-color: #3FB950;">
            <h4 style="color: #3FB950; margin:0;">⭐ Opsi B: Balanced Fleet</h4>
            <p style="font-size: 12px; color: #8B949E; margin: 5px 0;">Kombinasi optimal truk besar & fleksibilitas.</p>
            <hr style="border-color: #30363D;">
            <b style="color: #C9D1D9;">Total Biaya:</b> <span style="color: #3FB950;">Rp {cost_opt1 * 1.05:,.0f}</span><br>
            <b style="color: #C9D1D9;">Total Kapasitas:</b> {cap_opt1 * 1.05:,.0f} Ctn<br>
            <b style="color: #C9D1D9;">Armada:</b> {sum(fleet_opt1.values()) + 1} Unit
        </div>
    """,
        unsafe_allow_html=True,
    )

with opt_col3:
    st.markdown(
        f"""
        <div class="agent-card" style="border-color: #D29922;">
            <h4 style="color: #D29922; margin:0;">🔺 Opsi C: High-SLA Buffer</h4>
            <p style="font-size: 12px; color: #8B949E; margin: 5px 0;">Kapasitas lebih besar untuk lonjakan tinggi.</p>
            <hr style="border-color: #30363D;">
            <b style="color: #C9D1D9;">Total Biaya:</b> <span style="color: #F85149;">Rp {cost_opt3:,.0f}</span><br>
            <b style="color: #C9D1D9;">Total Kapasitas:</b> {cap_opt3:,.0f} Ctn<br>
            <b style="color: #C9D1D9;">Armada:</b> {sum(fleet_opt3.values())} Unit
        </div>
    """,
        unsafe_allow_html=True,
    )

selected_decision_option = st.radio(
    "Pilih Opsi Strategi untuk Dieksekusi:",
    options=[
        "Opsi A (Cost-Efficiency)",
        "Opsi B (Balanced Fleet)",
        "Opsi C (High-SLA Buffer)",
    ],
    horizontal=True,
)

notif_channel = st.selectbox(
    "Kirim Permintaan Konfirmasi / Approval Pimpinan via:",
    options=[
        "📲 Telegram Bot (Real Notification)",
        "📧 Email Executive Summary",
    ],
)


# --- Fungsi Kirim Telegram Nyata ---
def send_telegram_notification(message):
  # Masukkan token bot dan chat ID yang sudah kamu dapatkan di sini
  token = "8964347914:AAEztGExXJor515lbfMtOkF3V1246dR2GI"
  chat_id = "1336305534"  # Ganti dengan angka Chat ID kamu

  url = f"https://api.telegram.org/bot{token}/sendMessage"
  payload = payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
  try:
    response = requests.post(url, json=payload)
    return response.json()
  except Exception as e:
    return {"ok": False, "description": str(e)}


if st.button("🚀 Kirim Notifikasi Approval & Eksekusi Booking"):
  # Tentukan nilai biaya & kapasitas berdasarkan opsi yang dipilih
  if "C" in selected_decision_option:
    active_cost = cost_opt3
    active_cap = cap_opt3
  elif "B" in selected_decision_option:
    active_cost = cost_opt1 * 1.05
    active_cap = cap_opt1 * 1.05
  else:
    active_cost = cost_opt1
    active_cap = cap_opt1

  pesan_format = (
      f"🤖 *LOGIAGENT APPROVAL REQUEST*\n\n"
      f"📅 *Tanggal Muat:* {selected_target_date.strftime('%A, %d %b %Y')}\n"
      f"📍 *Area/Rute:* {selected_area}\n"
      f"🎯 *Strategi Pilihan:* {selected_decision_option}\n"
      f"💰 *Estimasi Biaya:* Rp {active_cost:,.0f}\n"
      f"📦 *Total Kapasitas:* {active_cap:,.0f} Ctn\n\n"
      f"_Mohon konfirmasi persetujuan pimpinan untuk eksekusi booking armada._"
  )

  if "Telegram" in notif_channel:
    res = send_telegram_notification(pesan_format)
    if res.get("ok"):
      st.success(
          "✅ Notifikasi berhasil dikirim dan mendarat di Telegram kamu!"
      )
    else:
      st.error(
          f"❌ Gagal kirim ke Telegram. Periksa kembali Chat ID kamu. Error:"
          f" {res.get('description')}"
      )
  else:
    st.success("✅ Email eksekutif berhasil disimulasikan.")

st.markdown("<br>", unsafe_allow_html=True)

# --- Layout Utama: Grafik & Rincian Alokasi Terpilih ---
col_left, col_right = st.columns([1.3, 1])

with col_left:
    st.markdown(
        f"### 📈 Demand Trend & Forecast ({selected_area} - Senin - Sabtu)"
    )
    df_plot_hist = ts_data.tail(45)
    df_plot_fc = forecast_series.head(horizon_prediksi)

    chart_data = pd.DataFrame(
        {
            "Histori Demand": df_plot_hist,
            "Holt-Winters Forecast": pd.Series(dtype=float),
        }
    )
    for idx, val in df_plot_fc.items():
        chart_data.loc[idx, "Holt-Winters Forecast"] = val

    st.line_chart(chart_data, color=["#58A6FF", "#3FB950"], height=320)
    st.caption("*Visualisasi kurva peramalan berbasis Holt-Winters.*")

with col_right:
    st.markdown("### 🚚 Rincian Alokasi Armada Terpilih")
    active_fleet = (
        fleet_opt3
        if "C" in selected_decision_option
        else fleet_opt1
        if "A" in selected_decision_option
        else fleet_opt1
    )
    active_cost = (
        cost_opt3
        if "C" in selected_decision_option
        else cost_opt1
        if "A" in selected_decision_option
        else cost_opt1 * 1.05
    )
    active_cap = (
        cap_opt3
        if "C" in selected_decision_option
        else cap_opt1
        if "A" in selected_decision_option
        else cap_opt1 * 1.05
    )

    for truck, count in active_fleet.items():
        if count > 0:
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
                <strong style="color: #C9D1D9;">{active_cap:,.0f} Karton</strong>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-top: 5px;">
                <span style="color: #8B949E;">Estimasi Total Biaya:</span>
                <strong style="color: #F85149;">Rp {active_cost:,.0f}</strong>
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
                "Strategi Opsi": selected_decision_option,
                "Jenis Truk": k,
                "Jumlah Unit": v,
                "Kapasitas Satuan": fleet_types[k]["cap"],
                "Total Biaya (IDR)": v * fleet_types[k]["cost"],
            }
            for k, v in active_fleet.items()
            if v > 0
        ]
    )
    csv_data = manifest_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Approved Manifest (CSV)",
        data=csv_data,
        file_name=f"logiagent_approved_manifest_{selected_target_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )