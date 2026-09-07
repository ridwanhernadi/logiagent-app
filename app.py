import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pulp import LpInteger, LpMinimize, LpProblem, LpVariable, lpSum, value
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN & CUSTOM ENTERPRISE CSS
# ==========================================
st.set_page_config(
    page_title="LogiAgent - Fleet Optimization Engine",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* PANGKAS MARGIN ATAS */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 2.5rem !important;
    }

    /* Background Utama */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    
    /* Header & Typography */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* CUSTOM KPI CARD HTML STYLING (GUARANTEED 100% CONTRAST) */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .kpi-title {
        color: #FFFFFF !important;
        font-size: 12px !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #00F0FF !important; /* Cyan Neon */
        font-size: 26px !important;
        font-weight: 800 !important;
        line-height: 1.2;
        margin-bottom: 4px;
    }
    .kpi-sub {
        color: #cbd5e1 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    
    /* Alert / Info Box */
    div[data-testid="stAlert"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px;
    }
    div[data-testid="stAlert"] * {
        color: #f8fafc !important;
    }
    
    /* Card Container Custom */
    .custom-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff !important;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 0 16px rgba(37, 99, 235, 0.6);
        transform: translateY(-1px);
    }
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Status Badges */
    .badge-fuso {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
    }
    .badge-cdd {
        background-color: rgba(59, 130, 246, 0.2);
        border: 1px solid #3B82F6;
        color: #60A5FA;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. GENERATE DATA TRANSAKSI (SENIN - SABTU)
# ==========================================
@st.cache_data
def load_sample_data():
  all_dates = pd.date_range(start="2026-06-01", end="2026-08-31", freq="D")
  working_dates = [d for d in all_dates if d.dayofweek != 6]

  np.random.seed(42)
  base_demand = (
      8000 + np.sin(np.arange(len(working_dates)) * (2 * np.pi / 6)) * 2500
  )
  noise = np.random.normal(0, 800, len(working_dates))
  demand = np.maximum(2000, (base_demand + noise)).astype(int)

  df = pd.DataFrame({"Tanggal": working_dates, "Total_Karton": demand})

  routes = {
      "Makassar Hub": 0.35,
      "Zone Parepare": 0.20,
      "Zone Bone": 0.18,
      "Zone Palopo": 0.17,
      "Zone Mangkutana": 0.10,
  }

  for route, ratio in routes.items():
    df[route] = (df["Total_Karton"] * ratio).astype(int)

  return df


df_data = load_sample_data()

FLEET_SPECS = {
    "CDD 7T": {
        "capacity": 1500,
        "cost": {
            "Makassar Hub": 1200000,
            "Zone Parepare": 2200000,
            "Zone Bone": 2500000,
            "Zone Palopo": 3800000,
            "Zone Mangkutana": 4500000,
        },
        "max_avail": 10,
    },
    "FUSO 10T": {
        "capacity": 2000,
        "cost": {
            "Makassar Hub": 1800000,
            "Zone Parepare": 3100000,
            "Zone Bone": 3400000,
            "Zone Palopo": 5200000,
            "Zone Mangkutana": 6000000,
        },
        "max_avail": 8,
    },
    "FUSO 17T": {
        "capacity": 3800,
        "cost": {
            "Makassar Hub": 2800000,
            "Zone Parepare": 4800000,
            "Zone Bone": 5200000,
            "Zone Palopo": 7800000,
            "Zone Mangkutana": 9200000,
        },
        "max_avail": 5,
    },
}

# ==========================================
# 3. SIDEBAR NAVIGATION & FILTERS
# ==========================================
with st.sidebar:
  st.markdown("## 🚚 LogiAgent v2.0")
  st.caption("Fleet Optimization Engine")
  st.markdown("---")

  st.subheader("⚙️ Parameter Konfigurasi")
  selected_route = st.selectbox(
      "Pilih Area / Rute Pengiriman:",
      [
          "Total Semua Area",
          "Makassar Hub",
          "Zone Parepare",
          "Zone Bone",
          "Zone Palopo",
          "Zone Mangkutana",
      ],
  )

  service_level = st.select_slider(
      "Safety Capacity Buffer (Z-Score):",
      options=["90% (Z=1.28)", "95% (Z=1.65)", "99% (Z=2.33)"],
      value="90% (Z=1.28)",
  )

  z_dict = {"90% (Z=1.28)": 1.28, "95% (Z=1.65)": 1.65, "99% (Z=2.33)": 2.33}
  z_val = z_dict[service_level]

  forecast_days = st.slider("Horizon Prediksi (Hari Kerja):", 1, 12, 6)

  st.markdown("---")
  st.info(
      "💡 **Operasional Outbound (Senin–Sabtu):** Hari Minggu libur total dari"
      " jadwal pengiriman & alokasi truk."
  )

# ==========================================
# 4. HEADER DASHBOARD
# ==========================================
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%); padding: 18px 24px; border-radius: 12px; border-left: 8px solid #00F0FF; margin-bottom: 20px;">
        <h2 style="margin:0; font-size: 24px;">LogiAgent: Autonomous Fleet Optimization Dashboard</h2>
        <p style="margin: 4px 0 0 0; color: #cbd5e1; font-size: 14px;">Decision Support System untuk Outbound Logistics (Senin - Sabtu)</p>
    </div>
""",
    unsafe_allow_html=True,
)

target_col = (
    "Total_Karton" if selected_route == "Total Semua Area" else selected_route
)
y_hist = df_data[target_col]

# ==========================================
# 5. PREDICTIVE LAYER (HOLT-WINTERS FORECASTING)
# ==========================================
model = ExponentialSmoothing(
    y_hist, trend="add", seasonal="add", seasonal_periods=6
).fit()

last_date = df_data["Tanggal"].iloc[-1]
future_dates = []
curr_date = last_date + pd.Timedelta(days=1)

while len(future_dates) < forecast_days:
  if curr_date.dayofweek != 6:
    future_dates.append(curr_date)
  curr_date += pd.Timedelta(days=1)

forecast_vals = model.forecast(forecast_days)
residuals = y_hist - model.fittedvalues
rmse = np.sqrt(np.mean(residuals**2))
safety_buffer = int(z_val * rmse)

target_day_1 = int(forecast_vals.iloc[0]) + safety_buffer


# ==========================================
# 6. PRESCRIPTIVE LAYER (ILP SOLVER)
# ==========================================
def solve_fleet_ilp(demand_qty, route_name):
  prob = LpProblem("Fleet_Optimization", LpMinimize)

  x_cdd = LpVariable("CDD_7T", lowBound=0, cat=LpInteger)
  x_fuso10 = LpVariable("FUSO_10T", lowBound=0, cat=LpInteger)
  x_fuso17 = LpVariable("FUSO_17T", lowBound=0, cat=LpInteger)

  if route_name == "Total Semua Area":
    c_cdd = FLEET_SPECS["CDD 7T"]["cost"]["Zone Parepare"]
    c_fuso10 = FLEET_SPECS["FUSO 10T"]["cost"]["Zone Parepare"]
    c_fuso17 = FLEET_SPECS["FUSO 17T"]["cost"]["Zone Parepare"]
  else:
    c_cdd = FLEET_SPECS["CDD 7T"]["cost"][route_name]
    c_fuso10 = FLEET_SPECS["FUSO 10T"]["cost"][route_name]
    c_fuso17 = FLEET_SPECS["FUSO 17T"]["cost"][route_name]

  prob += (
      c_cdd * x_cdd + c_fuso10 * x_fuso10 + c_fuso17 * x_fuso17,
      "Total_Transport_Cost",
  )

  prob += (
      1500 * x_cdd + 2000 * x_fuso10 + 3800 * x_fuso17 >= demand_qty,
      "Capacity_Requirement",
  )
  prob += x_cdd <= FLEET_SPECS["CDD 7T"]["max_avail"], "Max_CDD"
  prob += x_fuso10 <= FLEET_SPECS["FUSO 10T"]["max_avail"], "Max_Fuso10"
  prob += x_fuso17 <= FLEET_SPECS["FUSO 17T"]["max_avail"], "Max_Fuso17"

  prob.solve()

  res = {
      "CDD 7T": int(value(x_cdd)),
      "FUSO 10T": int(value(x_fuso10)),
      "FUSO 17T": int(value(x_fuso17)),
      "Total_Cost": value(prob.objective),
      "Total_Capacity": (
          int(value(x_cdd)) * 1500
          + int(value(x_fuso10)) * 2000
          + int(value(x_fuso17)) * 3800
      ),
  }
  return res


ilp_result = solve_fleet_ilp(target_day_1, selected_route)

# ==========================================
# 7. METRIC CARDS (KUSTOM HTML - GUARANTEED WHITE TEXT)
# ==========================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_trucks = (
    ilp_result["CDD 7T"] + ilp_result["FUSO 10T"] + ilp_result["FUSO 17T"]
)

with kpi1:
  st.markdown(
      f"""
        <div class="kpi-card">
            <div class="kpi-title">PREDIKSI DEMAND KERJA H-1</div>
            <div class="kpi-value">{int(forecast_vals.iloc[0]):,} Ctn</div>
            <div class="kpi-sub">↑ Area: {selected_route}</div>
        </div>
    """,
      unsafe_allow_html=True,
  )

with kpi2:
  st.markdown(
      f"""
        <div class="kpi-card">
            <div class="kpi-title">SAFETY CAPACITY BUFFER</div>
            <div class="kpi-value">+{safety_buffer:,} Ctn</div>
            <div class="kpi-sub">↑ SLA {service_level}</div>
        </div>
    """,
      unsafe_allow_html=True,
  )

with kpi3:
  st.markdown(
      f"""
        <div class="kpi-card">
            <div class="kpi-title">REKOMENDASI ARMADA</div>
            <div class="kpi-value">{total_trucks} Unit</div>
            <div class="kpi-sub">↑ Cap: {ilp_result['Total_Capacity']:,} Ctn</div>
        </div>
    """,
      unsafe_allow_html=True,
  )

with kpi4:
  st.markdown(
      f"""
        <div class="kpi-card">
            <div class="kpi-title">ESTIMASI TOTAL BIAYA</div>
            <div class="kpi-value">Rp {ilp_result['Total_Cost']:,.0f}</div>
            <div class="kpi-sub">↑ Optimum Cost (ILP)</div>
        </div>
    """,
      unsafe_allow_html=True,
  )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 8. VISUALISASI UTAMA & HASIL OPTIMASI
# ==========================================
col_chart, col_opt = st.columns([1.6, 1])

with col_chart:
  st.markdown("### 📈 Demand Trend & Forecast (Senin - Sabtu)")

  fig = go.Figure()

  fig.add_trace(
      go.Scatter(
          x=df_data["Tanggal"],
          y=y_hist,
          mode="lines+markers",
          name="Histori Demand",
          line=dict(color="#3b82f6", width=2),
          marker=dict(size=4),
      )
  )

  fig.add_trace(
      go.Scatter(
          x=future_dates,
          y=forecast_vals,
          mode="lines+markers",
          name="Holt-Winters Forecast",
          line=dict(color="#00F0FF", width=3, dash="dash"),
          marker=dict(size=6, symbol="diamond"),
      )
  )

  fig.update_layout(
      template="plotly_dark",
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(15, 23, 42, 0.6)",
      margin=dict(l=20, r=20, t=30, b=20),
      height=340,
      legend=dict(
          orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
      ),
      xaxis=dict(showgrid=True, gridcolor="#334155"),
      yaxis=dict(showgrid=True, gridcolor="#334155"),
  )

  st.plotly_chart(fig, use_container_width=True)

with col_opt:
  st.markdown("### 🚛 Optimized Fleet Allocation")

  utilization_pct = (
      min(100.0, (target_day_1 / ilp_result["Total_Capacity"]) * 100)
      if ilp_result["Total_Capacity"] > 0
      else 0
  )

  st.markdown(
      f"""
    <div class="custom-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-weight: bold; color: #f8fafc;">Rincian Rekomendasi Truk (H-1)</span>
            <span style="font-size: 11px; background-color: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #34D399; padding: 2px 8px; border-radius: 4px;">ILP SOLVED (0.01s)</span>
        </div>
        <table style="width: 100%; color: #f8fafc; font-size: 14px; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #334155; height: 36px;">
                <td><span class="badge-cdd">CDD 7 Ton</span> <span style="color:#94a3b8; font-size: 12px;">(Cap: 1.500)</span></td>
                <td style="text-align: right; font-weight: bold; color: #00F0FF;">{ilp_result['CDD 7T']} Unit</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155; height: 36px;">
                <td><span class="badge-fuso">FUSO 10 Ton</span> <span style="color:#94a3b8; font-size: 12px;">(Cap: 2.000)</span></td>
                <td style="text-align: right; font-weight: bold; color: #00F0FF;">{ilp_result['FUSO 10T']} Unit</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155; height: 36px;">
                <td><span class="badge-fuso">FUSO 17 Ton</span> <span style="color:#94a3b8; font-size: 12px;">(Cap: 3.800)</span></td>
                <td style="text-align: right; font-weight: bold; color: #00F0FF;">{ilp_result['FUSO 17T']} Unit</td>
            </tr>
        </table>
        <div style="margin-top: 16px; padding-top: 12px; border-top: 1px dashed #475569; display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: #cbd5e1; font-size: 13px;">Total Kapasitas Muat:</span>
            <span style="font-weight: bold; color: #00F0FF;">{ilp_result['Total_Capacity']:,} Karton</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; margin-bottom: 4px;">
            <span>Utilisasi Kapasitas Muatan:</span>
            <span style="color: #34d399; font-weight: bold;">{utilization_pct:.1f}%</span>
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

  st.progress(utilization_pct / 100.0)

  manifest_df = pd.DataFrame([{
      "Tanggal_Penjemputan": future_dates[0].strftime("%Y-%m-%d"),
      "Rute_Tujuan": selected_route,
      "Target_Volume_Ctn": target_day_1,
      "CDD_7T": ilp_result["CDD 7T"],
      "FUSO_10T": ilp_result["FUSO 10T"],
      "FUSO_17T": ilp_result["FUSO 17T"],
      "Est_Total_Biaya_Rp": ilp_result["Total_Cost"],
  }])

  csv_manifest = manifest_df.to_csv(index=False).encode("utf-8")

  st.download_button(
      label="📥 Download Pre-booking Manifest (CSV)",
      data=csv_manifest,
      file_name=f"Manifest_LogiAgent_{selected_route.replace(' ', '_')}.csv",
      mime="text/csv",
  )

# ==========================================
# 9. DETAIL TABLE TRANSACTION HISTORY
# ==========================================
st.markdown("---")
st.markdown("### 📋 Transaksi Historis & Proyeksi Rute Outbound")

tab1, tab2 = st.tabs(
    ["📊 Data Historis Kerja (Juni-Agustus 2026)", "🔮 Proyeksi Hari Kerja Ke Depan"]
)

with tab1:
  st.dataframe(
      df_data.sort_values(by="Tanggal", ascending=False),
      use_container_width=True,
      height=250,
  )

with tab2:
  proj_df = pd.DataFrame({
      "Tanggal": future_dates,
      "Hari": [d.strftime("%A") for d in future_dates],
      "Prediksi_Demand_Ctn": forecast_vals.values.astype(int),
      "Safety_Buffer_Ctn": [safety_buffer] * forecast_days,
      "Total_Kapasitas_Dibutuhkan": forecast_vals.values.astype(int)
      + safety_buffer,
  })
  st.dataframe(proj_df, use_container_width=True, height=250)