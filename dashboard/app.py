"""
dashboard/app.py
================
ValoreScout — Dashboard Streamlit

Lanzar con:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from models.risk_model import (
    PlayerProfile, TransferDeal, ClubProfile,
    TransferRiskEngine, load_player_from_data,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ValoreScout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .metric-card {
    background: #1c2333;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 6px 0;
    border-left: 4px solid #4f8ef7;
  }
  .verdict-buy    { color: #3fb950; font-size: 1.6rem; font-weight: 800; }
  .verdict-avoid  { color: #f78166; font-size: 1.6rem; font-weight: 800; }
  .verdict-neutral{ color: #d29922; font-size: 1.6rem; font-weight: 800; }
  .section-title  { color: #8b949e; font-size: 0.75rem; text-transform: uppercase;
                    letter-spacing: 0.1em; margin-bottom: 4px; }
  h1 { color: #e6edf3 !important; }
  h2, h3 { color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path(__file__).parent.parent / "data" / "processed" / "master_players.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

df = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/football.png", width=60)
    st.title("ValoreScout")
    st.caption("Quantitative risk intelligence\nfor football transfers")
    st.divider()

    st.subheader("🔍 Jugador")

    # Búsqueda
    search = st.text_input("Buscar jugador", placeholder="Ej: Mbappé, Salah...")

    if search and not df.empty:
        matches = df[df["player"].str.contains(search, case=False, na=False)]
        if not matches.empty:
            player_name = st.selectbox(
                "Seleccionar",
                matches["player"].tolist(),
            )
        else:
            st.warning("Jugador no encontrado")
            player_name = None
    else:
        player_name = None

    st.divider()
    st.subheader("💰 Parámetros del fichaje")

    fee     = st.number_input("Fee de traspaso (M€)", 0.0, 300.0, 50.0, step=5.0)
    salary  = st.number_input("Salario anual (M€)",   1.0, 50.0,  10.0, step=1.0)
    years   = st.slider("Años de contrato", 1, 6, 4)

    st.divider()
    st.subheader("🏟️ Perfil del club")

    club_name    = st.text_input("Nombre del club", "Mi Club")
    salary_cap   = st.number_input("Salario máximo club (M€)", 5.0, 60.0, 20.0, step=2.0)
    avg_perf     = st.slider("Nivel medio del equipo", -1.0, 2.0, 0.0, step=0.1,
                              help="Performance score medio del equipo actual")
    squad_age    = st.slider("Edad media del equipo", 22.0, 32.0, 26.0, step=0.5)
    tactical     = st.selectbox("Estilo táctico", ["balanced", "attacking", "defensive"])

    st.divider()
    st.subheader("⚙️ Configuración")

    injury_hist = st.selectbox("Historial de lesiones",
                                ["low", "medium", "high"],
                                index=1)
    n_sims      = st.select_slider("Simulaciones MC",
                                    options=[10_000, 50_000, 100_000, 200_000],
                                    value=100_000)

    run = st.button("🚀 Analizar fichaje", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("⚽ ValoreScout")
st.caption("Quantitative risk intelligence for football transfers")

if not run or player_name is None:
    # Landing
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Modelo cuantitativo")
        st.markdown("GBM + Poisson para modelar la evolución del valor de mercado con riesgo de lesión.")
    with col2:
        st.markdown("### 🎯 Análisis de fit")
        st.markdown("Score multidimensional de encaje táctico, salarial y deportivo club-jugador.")
    with col3:
        st.markdown("### 💡 Veredicto accionable")
        st.markdown("COMPRAR / NEUTRAL / EVITAR con VaR al 95% y probabilidad de retorno.")

    st.markdown("---")

    if not df.empty:
        st.subheader("🏆 Top jugadores por rendimiento")
        cols_show = [c for c in ["player","team","league","minutes_played",
                                  "goals_per90","assists_per90","performance_score"]
                     if c in df.columns]
        top = (df[cols_show]
               .dropna(subset=["performance_score"])
               .sort_values("performance_score", ascending=False)
               .head(20))
        top.columns = [c.replace("_"," ").title() for c in top.columns]
        st.dataframe(top, use_container_width=True, hide_index=True)
    st.stop()

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

player_data = load_player_from_data(player_name)

if player_data is None:
    st.error(f"No se pudo cargar el jugador: {player_name}")
    st.stop()

player_data.injury_history = injury_hist

deal = TransferDeal(fee_m=fee, salary_m=salary, contract_years=years)
club = ClubProfile(
    name=club_name,
    avg_performance_score=avg_perf,
    salary_cap_m=salary_cap,
    tactical_style=tactical,
    squad_age=squad_age,
)

with st.spinner(f"Ejecutando {n_sims:,} simulaciones Monte Carlo..."):
    engine = TransferRiskEngine(n_simulations=n_sims)
    result = engine.analyse(player_data, deal, club)

# ---------------------------------------------------------------------------
# Results layout
# ---------------------------------------------------------------------------

# Header
st.markdown("---")
col_title, col_verdict = st.columns([3, 1])
with col_title:
    st.header(f"{player_data.name}")
    st.caption(f"{player_data.position} · {player_data.age} años · "
               f"{player_data.league_origin} · "
               f"Disponibilidad: {player_data.availability_rate:.0%}")
with col_verdict:
    verdict_class = {
        "✅ COMPRAR": "verdict-buy",
        "❌ EVITAR":  "verdict-avoid",
        "⚠️  NEUTRAL": "verdict-neutral",
    }.get(result.verdict, "verdict-neutral")
    st.markdown(f'<p class="{verdict_class}">{result.verdict}</p>',
                unsafe_allow_html=True)

st.markdown("---")

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Fair Value",     f"{result.fair_value_m:.1f}M€",
          delta=f"{result.fair_value_m - fee:+.1f}M€ vs fee")
k2.metric("ROI Esperado",   f"{result.expected_roi_m:+.1f}M€")
k3.metric("VaR 95%",        f"{result.var_95_m:+.1f}M€")
k4.metric("P(ROI > 0)",     f"{result.prob_positive:.1%}")
k5.metric("Fit Score",      f"{result.fit_score:.2f} / 1.00")

st.markdown("---")

# Charts row
col_hist, col_fit = st.columns([3, 2])

# --- Distribution of outcomes ---
with col_hist:
    st.subheader("Distribución de outcomes (Monte Carlo)")
    outcomes = result.simulated_outcomes
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=outcomes,
        nbinsx=80,
        marker_color="#4f8ef7",
        opacity=0.7,
        name="Outcomes",
    ))

    # VaR line
    fig.add_vline(x=result.var_95_m, line_dash="dash", line_color="#f78166",
                  annotation_text=f"VaR 95%: {result.var_95_m:+.1f}M€",
                  annotation_position="top left",
                  annotation_font_color="#f78166")

    # Expected ROI line
    fig.add_vline(x=result.expected_roi_m, line_dash="dot", line_color="#3fb950",
                  annotation_text=f"E[ROI]: {result.expected_roi_m:+.1f}M€",
                  annotation_position="top right",
                  annotation_font_color="#3fb950")

    # Zero line
    fig.add_vline(x=0, line_color="#8b949e", line_width=1)

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1c2333",
        font_color="#c9d1d9",
        xaxis_title="Outcome (M€)",
        yaxis_title="Frecuencia",
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Fit score breakdown ---
with col_fit:
    st.subheader("Fit Score — desglose")
    fit_engine = engine.fit
    _, fit_breakdown = fit_engine.score(player_data, deal, club)

    labels = list(fit_breakdown.keys())
    values = list(fit_breakdown.values())

    fig2 = go.Figure(go.Bar(
        x=values,
        y=[l.capitalize() for l in labels],
        orientation="h",
        marker=dict(
            color=["#3fb950" if v >= 0.6 else "#d29922" if v >= 0.4 else "#f78166"
                   for v in values],
        ),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig2.add_vline(x=0.6, line_dash="dot", line_color="#8b949e")
    fig2.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1c2333",
        font_color="#c9d1d9",
        xaxis=dict(range=[0, 1.1], title="Score"),
        height=350,
        margin=dict(l=20, r=20, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- Convergence mini-plot ---
st.subheader("Convergencia del estimador Monte Carlo")

sample_sizes = np.logspace(2, np.log10(n_sims), 30, dtype=int)
running_means = [result.simulated_outcomes[:n].mean() for n in sample_sizes]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=sample_sizes, y=running_means,
    mode="lines", line=dict(color="#4f8ef7", width=2),
    name="ROI medio"
))
fig3.add_hline(y=result.expected_roi_m, line_dash="dash",
               line_color="#3fb950",
               annotation_text=f"Estimación final: {result.expected_roi_m:+.1f}M€",
               annotation_font_color="#3fb950")
fig3.update_layout(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1c2333",
    font_color="#c9d1d9",
    xaxis=dict(type="log", title="Número de simulaciones"),
    yaxis_title="ROI esperado (M€)",
    height=280,
    margin=dict(l=20, r=20, t=10, b=40),
    showlegend=False,
)
st.plotly_chart(fig3, use_container_width=True)

# --- Summary table ---
st.markdown("---")
st.subheader("📋 Resumen del fichaje")

summary = {
    "Parámetro": [
        "Jugador", "Club", "Posición", "Edad",
        "Fee", "Salario anual", "Años contrato", "Coste total",
        "Fair value estimado", "ROI esperado", "VaR 95%",
        "P(ROI positivo)", "P(recuperar fee)", "Fit score", "Veredicto"
    ],
    "Valor": [
        player_data.name, club_name, player_data.position, f"{player_data.age} años",
        f"{fee:.1f}M€", f"{salary:.1f}M€", f"{years} años",
        f"{deal.total_cost_m:.1f}M€",
        f"{result.fair_value_m:.1f}M€",
        f"{result.expected_roi_m:+.1f}M€",
        f"{result.var_95_m:+.1f}M€",
        f"{result.prob_positive:.1%}",
        f"{result.prob_recoup:.1%}",
        f"{result.fit_score:.2f} / 1.00",
        result.verdict,
    ]
}
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)