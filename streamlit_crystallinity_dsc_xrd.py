# Streamlit App – DSC & XRD Crystallinity (Auto Regions)
# Author: ChatGPT (GPT-5 Thinking)
# Description:
#   One-stop app for computing crystallinity from DSC and XRD data.
#   • DSC: integrates ΔHm and ΔHcc using automatic peak detection and baseline, then
#           computes Xc = ((ΔHm − ΔHcc) / (ΔH0 * w_polymer)) * 100.
#           Handles sign convention (endothermic up/down) and heating rate conversion.
#   • XRD: estimates amorphous background via asymmetric least squares (AsLS) and computes
#           Xc = Acryst / (Acryst + Aam) * 100 over a chosen 2θ range.
#           Optional Segal method (for cellulose) with user-defined peak/min positions.
#   Plots are interactive (Plotly). Exports CSVs with results and processed curves.

import io
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from scipy.signal import find_peaks, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

st.set_page_config(page_title="Crystallinity – DSC & XRD", page_icon="🧊", layout="wide")

# -------------------------- Helpers & Baselines --------------------------- #

def asls_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens).
    y: signal (1D), lam: smoothness (higher = smoother), p: asymmetry 0<p<1, niter: iterations.
    Returns baseline vector.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L))
    D = D[2:]  # second-difference operator (L-2 x L)
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def integrate_trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def nearest_index(x: np.ndarray, value: float) -> int:
    return int(np.clip(np.searchsorted(x, value), 0, len(x) - 1))


@dataclass
class DSCResult:
    dH_melt: float
    dH_cc: float
    Xc_percent: float
    details: Dict


@dataclass
class XRDResult:
    Xc_percent: float
    A_total: float
    A_amorphous: float
    A_crystalline: float
    details: Dict


# ------------------------------ UI Header -------------------------------- #
st.title("🧊 Cristalinidade por DSC & DRX")
st.caption("Upload de CSVs, identificação automática de regiões amorfas/cristalinas e cálculo do grau de cristalinidade.")

# ------------------------------- Tabs ------------------------------------ #
tab_dsc, tab_xrd = st.tabs(["DSC", "XRD"])

# ================================== DSC ================================== #
with tab_dsc:
    st.subheader("DSC – Cristalinidade via ΔH")
    colu = st.columns(2)
    with colu[0]:
        up = st.file_uploader("CSV de DSC (Temperatura, Fluxo de Calor)", type=["csv", "txt"], key="dsc_up")
    with colu[1]:
        st.markdown("""
        **CSV esperado (exemplos de cabeçalhos)**
        - `temperature,heat_flow` (°C ou K; W/g ou mW/mg)
        - Outras colunas são ignoradas. Você mapeará abaixo.
        """)

    if up is not None:
        raw = pd.read_csv(up)
    else:
        st.info("Envie um CSV para continuar.")
        st.stop()

    cols = list(raw.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        col_T = st.selectbox("Coluna de Temperatura", cols, index=0)
        unit_T = st.selectbox("Unidade de Temperatura", ["°C", "K"], index=0)
    with c2:
        col_HF = st.selectbox("Coluna de Fluxo de Calor", cols, index=min(1, len(cols) - 1))
        unit_HF = st.selectbox("Unidade de Fluxo", ["W/g", "mW/mg", "mW", "W"], index=0)
    with c3:
        beta_val = st.number_input("Taxa de aquecimento β", value=10.0, min_value=0.001, step=0.5, help="K/min ou °C/min")
        beta_unit = st.selectbox("Unidade de β", ["K/min", "°C/min"], index=0)

    # Normalizações e conversões
    df = raw[[col_T, col_HF]].dropna().copy()
    df.columns = ["T_in", "HF_in"]

    # Temperatura em K para consistência de cálculo de delta T (mas plot em °C se usuário selecionar)
    if unit_T == "°C":
        df["T_K"] = df["T_in"] + 273.15
        T_plot = df["T_in"].values  # manter °C no plot
        T_units_label = "°C"
    else:
        df["T_K"] = df["T_in"]
        T_plot = df["T_in"].values - 273.15  # plot em °C por familiaridade
        T_units_label = "°C"

    # Converter fluxo para W/g se possível
    # Assumimos que o arquivo já é normalizado por massa (W/g ou mW/mg). Para mW ou W sem normalização, o usuário deve ajustar externamente.
    if unit_HF == "W/g":
        df["HF_Wg"] = df["HF_in"].astype(float)
    elif unit_HF == "mW/mg":  # 1 mW/mg = 1 W/g
        df["HF_Wg"] = df["HF_in"].astype(float)
    elif unit_HF == "mW":
        st.warning("mW absoluto detectado. O app assumirá que os dados já estão normalizados por massa. Converta para W/g para resultados corretos.")
        df["HF_Wg"] = df["HF_in"].astype(float) * 1e-3  # Proxy
    else:  # W
        st.warning("W absoluto detectado. O app assumirá que os dados já estão normalizados por massa. Converta para W/g para resultados corretos.")
        df["HF_Wg"] = df["HF_in"].astype(float)

    # Sinal: alguns instrumentos plottam endo para baixo. Permitir inversão.
    c1, c2, c3 = st.columns(3)
    with c1:
        endo_up = st.checkbox("Endotérmico para CIMA", value=True, help="Desmarque se seu DSC plota endo para baixo.")
    with c2:
        do_smooth = st.checkbox("Suavizar (Savitzky–Golay)", value=True)
    with c3:
        sg_window = st.slider("Janela SG (pontos, ímpar)", 5, 101, 21, step=2)

    y = df["HF_Wg"].to_numpy()
    if not endo_up:
        y = -y

    if do_smooth:
        try:
            y_sm = savgol_filter(y, sg_window, 3)
        except Exception:
            y_sm = y
    else:
        y_sm = y

    # Baseline por AsLS
    st.markdown("### Baseline automático (AsLS)")
    b1, b2, b3 = st.columns(3)
    with b1:
        lam = st.number_input("λ (suavidade)", value=1e6, min_value=1e3, max_value=1e12, step=1e6, format="%.0f")
    with b2:
        p_asym = st.slider("p (assimetria)", 0.001, 0.5, 0.01)
    with b3:
        niter = st.slider("Iterações", 5, 50, 15)

    base = asls_baseline(y_sm, lam=float(lam), p=float(p_asym), niter=int(niter))
    y_corr = y_sm - base

    # Seleção de janela para análise (para evitar regiões de baseline ruim)
    T_min = float(np.min(T_plot))
    T_max = float(np.max(T_plot))
    t1, t2 = st.slider("Janela de temperatura para análise (°C)", T_min, T_max, (T_min, T_max))
    idx_win = (T_plot >= t1) & (T_plot <= t2)

    # Plot DSC
    fig_dsc = go.Figure()
    fig_dsc.add_trace(go.Scatter(x=T_plot, y=y, mode="lines", name="HF (original)"))
    fig_dsc.add_trace(go.Scatter(x=T_plot, y=y_sm, mode="lines", name="HF (suavizado)", line=dict(width=2)))
    fig_dsc.add_trace(go.Scatter(x=T_plot, y=base, mode="lines", name="Baseline (AsLS)", line=dict(dash="dot")))
    fig_dsc.add_trace(go.Scatter(x=T_plot, y=y_corr, mode="lines", name="HF - baseline", line=dict(width=3)))
    fig_dsc.add_vrect(x0=t1, x1=t2, fillcolor="LightSkyBlue", opacity=0.15, line_width=0, annotation_text="Janela")
    fig_dsc.update_layout(height=520, template="plotly_dark", xaxis_title=f"Temperatura ({T_units_label})", yaxis_title="Fluxo (W/g, endo ↑)")
    st.plotly_chart(fig_dsc, use_container_width=True)

    st.markdown("### Detecção automática de picos (fusão e cristalização)")
    c1, c2, c3 = st.columns(3)
    with c1:
        prom = st.number_input("Proeminência mínima", value=0.01, min_value=0.0, step=0.01)
    with c2:
        dist = st.number_input("Distância mínima entre picos (pontos)", value=30, min_value=1, step=5)
    with c3:
        frac_bounds = st.slider("Limite de integração (% do pico)", 1, 40, 5) / 100.0

    # heating rate conversion
    beta_Ks = (beta_val / 60.0)  # K/s (°C/min == K/min numericamente)

    # Usar janela
    T_w = T_plot[idx_win]
    y_corr_w = y_corr[idx_win]
    T_K_w = df["T_K"].to_numpy()[idx_win]

    # Endotérmicos (fusão): picos positivos em y_corr
    pk_pos, prop_pos = find_peaks(y_corr_w, prominence=prom, distance=int(dist))
    # Exotérmicos (CC): picos negativos -> procurar em -y
    pk_neg, prop_neg = find_peaks(-y_corr_w, prominence=prom, distance=int(dist))

    # Função para integrar em limites onde sinal cai para frac_bounds*altura
    def integrate_peak(T, yb, p_idx, sign="pos") -> Tuple[float, float, float]:
        y0 = yb[p_idx]
        thr = frac_bounds * abs(y0)
        # esquerda
        i = p_idx
        while i > 0 and abs(yb[i]) > thr:
            i -= 1
        left = i
        # direita
        j = p_idx
        while j < len(yb) - 1 and abs(yb[j]) > thr:
            j += 1
        right = j
        if right <= left + 2:
            return 0.0, T[p_idx], T[p_idx]
        area = integrate_trapz(T[left:right + 1], yb[left:right + 1]) / max(beta_Ks, 1e-12)  # J/g
        return float(area), float(T[left]), float(T[right])

    # Integrar áreas
    dH_melt = 0.0
    melt_windows = []
    for p in pk_pos:
        a, L, R = integrate_peak(T_K_w, y_corr_w, int(p), sign="pos")
        dH_melt += a
        melt_windows.append((L, R, a))

    dH_cc = 0.0
    cc_windows = []
    for p in pk_neg:
        a, L, R = integrate_peak(T_K_w, y_corr_w, int(p), sign="neg")
        dH_cc += abs(a)  # exo (negativo), usar módulo
        cc_windows.append((L, R, -abs(a)))

    # Parâmetros para Xc
    st.markdown("### Parâmetros de referência")
    r1, r2, r3 = st.columns(3)
    with r1:
        dH0 = st.number_input("ΔH₀ (J/g) do polímero 100% cristalino", value=293.0, min_value=0.1, help="Ex.: PE ~ 293 J/g; PP ~ 207 J/g; PLA ~ 93–106 J/g; ver literatura.")
    with r2:
        w_poly = st.number_input("Fraç. mássica de polímero na amostra (w)", value=1.0, min_value=0.001, max_value=1.0, step=0.01)
    with r3:
        clip_0_100 = st.checkbox("Ajustar Xc para [0,100]%", value=True)

    Xc = ((dH_melt - dH_cc) / (max(dH0, 1e-12) * max(w_poly, 1e-12))) * 100.0
    if clip_0_100:
        Xc = float(np.clip(Xc, 0.0, 100.0))

    # Resultados
    st.success(f"ΔH_fusão = {dH_melt:.2f} J/g | ΔH_cc = {dH_cc:.2f} J/g | **Xc = {Xc:.2f}%**")

    # Plot com janelas integradas
    fig_int = go.Figure()
    fig_int.add_trace(go.Scatter(x=T_plot[idx_win], y=y_corr_w, mode="lines", name="HF - baseline"))
    for L, R, a in melt_windows:
        fig_int.add_vrect(x0=L - 273.15, x1=R - 273.15, fillcolor="LightGreen", opacity=0.25, line_width=0, annotation_text=f"ΔHm {a:.1f} J/g")
    for L, R, a in cc_windows:
        fig_int.add_vrect(x0=L - 273.15, x1=R - 273.15, fillcolor="LightCoral", opacity=0.25, line_width=0, annotation_text=f"ΔHcc {abs(a):.1f} J/g")
    fig_int.update_layout(height=420, template="plotly_dark", xaxis_title="Temperatura (°C)", yaxis_title="HF - baseline (W/g)")
    st.plotly_chart(fig_int, use_container_width=True)

    # Exportar resultados e curvas
    res_dsc = DSCResult(dH_melt=float(dH_melt), dH_cc=float(dH_cc), Xc_percent=float(Xc), details={"windows_melt": melt_windows, "windows_cc": cc_windows, "beta_K_per_s": beta_Ks, "endo_up": endo_up, "lam": lam, "p": p_asym})
    df_out = pd.DataFrame({
        "T_C": T_plot,
        "HF_original(W/g, endo up?)": y,
        "HF_smooth(W/g)": y_sm,
        "Baseline(W/g)": base,
        "HF_minus_baseline(W/g)": y_corr,
    })
    csv_buf = io.StringIO()
    df_out.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Baixar curvas processadas (CSV)", csv_buf.getvalue(), file_name="dsc_processed.csv", mime="text/csv")

    res_buf = io.StringIO()
    pd.DataFrame({
        "dH_melt_Jg": [res_dsc.dH_melt],
        "dH_cc_Jg": [res_dsc.dH_cc],
        "Xc_percent": [res_dsc.Xc_percent],
    }).to_csv(res_buf, index=False)
    st.download_button("⬇️ Baixar resultados (CSV)", res_buf.getvalue(), file_name="dsc_results.csv", mime="text/csv")

# ================================== XRD ================================== #
with tab_xrd:
    st.subheader("XRD – Cristalinidade por área (baseline amorfo) e Segal (opcional)")
    colu = st.columns(2)
    with colu[0]:
        upx = st.file_uploader("CSV de XRD (2θ, Intensidade)", type=["csv", "txt"], key="xrd_up")
    with colu[1]:
        st.markdown("""
        **CSV esperado**
        - `two_theta,intensity`
        - Você mapeará abaixo. O app calcula automaticamente o background amorfo.
        """)

    if upx is None:
        st.info("Envie um CSV para continuar.")
        st.stop()

    rawx = pd.read_csv(upx)
    colsx = list(rawx.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        col_tt = st.selectbox("Coluna de 2θ (graus)", colsx, index=0)
    with c2:
        col_I = st.selectbox("Coluna de Intensidade (contagens)", colsx, index=min(1, len(colsx) - 1))
    with c3:
        do_smooth_x = st.checkbox("Suavizar (Savitzky–Golay)", value=True)

    xrd = rawx[[col_tt, col_I]].dropna().astype(float)
    xrd.columns = ["tt", "I"]
    xrd = xrd.sort_values("tt")

    if do_smooth_x:
        # garantir janela ímpar e >=5
        win = st.slider("Janela SG (pontos, ímpar)", 5, 151, 21, step=2)
        try:
            I_sm = savgol_filter(xrd["I"].to_numpy(), win, 3)
        except Exception:
            I_sm = xrd["I"].to_numpy()
    else:
        I_sm = xrd["I"].to_numpy()

    tt = xrd["tt"].to_numpy()

    st.markdown("### Baseline amorfo (AsLS)")
    b1, b2, b3 = st.columns(3)
    with b1:
        lamx = st.number_input("λ (suavidade)", value=1e7, min_value=1e4, max_value=1e12, step=1e6, format="%.0f")
    with b2:
        p_asymx = st.slider("p (assimetria)", 0.001, 0.5, 0.01)
    with b3:
        niterx = st.slider("Iterações", 5, 80, 30)

    base_x = asls_baseline(I_sm, lam=float(lamx), p=float(p_asymx), niter=int(niterx))
    I_corr = I_sm - base_x
    I_corr = np.clip(I_corr, 0.0, None)  # parte cristalina não-negativa

    # Janela de 2θ para cálculo
    tt_min = float(np.min(tt)); tt_max = float(np.max(tt))
    t1, t2 = st.slider("Janela 2θ (graus)", tt_min, tt_max, (tt_min, tt_max))
    idx = (tt >= t1) & (tt <= t2)

    # Áreas
    A_total = integrate_trapz(tt[idx], I_sm[idx])
    A_am = integrate_trapz(tt[idx], base_x[idx])
    # assegurar que A_am não exceda A_total por ruído
    A_am = min(A_am, A_total)
    A_cr = max(A_total - A_am, 0.0)

    Xc_area = 100.0 * A_cr / max(A_total, 1e-12)

    # Picos (visual)
    promx = st.number_input("Proeminência mínima (picos)", value=float(np.max(I_corr) * 0.02 if np.max(I_corr) > 0 else 1.0), min_value=0.0, step=1.0)
    distx = st.number_input("Distância mínima (pontos)", value=20, min_value=1, step=5)
    pk, props = find_peaks(I_corr[idx], prominence=promx, distance=int(distx))
    tt_idx = tt[idx]

    # Plot XRD
    fig_x = go.Figure()
    fig_x.add_trace(go.Scatter(x=tt, y=xrd["I"], mode="lines", name="Int. original", opacity=0.4))
    fig_x.add_trace(go.Scatter(x=tt, y=I_sm, mode="lines", name="Int. suavizada", line=dict(width=2)))
    fig_x.add_trace(go.Scatter(x=tt, y=base_x, mode="lines", name="Baseline amorfo", line=dict(dash="dot")))
    fig_x.add_trace(go.Scatter(x=tt, y=I_corr, mode="lines", name="Cristalino (I - base)", line=dict(width=3)))
    for p in pk:
        fig_x.add_vline(x=float(tt_idx[p]), line=dict(color="MediumSpringGreen", width=1))
    fig_x.add_vrect(x0=t1, x1=t2, fillcolor="LightSkyBlue", opacity=0.15, line_width=0, annotation_text="Janela")
    fig_x.update_layout(height=520, template="plotly_dark", xaxis_title="2θ (graus)", yaxis_title="Intensidade (contagens)")
    st.plotly_chart(fig_x, use_container_width=True)

    # Método de Segal (opcional)
    st.markdown("### Método de Segal (opcional – útil para celulose)")
    use_segal = st.checkbox("Calcular também por Segal", value=False)
    if use_segal:
        seg1, seg2 = st.columns(2)
        with seg1:
            tt_peak = st.number_input("2θ do pico cristalino (ex.: I002)", value=22.6)
        with seg2:
            tt_am = st.number_input("2θ do mínimo amorfo", value=18.0)
        # Interpolar intensidades na suavizada (mais estável)
        I_interp = np.interp([tt_peak, tt_am], tt, I_sm)
        I002, Iam = float(I_interp[0]), float(I_interp[1])
        if I002 <= 0:
            Xc_segal = 0.0
        else:
            Xc_segal = max(0.0, min(100.0, (I002 - Iam) / I002 * 100.0))
        st.info(f"Segal: I002={I002:.1f}, Iam={Iam:.1f} → **Xc = {Xc_segal:.2f}%**")
    else:
        Xc_segal = None

    # Resultados finais (área)
    st.success(f"Área total = {A_total:.1f}, Amorfa = {A_am:.1f}, Cristalina = {A_cr:.1f} → **Xc = {Xc_area:.2f}%**")

    # Exports
    res_xrd = XRDResult(Xc_percent=float(Xc_area), A_total=float(A_total), A_amorphous=float(A_am), A_crystalline=float(A_cr), details={"lam": lamx, "p": p_asymx, "niter": niterx, "range": (t1, t2), "Xc_segal": Xc_segal})
    x_csv = io.StringIO()
    pd.DataFrame({
        "two_theta": tt,
        "I_raw": xrd["I"].to_numpy(),
        "I_smooth": I_sm,
        "baseline": base_x,
        "I_crystalline": I_corr,
    }).to_csv(x_csv, index=False)
    st.download_button("⬇️ Baixar curvas processadas (CSV)", x_csv.getvalue(), file_name="xrd_processed.csv", mime="text/csv")

    r_csv = io.StringIO()
    pd.DataFrame({
        "Xc_area_percent": [res_xrd.Xc_percent],
        "A_total": [res_xrd.A_total],
        "A_amorphous": [res_xrd.A_amorphous],
        "A_crystalline": [res_xrd.A_crystalline],
        "Xc_segal_percent": [Xc_segal if Xc_segal is not None else np.nan],
    }).to_csv(r_csv, index=False)
    st.download_button("⬇️ Baixar resultados (CSV)", r_csv.getvalue(), file_name="xrd_results.csv", mime="text/csv")

# ------------------------------ Footnotes -------------------------------- #
with st.expander("Notas & Boas Práticas"):
    st.markdown(
        """
        **DSC**
        - A integral de entalpia é calculada em J/g a partir de `HF (W/g)` e da taxa de aquecimento β (K/s): \(\Delta H = \int \frac{HF}{\beta} \, dT\).
        - A detecção automática usa proeminência e limite (\% da altura do pico) para definir as janelas; ajuste conforme ruído.
        - `ΔH₀` depende do polímero e do grau de perfeição cristalina (consulte literatura). Use a fração mássica de polímero se houver aditivos/cargas.
        - Endotérmico para cima/baixo varia por instrumento. Use o seletor para padronizar.

        **XRD**
        - O background amorfo é estimado por AsLS. Ajuste λ (suavidade) e *p* (assimetria) até que a linha siga bem a banda larga amorfa sem engolir os picos.
        - O método por área considera `A_cristalina = ∫(I - baseline)_+` e `A_amorfa = ∫ baseline` dentro da janela de 2θ.
        - O método de Segal é útil para celulose (requer posições de pico/mínimo típicas do polimorfo analisado).

        **Exportação**
        - Baixe as curvas processadas e os resultados em CSV para documentação/traços de auditoria.
        """
    )
