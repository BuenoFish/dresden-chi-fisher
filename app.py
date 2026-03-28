import streamlit as st
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import chi2 as chi2_dist
from itertools import product
from io import BytesIO

st.set_page_config(page_title="Kontingenztest", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .stApp { background-color: #f5f4f0; color: #1a1a1a; }
    .header-box { background:#1a1a1a; color:#f5f4f0; padding:2rem 2.5rem; border-radius:4px; margin-bottom:2rem; }
    .header-box h1 { font-size:1.8rem; margin:0 0 0.3rem 0; color:#f5f4f0; }
    .header-box p  { margin:0; color:#aaa; font-size:0.9rem; font-family:'IBM Plex Mono',monospace; }
    .param-card  { background:white; border:1px solid #e0e0e0; border-radius:4px; padding:1rem 1.2rem; margin-bottom:0.5rem; }
    .param-label { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:1px; }
    .param-value { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#1a1a1a; }
    .param-value.sig   { color:#16a34a; }
    .param-value.nosig { color:#dc2626; }
    .info-box  { background:#eef6ff; border-left:3px solid #2563eb; padding:0.8rem 1rem; border-radius:0 4px 4px 0; font-size:0.88rem; margin-bottom:1rem; }
    .warn-box  { background:#fff8e6; border-left:3px solid #f59e0b; padding:0.8rem 1rem; border-radius:0 4px 4px 0; font-size:0.88rem; margin-bottom:1rem; }
    .err-box   { background:#fff0f0; border-left:3px solid #ef4444; padding:0.8rem 1rem; border-radius:0 4px 4px 0; font-size:0.88rem; margin-bottom:1rem; }
    .sig-box   { background:#f0fdf4; border-left:3px solid #16a34a; padding:0.8rem 1rem; border-radius:0 4px 4px 0; font-size:0.88rem; margin-bottom:1rem; }
    .nosig-box { background:#fff0f0; border-left:3px solid #dc2626; padding:0.8rem 1rem; border-radius:0 4px 4px 0; font-size:0.88rem; margin-bottom:1rem; }
    .stButton>button { background:#1a1a1a; color:white; border:none; border-radius:4px;
        font-family:'IBM Plex Mono',monospace; font-size:0.85rem; padding:0.5rem 1.5rem; }
    div[data-testid="stExpander"] { background:white; border:1px solid #e0e0e0; border-radius:4px; }
    div[data-testid="stNumberInput"] input { font-family:'IBM Plex Mono',monospace; }
</style>
""", unsafe_allow_html=True)

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    img_b64 = img_to_base64("frauenkirche_flach.jpg")
    st.markdown(f"""
<div class="header-box" style="display:flex; align-items:stretch; justify-content:space-between; padding:0; overflow:hidden; border-radius:4px; margin:0 0 2rem 0;">
    <div style="padding:2rem 2.5rem; flex:1;">
        <h1 style="font-size:1.8rem; margin:0 0 0.3rem 0; color:#f5f4f0;">📊 Kontingenztest</h1>
        <p style="margin:0; color:#aaa; font-size:0.9rem; font-family:'IBM Plex Mono',monospace;">
            Chi-Quadrat-Test · Fisher's Exakter Test · bis 5×5 Tabellen
        </p>
    </div>
    <div style="width:350px; flex-shrink:0; position:relative; line-height:0;">
        <img src="data:image/jpeg;base64,{img_b64}"
             style="width:100%; height:100%; object-fit:cover; object-position:center; display:block;">
        <div style="position:absolute; top:0; left:0; width:100%; height:100%;
            background: linear-gradient(to right, #1a1a1a 0%, transparent 60%);"></div>
    </div>
</div>
""", unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("""
<div class="header-box">
    <h1>📊 Kontingenztest</h1>
    <p>Chi-Quadrat-Test · Fisher's Exakter Test · bis 5×5 Tabellen</p>
</div>
""", unsafe_allow_html=True)

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def fisher_exact_rxc(table):
    """
    Fisher's exakter Test fuer RxC Tabellen.
    2x2: scipy exakt. 2xC / Rx2: vollstaendige Enumeration (exakt).
    RxC mit grossen N: Monte Carlo (200.000 Permutationen).
    """
    from itertools import product as iproduct
    from math import lgamma
    table = np.array(table, dtype=int)
    r, c = table.shape

    if r == 2 and c == 2:
        _, p = fisher_exact(table)
        return p, "exakt (2x2)"

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    n = int(table.sum())

    def log_prob(t):
        lp = 0
        for i in range(r): lp += lgamma(row_sums[i] + 1)
        for j in range(c): lp += lgamma(col_sums[j] + 1)
        lp -= lgamma(n + 1)
        for i in range(r):
            for j in range(c):
                lp -= lgamma(int(t[i, j]) + 1)
        return lp

    log_p_obs = log_prob(table)

    # Monte Carlo fuer grosse/komplexe Tabellen
    use_mc = (r >= 3 and c >= 3 and n > 30) or n > 200
    if use_mc:
        chi2_obs, _, _, expected = chi2_contingency(table, correction=False)
        n_sim = 200000
        rng = np.random.default_rng(42)
        count = 0
        for _ in range(n_sim):
            sim = rng.multinomial(n, (expected / n).ravel()).reshape(r, c)
            try:
                chi2_sim, _, _, _ = chi2_contingency(sim, correction=False)
                if chi2_sim >= chi2_obs: count += 1
            except Exception: pass
        return count / n_sim, "Monte Carlo (200.000)"

    # Exakte Enumeration via vollstaendige Aufzaehlung
    def enumerate_tables(rs, cs):
        def fill(row, rem_cols, rem_rs, current):
            if row == r - 1:
                last = rem_cols.copy()
                if np.all(last >= 0) and last.sum() == rem_rs[row]:
                    yield np.vstack(current + [last])
                return
            ranges = [range(int(min(rem_cols[j], rem_rs[row])) + 1) for j in range(c)]
            for combo in iproduct(*ranges):
                arr = np.array(combo)
                if arr.sum() != rem_rs[row]: continue
                new_rem = rem_cols - arr
                if np.any(new_rem < 0): continue
                yield from fill(row + 1, new_rem, rem_rs, current + [arr])
        yield from fill(0, cs.copy(), rs, [])

    def logsumexp(vals):
        if not vals: return -np.inf
        m = max(vals)
        return m + np.log(sum(np.exp(v - m) for v in vals))

    log_all, log_extreme = [], []
    for t in enumerate_tables(row_sums, col_sums):
        lp = log_prob(t)
        log_all.append(lp)
        if lp <= log_p_obs + 1e-10:
            log_extreme.append(lp)

    if not log_all: return 1.0, "exakt"
    p = float(np.exp(logsumexp(log_extreme) - logsumexp(log_all)))
    return min(p, 1.0), "exakt"

def expected_counts(table):
    table = np.array(table, dtype=float)
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    n = table.sum()
    return (row_sums @ col_sums) / n

def cramers_v(chi2, n, r, c):
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))

def effect_interpretation(v, r, c):
    k = min(r, c)
    if k == 2:
        thresholds = (0.1, 0.3, 0.5)
    elif k == 3:
        thresholds = (0.07, 0.21, 0.35)
    else:
        thresholds = (0.06, 0.17, 0.29)
    if v < thresholds[0]: return "vernachlässigbar"
    if v < thresholds[1]: return "schwach"
    if v < thresholds[2]: return "mittel"
    return "stark"

# ── Schritt 1: Tabellendesign ─────────────────────────────────────────────────
st.markdown("## 1 · Tabellengröße & Beschriftungen")

col1, col2, col3 = st.columns(3)
with col1:
    n_rows = st.selectbox("Anzahl Reihen", [2, 3, 4, 5], index=0,
                          help="Kategorien der Zeilenvariable")
with col2:
    n_cols = st.selectbox("Anzahl Spalten", [2, 3, 4, 5], index=0,
                          help="Kategorien der Spaltenvariable")
with col3:
    alpha = st.selectbox("Signifikanzniveau α", [0.05, 0.01, 0.001], index=0,
                         format_func=lambda x: f"{x}")

st.markdown("**Variablen-Bezeichnungen** *(optional)*")
lc1, lc2 = st.columns(2)
with lc1:
    row_var = st.text_input("Zeilenvariable", value="Gruppe", placeholder="z.B. Behandlung")
with lc2:
    col_var = st.text_input("Spaltenvariable", value="Ergebnis", placeholder="z.B. Outcome")

# Zeilen- und Spaltenbeschriftungen
st.markdown("**Kategorien-Namen** *(optional)*")
rc1, rc2 = st.columns(2)

row_labels = []
with rc1:
    st.markdown(f"*{row_var}*-Kategorien:")
    for i in range(n_rows):
        lbl = st.text_input(f"Reihe {i+1}", value=f"R{i+1}", key=f"rl_{i}",
                            label_visibility="collapsed")
        row_labels.append(lbl)

col_labels = []
with rc2:
    st.markdown(f"*{col_var}*-Kategorien:")
    for j in range(n_cols):
        lbl = st.text_input(f"Spalte {j+1}", value=f"S{j+1}", key=f"cl_{j}",
                            label_visibility="collapsed")
        col_labels.append(lbl)

# ── Schritt 2: Dateneingabe ───────────────────────────────────────────────────
st.markdown("## 2 · Beobachtete Häufigkeiten eingeben")

st.markdown("""
<div class="info-box">
Gib die <b>beobachteten Häufigkeiten</b> in die Tabelle ein.
Jede Zelle = Anzahl der Beobachtungen in dieser Kombination.
</div>
""", unsafe_allow_html=True)

# Jede Zelle als eigenes number_input – zuverlässig im Session State
table = np.zeros((n_rows, n_cols), dtype=int)

# Spaltenheader
header_cols = st.columns([1] + [2] * n_cols)
header_cols[0].markdown("")
for j, cl in enumerate(col_labels):
    header_cols[j+1].markdown(
        f"<div style='text-align:center;font-weight:600;font-family:IBM Plex Mono,monospace;"
        f"font-size:13px;color:#555;padding-bottom:4px;'>{cl}</div>",
        unsafe_allow_html=True)

# Reihen mit number_input
for i in range(n_rows):
    row_cols = st.columns([1] + [2] * n_cols)
    row_cols[0].markdown(
        f"<div style='display:flex;align-items:center;height:50px;"
        f"font-weight:600;font-family:IBM Plex Mono,monospace;font-size:13px;color:#555;'>"
        f"{row_labels[i]}</div>",
        unsafe_allow_html=True)
    for j in range(n_cols):
        key = f"cell_{n_rows}_{n_cols}_{i}_{j}"
        val = row_cols[j+1].number_input(
            f"{row_labels[i]} x {col_labels[j]}",
            min_value=0, value=0, step=1,
            key=key,
            label_visibility="collapsed"
        )
        table[i, j] = int(val)

n_total = table.sum()

# Randsummen anzeigen
st.markdown("**Randsummen:**")
margin_df = pd.DataFrame(table, index=row_labels, columns=col_labels)
margin_df["Summe"] = margin_df.sum(axis=1)
col_sums = pd.DataFrame([margin_df.sum()], index=["Summe"])
margin_df = pd.concat([margin_df, col_sums])
st.dataframe(margin_df.style.highlight_max(axis=None, subset=col_labels, color="#e8f5e9"),
             use_container_width=True)

# Validierung
if n_total == 0:
    st.markdown('<div class="warn-box">⚠️ Bitte gib Häufigkeiten ein.</div>',
                unsafe_allow_html=True)
    st.stop()

if np.any(table < 0):
    st.markdown('<div class="err-box">❌ Keine negativen Werte erlaubt.</div>',
                unsafe_allow_html=True)
    st.stop()

# ── Schritt 3: Testauswahl & Berechnung ──────────────────────────────────────
st.markdown("## 3 · Testauswahl & Ergebnisse")

# Erwartete Häufigkeiten berechnen
exp = expected_counts(table)
min_expected = exp.min()
cells_below_5 = (exp < 5).sum()
pct_below_5 = cells_below_5 / exp.size * 100

# Automatische Testempfehlung
is_2x2 = (n_rows == 2 and n_cols == 2)
recommend_fisher = (min_expected < 5) or (is_2x2 and n_total < 20)

if recommend_fisher:
    st.markdown(
        f'<div class="warn-box">⚠️ {cells_below_5} Zelle(n) ({pct_below_5:.0f}%) haben erwartete Häufigkeit < 5 '
        f'(Minimum: {min_expected:.2f}). <b>Fisher\'s exakter Test empfohlen.</b></div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="info-box">✅ Alle erwarteten Häufigkeiten ≥ 5 (Minimum: {min_expected:.2f}). '
        f'Chi-Quadrat-Test geeignet.</div>', unsafe_allow_html=True)

# Testauswahl
tc1, tc2 = st.columns(2)
with tc1:
    run_chi2 = st.checkbox("Chi-Quadrat-Test", value=True)
    if is_2x2:
        yates = st.checkbox("Yates-Korrektur (2×2)", value=recommend_fisher,
                            help="Empfohlen bei kleinen Stichproben in 2×2-Tabellen")
    else:
        yates = False
with tc2:
    run_fisher = st.checkbox(
        "Fisher's Exakter Test",
        value=recommend_fisher,
        help="Für 2×2 exakt. Für größere Tabellen: Monte-Carlo-Simulation (100.000 Permutationen)"
    )
    if not is_2x2 and run_fisher:
        st.markdown('<div class="info-box" style="font-size:12px;">ℹ️ Für >2×2: Monte-Carlo-Simulation</div>',
                    unsafe_allow_html=True)

if st.button("🔬 Berechnen"):
    st.session_state.run_analysis = True

if st.session_state.get("run_analysis"):

    results = {}

    # Chi-Quadrat
    if run_chi2:
        chi2_val, p_chi2, df, expected_vals = chi2_contingency(table, correction=yates)
        results["chi2"] = {
            "chi2": chi2_val, "p": p_chi2, "df": df,
            "expected": expected_vals, "yates": yates
        }

    # Fisher
    if run_fisher:
        with st.spinner("Fisher\'s Test wird berechnet..."):
            p_fisher, fisher_method = fisher_exact_rxc(table)
        results["fisher"] = {"p": p_fisher, "method": fisher_method}

    # Cramér's V
    chi2_for_v = results["chi2"]["chi2"] if "chi2" in results else \
        chi2_contingency(table, correction=False)[0]
    v = cramers_v(chi2_for_v, n_total, n_rows, n_cols)
    effect = effect_interpretation(v, n_rows, n_cols)

    # ── Ergebnis-Karten ───────────────────────────────────────────────────────
    st.markdown("### Ergebnisse")

    cards = []
    if "chi2" in results:
        r = results["chi2"]
        cards += [
            (f"χ²{'(Yates)' if yates else ''}", f"{r['chi2']:.4f}"),
            ("Freiheitsgrade (df)", str(r["df"])),
            ("p-Wert (χ²)", f"{r['p']:.4f}" if r['p'] >= 0.0001 else "< 0.0001"),
        ]
    if "fisher" in results:
        p_f = results["fisher"]["p"]
        fm = results["fisher"].get("method", "exakt")
        cards.append((f"p-Wert Fisher ({fm})", f"{p_f:.4f}" if p_f >= 0.0001 else "< 0.0001"))
    cards += [
        ("Cramér's V", f"{v:.4f}"),
        ("Effektstärke", effect),
        ("N (gesamt)", str(n_total)),
    ]

    cols_cards = st.columns(len(cards))
    for col, (lbl, val) in zip(cols_cards, cards):
        is_p = "p-Wert" in lbl
        p_val = None
        if is_p:
            try:
                p_val = float(val.replace("< ", ""))
            except Exception:
                p_val = 0.0001
        color_class = ""
        if is_p:
            color_class = "sig" if p_val < alpha else "nosig"
        col.markdown(
            f'<div class="param-card"><div class="param-label">{lbl}</div>'
            f'<div class="param-value {color_class}">{val}</div></div>',
            unsafe_allow_html=True)

    # Signifikanz-Fazit
    p_main = results.get("chi2", {}).get("p") or results.get("fisher", {}).get("p")
    if p_main is not None:
        if p_main < alpha:
            st.markdown(
                f'<div class="sig-box">✅ <b>Signifikant</b> (p = {p_main:.4f} < α = {alpha}): '
                f'Es besteht ein statistisch signifikanter Zusammenhang zwischen '
                f'<i>{row_var}</i> und <i>{col_var}</i>. '
                f'Effektstärke: Cramér\'s V = {v:.3f} ({effect}).</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="nosig-box">✗ <b>Nicht signifikant</b> (p = {p_main:.4f} ≥ α = {alpha}): '
                f'Kein statistisch signifikanter Zusammenhang zwischen '
                f'<i>{row_var}</i> und <i>{col_var}</i> nachweisbar.</div>',
                unsafe_allow_html=True)

    # ── Visualisierungen ──────────────────────────────────────────────────────
    st.markdown("### Visualisierungen")

    tab1, tab2, tab3 = st.tabs(["Balkendiagramm", "Mosaikplot", "Erwartete vs. Beobachtete"])

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#f5f4f0")
        ax.set_facecolor("white")
        x = np.arange(n_cols)
        width = 0.8 / n_rows
        colors = ["#1a1a1a", "#2563eb", "#16a34a", "#dc2626", "#9333ea"]
        for i in range(n_rows):
            offset = (i - n_rows/2 + 0.5) * width
            bars = ax.bar(x + offset, table[i], width, label=row_labels[i],
                         color=colors[i % len(colors)], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                           str(int(h)), ha="center", va="bottom", fontsize=9,
                           fontfamily="monospace")
        ax.set_xticks(x)
        ax.set_xticklabels(col_labels, fontfamily="monospace")
        ax.set_xlabel(col_var, fontsize=11, fontfamily="monospace")
        ax.set_ylabel("Häufigkeit", fontsize=11, fontfamily="monospace")
        ax.set_title(f"{row_var} × {col_var}", fontsize=12,
                    fontfamily="monospace", fontweight="bold")
        ax.legend(title=row_var, fontsize=9, title_fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        for sp in ax.spines.values(): sp.set_color("#e0e0e0")
        st.pyplot(fig); plt.close(fig)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor("#f5f4f0")
        ax2.set_facecolor("white")
        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)
        colors_mosaic = ["#1a1a1a", "#2563eb", "#16a34a", "#dc2626", "#9333ea"]
        x_pos = 0
        col_widths = col_sums / n_total
        for j in range(n_cols):
            y_pos = 0
            for i in range(n_rows):
                width_rect = col_widths[j]
                height_rect = table[i, j] / col_sums[j] if col_sums[j] > 0 else 0
                rect = mpatches.Rectangle(
                    (x_pos + 0.005, y_pos + 0.005),
                    width_rect - 0.01, height_rect - 0.01,
                    facecolor=colors_mosaic[i % len(colors_mosaic)],
                    alpha=0.8, linewidth=0)
                ax2.add_patch(rect)
                if height_rect > 0.05:
                    ax2.text(x_pos + width_rect/2,
                            y_pos + height_rect/2,
                            f"{table[i,j]}", ha="center", va="center",
                            fontsize=10, color="white", fontweight="bold",
                            fontfamily="monospace")
                y_pos += height_rect
            ax2.text(x_pos + width_rect/2, -0.04, col_labels[j],
                    ha="center", va="top", fontsize=10, fontfamily="monospace")
            x_pos += col_widths[j]

        legend_patches = [mpatches.Patch(color=colors_mosaic[i % len(colors_mosaic)],
                         label=row_labels[i], alpha=0.8) for i in range(n_rows)]
        ax2.legend(handles=legend_patches, title=row_var, loc="upper right",
                  fontsize=9, title_fontsize=9)
        ax2.set_xlim(0, 1); ax2.set_ylim(-0.1, 1.05)
        ax2.set_xlabel(col_var, fontsize=11, fontfamily="monospace")
        ax2.set_ylabel("Relative Häufigkeit", fontsize=11, fontfamily="monospace")
        ax2.set_title(f"Mosaikplot: {row_var} × {col_var}", fontsize=12,
                     fontfamily="monospace", fontweight="bold")
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        for sp in ax2.spines.values(): sp.set_color("#e0e0e0")
        st.pyplot(fig2); plt.close(fig2)

    with tab3:
        fig3, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig3.patch.set_facecolor("#f5f4f0")
        exp_rounded = exp.round(2)
        im1 = axes[0].imshow(table, cmap="Blues", aspect="auto")
        axes[0].set_xticks(range(n_cols)); axes[0].set_xticklabels(col_labels, fontsize=9)
        axes[0].set_yticks(range(n_rows)); axes[0].set_yticklabels(row_labels, fontsize=9)
        axes[0].set_title("Beobachtet", fontfamily="monospace", fontweight="bold")
        for i in range(n_rows):
            for j in range(n_cols):
                axes[0].text(j, i, str(table[i,j]), ha="center", va="center",
                           fontsize=11, fontfamily="monospace",
                           color="white" if table[i,j] > table.max()*0.6 else "black")
        im2 = axes[1].imshow(exp_rounded, cmap="Greens", aspect="auto")
        axes[1].set_xticks(range(n_cols)); axes[1].set_xticklabels(col_labels, fontsize=9)
        axes[1].set_yticks(range(n_rows)); axes[1].set_yticklabels(row_labels, fontsize=9)
        axes[1].set_title("Erwartet (H₀)", fontfamily="monospace", fontweight="bold")
        for i in range(n_rows):
            for j in range(n_cols):
                axes[1].text(j, i, f"{exp_rounded[i,j]:.1f}", ha="center", va="center",
                           fontsize=10, fontfamily="monospace",
                           color="white" if exp_rounded[i,j] > exp_rounded.max()*0.6 else "black")
        for ax in axes:
            ax.set_facecolor("white")
            for sp in ax.spines.values(): sp.set_color("#e0e0e0")
        fig3.tight_layout()
        st.pyplot(fig3); plt.close(fig3)

    # ── Detailtabellen ────────────────────────────────────────────────────────
    with st.expander("📋 Detailtabellen"):
        st.markdown("**Erwartete Häufigkeiten:**")
        exp_df = pd.DataFrame(exp.round(2), index=row_labels, columns=col_labels)
        st.dataframe(exp_df, use_container_width=True)

        st.markdown("**Residuen (beobachtet − erwartet):**")
        resid = table - exp
        resid_df = pd.DataFrame(resid.round(2), index=row_labels, columns=col_labels)
        st.dataframe(resid_df.style.background_gradient(cmap="RdYlGn", axis=None),
                    use_container_width=True)

        st.markdown("**Standardisierte Residuen:**")
        std_resid = resid / np.sqrt(exp)
        std_resid_df = pd.DataFrame(std_resid.round(3), index=row_labels, columns=col_labels)
        st.dataframe(std_resid_df.style.background_gradient(cmap="RdYlGn", axis=None),
                    use_container_width=True)

        if "chi2" in results:
            st.markdown("**Beitrag jeder Zelle zum χ²:**")
            contrib = (table - exp)**2 / exp
            contrib_df = pd.DataFrame(contrib.round(3), index=row_labels, columns=col_labels)
            st.dataframe(contrib_df.style.background_gradient(cmap="YlOrRd", axis=None),
                        use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("### Export")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(table, index=row_labels, columns=col_labels).to_excel(
            writer, sheet_name="Beobachtet")
        pd.DataFrame(exp.round(2), index=row_labels, columns=col_labels).to_excel(
            writer, sheet_name="Erwartet")
        pd.DataFrame(std_resid.round(3), index=row_labels, columns=col_labels).to_excel(
            writer, sheet_name="Std. Residuen")

        summary_data = {
            "Test": [], "Statistik": [], "Wert": []
        }
        if "chi2" in results:
            r = results["chi2"]
            summary_data["Test"]      += ["Chi-Quadrat", "Chi-Quadrat", "Chi-Quadrat"]
            summary_data["Statistik"] += [f"χ²{'(Yates)' if yates else ''}", "df", "p-Wert"]
            summary_data["Wert"]      += [round(r["chi2"], 4), r["df"], round(r["p"], 6)]
        if "fisher" in results:
            summary_data["Test"]      += ["Fisher"]
            summary_data["Statistik"] += ["p-Wert"]
            summary_data["Wert"]      += [round(results["fisher"]["p"], 6)]
        summary_data["Test"]      += ["Effekt", "Effekt", "Allgemein"]
        summary_data["Statistik"] += ["Cramér's V", "Interpretation", "N"]
        summary_data["Wert"]      += [round(v, 4), effect, n_total]

        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Ergebnisse", index=False)

    st.download_button(
        "📥 Ergebnisse als Excel herunterladen",
        buf.getvalue(), "kontingenztest.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.markdown(
    "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#aaa;'>"
    "Kontingenztest · Chi-Quadrat & Fisher's Exakter Test · Nur für Forschungszwecke</p>",
    unsafe_allow_html=True)
