import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── import IsoCor process class ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from IsoCor import process

# ── helpers: load .dat files ─────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_iso(path):
    d = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                d[parts[0]] = [float(v) for v in parts[1:]]
    return d

def load_db(path):
    formulas, names = {}, []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    formulas[parts[0]] = parts[1] if len(parts) > 1 else ""
                    names.append(parts[0])
    return formulas, names

# ── BEFORE: CSV → isocor input structure ─────────────────────────────────────
def run_before(dataset: pd.DataFrame):
    """
    Replicates before.py exactly.
    Returns (outputdataset, samplenames, sum_table).
    outputdataset is the long-format table IsoCor expects.
    """
    data = dataset.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    samplenum   = data.shape[1]
    samplenames = data.columns.tolist()

    data.columns = ["Intensity"] * samplenum

    tempcolumn    = [""] * data.shape[0]
    tempcolumn[0] = samplenames[0]
    tempdata      = dataset.iloc[:, :2].copy()
    tempdata["Intensity"] = data.iloc[:, 0]

    outputdataset = pd.concat(
        [pd.DataFrame({"Sample": tempcolumn}), tempdata], axis=1
    )

    for i in range(1, samplenum):
        tempcolumn    = [""] * data.shape[0]
        tempcolumn[0] = samplenames[i]
        tempdata = tempdata.copy()
        tempdata.iloc[:, 2] = data.iloc[:, i]
        tempdataset = pd.concat(
            [pd.DataFrame({"Sample": tempcolumn}), tempdata], axis=1
        )
        outputdataset = pd.concat([outputdataset, tempdataset], axis=0)

    # totals
    compoundname_rows = dataset[
        dataset.iloc[:, 0].notna() & (dataset.iloc[:, 0] != "")
    ].index.tolist()
    end_rows = [idx - 1 for idx in compoundname_rows[1:]]
    end_rows.append(len(dataset) - 1)
    compoundnames = dataset.iloc[compoundname_rows, 0].astype(str).tolist()

    sum_table = pd.DataFrame(
        np.zeros((len(compoundnames), samplenum)),
        index=compoundnames,
        columns=samplenames,
    )
    for k in range(len(compoundname_rows)):
        start = compoundname_rows[k]
        end   = end_rows[k]
        sum_table.iloc[k, :] = data.iloc[start : end + 1, :].sum(axis=0).values

    return outputdataset, samplenames, sum_table

# ── ISOCOR BATCH: run correction on the long-format table ────────────────────
def run_isocor(outputdataset, isotop, dict_form_meta, dict_form_der,
               el_cor, purity, el_excluded, calc_enr):
    """
    Mimics what IsoCor.py does in batch mode.
    Reads outputdataset sequentially just like the original IsoCor file parser:
      - col0: sample name (non-empty only on first row of each metabolite block)
      - col1: metabolite name
      - col2: derivative name
      - col3: intensity value
    Returns a DataFrame equivalent to _isocor_res.txt.
    """
    rows = []
    sample_col = outputdataset.columns[0]
    meta_col   = outputdataset.columns[1]
    der_col    = outputdataset.columns[2]
    int_col    = outputdataset.columns[3]

    df = outputdataset.reset_index(drop=True)

    # Build list of (sample, meta, der, [v_mes]) by parsing sequentially
    # exactly as the original IsoCor cmd_parse_multiple does
    current_sample = ""
    jobs = []   # list of dicts: {sample, meta, der, v_mes}

    for _, row in df.iterrows():
        s_val = str(row[sample_col]).strip()
        m_val = str(row[meta_col]).strip()
        d_val = str(row[der_col]).strip()
        i_val = row[int_col]

        # new sample name on this row?
        if s_val and s_val.lower() != "nan":
            current_sample = s_val

        # parse intensity
        try:
            intensity = float(str(i_val).replace(",", "."))
        except (ValueError, TypeError):
            intensity = 0.0

        # new metabolite block starts when meta_name is non-empty
        if m_val and m_val.lower() != "nan":
            jobs.append({
                "sample": current_sample,
                "meta":   m_val,
                "der":    d_val if d_val.lower() != "nan" else "",
                "v_mes":  [intensity],
            })
        else:
            # continuation row — append intensity to current job
            if jobs:
                jobs[-1]["v_mes"].append(intensity)

    # now run process() on each job
    for job in jobs:
        sample    = job["sample"]
        meta_name = job["meta"]
        der_name  = job["der"]
        v_mes     = job["v_mes"]

        def append_error(msg):
            for i in range(len(v_mes)):
                rows.append({
                    "sample": sample, "metabolite": meta_name,
                    "derivative": der_name, "isotopologue": i,
                    "isotopologue_fraction": np.nan,
                    "residuum": np.nan, "mean_enrichment": np.nan,
                    "error": msg,
                })

        if meta_name not in dict_form_meta:
            append_error(f"metabolite '{meta_name}' not found in Metabolites.dat")
            continue

        meta_form = dict_form_meta[meta_name]
        der_form  = dict_form_der.get(der_name, "")

        try:
            res = process(
                isotop, v_mes, meta_form, der_form,
                calc_enr, el_excluded, purity, el_cor
            )
        except Exception as e:
            append_error(str(e))
            continue

        if res.err:
            append_error(res.err)
            continue

        total = sum(v_mes)
        for i, (mid_val, resid_val) in enumerate(zip(res.mid, res.residuum)):
            rows.append({
                "sample": sample, "metabolite": meta_name,
                "derivative": der_name, "isotopologue": i,
                "isotopologue_fraction": round(mid_val, 5),
                "residuum": round(resid_val / total, 5) if total else resid_val,
                "mean_enrichment": round(res.enr_calc, 4) if calc_enr and i == 0 else "",
                "error": "",
            })

    cols = ["sample", "metabolite", "derivative", "isotopologue",
            "isotopologue_fraction", "residuum", "mean_enrichment", "error"]
    return pd.DataFrame(rows, columns=cols)

# ── AFTER: isocor result → wide CSV ──────────────────────────────────────────
def run_after(dataset: pd.DataFrame, corrected: pd.DataFrame, samplenames: list):
    """
    Replicates after.py exactly, matching offline output format:
      col0: Metabolite  (name only on first row of each compound block, NaN elsewhere)
      col1: Peak index  (0,1,2,3,4...)
      col2: derivative column from original CSV (NaN for most metabolomics datasets)
      col3+: one column per sample
    """
    samplenum = len(samplenames)
    mult = corrected.shape[0] // samplenum

    # --- build the first-sample block: Metabolite + Peak index + fraction ---
    # Our corrected df columns: sample, metabolite, derivative, isotopologue,
    #                           isotopologue_fraction, residuum, mean_enrichment, error
    first_block = corrected.iloc[0:mult].copy().reset_index(drop=True)

    # Metabolite: name only on first row of each compound, NaN elsewhere
    # (replicates R / offline behavior)
    meta_col = first_block["metabolite"].copy()
    seen_meta = None
    for i in range(len(meta_col)):
        if meta_col.iloc[i] == seen_meta:
            meta_col.iloc[i] = np.nan
        else:
            seen_meta = meta_col.iloc[i]

    peak_index = first_block["isotopologue"].values          # 0,1,2,...
    fraction_s1 = first_block["isotopologue_fraction"].values

    tempoutput = pd.DataFrame({
        "Metabolite": meta_col.values,
        "Peak index": peak_index,
        samplenames[0]: fraction_s1,
    })

    # --- append remaining samples ---
    for j in range(1, samplenum):
        start = j * mult
        end   = (j + 1) * mult
        col = corrected.iloc[start:end]["isotopologue_fraction"].reset_index(drop=True)
        tempoutput[samplenames[j]] = col.values

    # --- insert derivative column (col2 of original dataset) ---
    der_col = dataset.iloc[:mult, 1].reset_index(drop=True)

    final = pd.concat([
        tempoutput.iloc[:, :2].reset_index(drop=True),   # Metabolite, Peak index
        der_col,                                          # derivative (Unnamed: 1 / NaN)
        tempoutput.iloc[:, 2:].reset_index(drop=True),   # sample columns
    ], axis=1)

    return final

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IsoCor Online", layout="centered")
st.title("IsoCor Online")
st.caption("Pipeline completo: CSV → before → IsoCor → after → resultado")

with st.expander("Configurações do tracer", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        tracer = st.selectbox("Isotopic tracer", ["C", "N", "H", "O", "S"])
    with col2:
        purity_str = st.text_input(
            "Purity (separado por ;)",
            "0.0;1.0",
            help="Ex. para 13C puro: 0.0;1.0 | Para 99% puro: 0.01;0.99",
        )
    col3, col4 = st.columns(2)
    with col3:
        correct_nat_ab = st.selectbox(
            "Corrigir ab. natural do tracer?", ["yes", "no"]
        )
    with col4:
        calc_enr = st.selectbox("Calcular mean enrichment?", ["yes", "no"])

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
plot_after = st.radio("Plot after correction?", ["No", "Yes"], horizontal=True)

if uploaded_file and st.button("▶ Run IsoCor", type="primary"):
    # validate purity
    try:
        purity = [float(x) for x in purity_str.split(";")]
        assert abs(sum(purity) - 1.0) < 1e-6, "A soma da purity deve ser 1.0"
    except Exception as e:
        st.error(f"Purity inválido: {e}")
        st.stop()

    el_excluded = tracer if correct_nat_ab == "no" else ""

    try:
        isotop        = load_iso(os.path.join(DATA_DIR, "Isotopes.dat"))
        dict_form_meta, _ = load_db(os.path.join(DATA_DIR, "Metabolites.dat"))
        dict_form_der, _  = load_db(os.path.join(DATA_DIR, "Derivatives.dat"))
    except FileNotFoundError as e:
        st.error(f"Arquivo .dat não encontrado em /data: {e}")
        st.stop()

    if tracer not in isotop:
        st.error(f"Tracer '{tracer}' não encontrado em Isotopes.dat. Disponíveis: {list(isotop.keys())}")
        st.stop()

    if len(purity) != len(isotop[tracer]):
        st.error(f"Purity deve ter {len(isotop[tracer])} valores para '{tracer}'.")
        st.stop()

    dataset = pd.read_csv(uploaded_file)

    with st.spinner("Rodando pipeline..."):
        # step 1: before
        outputdataset, samplenames, sum_table = run_before(dataset)

        # step 2: IsoCor
        corrected = run_isocor(
            outputdataset, isotop, dict_form_meta, dict_form_der,
            tracer, purity,
            el_excluded=el_excluded,
            calc_enr=(calc_enr == "yes"),
        )

        errors = corrected[corrected["error"] != ""]
        if not errors.empty:
            st.warning(f"{len(errors)} erros durante correção:")
            st.dataframe(errors[["metabolite", "derivative", "sample", "error"]].drop_duplicates())

        # step 3: after
        final = run_after(dataset, corrected, samplenames)

    st.success("Pronto!")

    tab1, tab2, tab3 = st.tabs(["Resultado (wide)", "Totals", "IsoCor raw"])
    with tab1:
        st.dataframe(final)
        st.download_button(
            "⬇ Download _res.csv",
            final.to_csv(index=False).encode("utf-8"),
            file_name=uploaded_file.name.replace(".csv", "_res.csv"),
            mime="text/csv",
        )
    with tab2:
        st.dataframe(sum_table)
        st.download_button(
            "⬇ Download _totals.csv",
            sum_table.to_csv(index=True).encode("utf-8"),
            file_name=uploaded_file.name.replace(".csv", "_totals.csv"),
            mime="text/csv",
        )
    with tab3:
        st.dataframe(corrected)
        st.download_button(
            "⬇ Download _isocor_res.txt",
            corrected.to_csv(index=False, sep="\t").encode("utf-8"),
            file_name=uploaded_file.name.replace(".csv", "_isocor_res.txt"),
            mime="text/plain",
        )

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    if plot_after == "Yes":
        st.markdown("---")
        st.subheader("Isotopologue distribution")

        # build long-form from corrected: one row per (sample, metabolite, isotopologue)
        plot_df = corrected[corrected["error"] == ""][
            ["sample", "metabolite", "isotopologue", "isotopologue_fraction"]
        ].copy()
        plot_df["isotopologue_fraction"] = pd.to_numeric(
            plot_df["isotopologue_fraction"], errors="coerce"
        ).fillna(0)

        metabolites = plot_df["metabolite"].unique().tolist()

        for meta in metabolites:
            sub = plot_df[plot_df["metabolite"] == meta]
            pivot = sub.pivot_table(
                index="sample",
                columns="isotopologue",
                values="isotopologue_fraction",
                aggfunc="mean",
            ).reindex(columns=sorted(sub["isotopologue"].unique()))

            # keep sample order from original file
            pivot = pivot.reindex([s for s in samplenames if s in pivot.index])

            n_iso = pivot.shape[1]
            colors = cm.tab10.colors[:n_iso] if n_iso <= 10 else cm.tab20.colors[:n_iso]

            fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.35), 4))
            bottom = np.zeros(len(pivot))
            for idx, iso in enumerate(pivot.columns):
                vals = pivot[iso].fillna(0).values
                ax.bar(range(len(pivot)), vals, bottom=bottom,
                       color=colors[idx], label=f"m+{iso}", width=0.8)
                bottom += vals

            ax.set_xticks(range(len(pivot)))
            ax.set_xticklabels(pivot.index, rotation=90, fontsize=7)
            ax.set_ylabel("Isotopologue fraction")
            ax.set_ylim(0, 1)
            ax.set_title(meta)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7,
                      ncol=max(1, n_iso // 5))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
