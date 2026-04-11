import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from IsoCor import process

st.title("IsoCor Online")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
tracer = st.selectbox("Tracer", ["C", "N", "H", "O"])
purity_str = st.text_input("Purity (semicolon-separated)", "0.0;1.0")

if uploaded_file and st.button("Run"):
    # --- parse purity ---
    try:
        purity = [float(x) for x in purity_str.split(";")]
    except Exception as e:
        st.error(f"Purity inválido: {e}")
        st.stop()

    # --- carregar isotopes/metabolites/derivatives ---
    data_dir = os.path.join(os.path.dirname(__file__), "data")

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
                        formulas[parts[0]] = parts[1]
                        names.append(parts[0])
        return formulas, names

    isotop = load_iso(os.path.join(data_dir, "Isotopes.dat"))
    dict_form_meta, meta_list = load_db(os.path.join(data_dir, "Metabolites.dat"))
    dict_form_der, der_list   = load_db(os.path.join(data_dir, "Derivatives.dat"))

    # --- ler CSV e montar lista de (sample, meta_name, der_name, v_mes) ---
    df = pd.read_csv(uploaded_file)
    # esperado: col0=metabolite, col1=derivative, col2..N=amostras
    meta_col  = df.columns[0]
    der_col   = df.columns[1]
    sample_cols = list(df.columns[2:])

    rows = []
    errors = []

    for _, row in df.iterrows():
        meta_name = str(row[meta_col]).strip()
        der_name  = str(row[der_col]).strip() if str(row[der_col]).strip().lower() != "nan" else ""

        if meta_name not in dict_form_meta:
            errors.append(f"Metabolite '{meta_name}' não encontrado em Metabolites.dat")
            continue

        meta_form = dict_form_meta[meta_name]
        der_form  = dict_form_der.get(der_name, "")

        for sample in sample_cols:
            val = row[sample]
            # agrupa linhas consecutivas do mesmo metabolito na mesma amostra em v_mes
            # aqui assume que cada linha é um pico (m0, m1, m2...)
            pass

    # --- re-agrupar: cada (sample, meta_name, der_name) tem lista de picos ---
    # O CSV deve ter linhas consecutivas para m0, m1, m2... do mesmo metabolito
    # Detecta grupos por blocos: linhas com mesmo meta_name pertencem ao mesmo grupo

    results = []
    groups = df.groupby([meta_col, der_col], sort=False)

    for (meta_name, der_name), grp in groups:
        meta_name = str(meta_name).strip()
        der_name  = str(der_name).strip() if str(der_name).strip().lower() != "nan" else ""

        if meta_name not in dict_form_meta:
            st.warning(f"Metabolite '{meta_name}' não encontrado — pulado.")
            continue

        meta_form = dict_form_meta[meta_name]
        der_form  = dict_form_der.get(der_name, "")

        for sample in sample_cols:
            v_mes = grp[sample].apply(pd.to_numeric, errors="coerce").tolist()
            if all(pd.isna(v) for v in v_mes):
                continue
            v_mes = [v if not pd.isna(v) else 0.0 for v in v_mes]

            el_excluded = ""  # inclui correção de abundância natural do tracer
            calc_enr = True

            try:
                res = process(
                    isotop, v_mes, meta_form, der_form,
                    calc_enr, el_excluded, purity, tracer
                )
                if res.err:
                    st.warning(f"{sample} / {meta_name}: {res.err}")
                    continue
                for i, (mid_val, resid_val) in enumerate(zip(res.mid, res.residuum)):
                    row_out = {
                        "Sample": sample,
                        "Metabolite": meta_name,
                        "Derivative": der_name,
                        "Peak index": i,
                        "Isotopologue dist.": round(mid_val, 5),
                        "Residuum": round(resid_val, 5) if not np.isinf(resid_val) else "",
                    }
                    if calc_enr:
                        row_out["Mean enrichment"] = round(res.enr_calc, 4) if i == 0 else ""
                    results.append(row_out)
            except Exception as e:
                st.warning(f"Erro em {sample}/{meta_name}: {e}")

    if results:
        out_df = pd.DataFrame(results)
        st.success(f"Processado: {len(results)} linhas")
        st.dataframe(out_df.head(50))

        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download resultado CSV", csv_bytes, "isocor_result.csv", "text/csv")
    else:
        st.error("Nenhum resultado gerado. Verifique os dados e os arquivos .dat.")
