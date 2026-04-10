import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import shutil
import re
from scipy import optimize

# ======================================================
# COPIA DO MOTOR IsoCor (SEM WX)
# ======================================================

class process:
    def __init__(self, data_iso, v_measured, meta_form, der_form,
                 calc_mean_enr, el_excluded, el_pur, el_cor):

        self.err = ""
        self.data = data_iso
        self.meta_form = meta_form
        self.der_form = der_form
        self.el_excluded = el_excluded
        self.el_cor = el_cor

        el_dict_meta = self.parse_formula(meta_form)
        el_dict_der = self.parse_formula(der_form)

        self.nAtom_cor = el_dict_meta[el_cor]
        correction_vector = self.calc_mdv(el_dict_meta, el_dict_der)

        m_size = len(v_measured)

        self.correction_matrix = np.zeros((m_size, self.nAtom_cor+1))

        for i in range(self.nAtom_cor+1):
            column = correction_vector[:m_size]
            for na in range(i):
                column = np.convolve(column, el_pur)[:m_size]
            self.correction_matrix[:,i] = column

        mid_ini = np.zeros(self.nAtom_cor+1)
        v_mes = np.array(v_measured)

        mid, _, _ = optimize.fmin_l_bfgs_b(
            lambda x: np.sum((v_mes - self.correction_matrix @ x)**2),
            mid_ini,
            approx_grad=True
        )

        self.mid = mid / np.sum(mid)

    def parse_formula(self, f):
        d = dict((el,0) for el in self.data.keys())
        for el,n in re.findall(r"([A-Z][a-z]*)([0-9]*)", f):
            d[el] += int(n) if n else 1
        return d

    def calc_mdv(self, el_dict_meta, el_dict_der):
        result = [1.]
        for el,n in el_dict_meta.items():
            if el != self.el_cor:
                for i in range(n):
                    result = np.convolve(result, self.data[el])
        return list(result)

# ======================================================
# STREAMLIT APP
# ======================================================

st.title("🧪 IsoCor Online (Headless)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file and st.button("Run"):

    df = pd.read_csv(uploaded_file)

    st.write("Input preview:")
    st.dataframe(df.head())

    # exemplo simples (você pode adaptar depois)
    v_measured = df.iloc[:, 2].dropna().values

    # isotopes dummy (substituir pelos seus .dat depois)
    isotopes = {
        "C": [0.989, 0.011],
        "H": [0.99985, 0.00015],
        "O": [0.9976, 0.0004, 0.002],
        "N": [0.9963, 0.0037]
    }

    try:
        result = process(
            isotopes,
            v_measured,
            "C6H12O6",
            "",
            True,
            "",
            [0,1],
            "C"
        )

        st.success("Processed!")

        st.write("MID:")
        st.write(result.mid)

    except Exception as e:
        st.error(str(e))
