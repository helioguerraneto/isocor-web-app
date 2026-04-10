import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
import uuid
import shutil

st.set_page_config(page_title="IsoCor Online", layout="centered")

st.title("🧪 IsoCor Online Processor")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

tracer = st.selectbox("Tracer", ["C","N","H","O"])
purity = st.text_input("Purity", "[0.0,1.0]")

run = st.button("▶ Run")

if uploaded_file and run:

    # ======================================================
    # ISOLAMENTO
    # ======================================================
    run_id = str(uuid.uuid4())
    workdir = f"run_{run_id}"
    os.makedirs(workdir, exist_ok=True)

    csv_path = os.path.join(workdir, "input.csv")

    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("⚙️ Running pipeline...")

    # ======================================================
    # BEFORE
    # ======================================================
    dataset = pd.read_csv(csv_path)
    data = dataset.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

    samplenames = data.columns.tolist()
    data.columns = ["Intensity"] * len(samplenames)

    tempcolumn = [""] * len(data)
    tempcolumn[0] = samplenames[0]

    tempdata = dataset.iloc[:, :2].copy()
    tempdata["Intensity"] = data.iloc[:, 0]

    outputdataset = pd.concat(
        [pd.DataFrame({"Sample": tempcolumn}), tempdata],
        axis=1
    )

    for i in range(1, len(samplenames)):
        tempcolumn = [""] * len(data)
        tempcolumn[0] = samplenames[i]
        tempdata.iloc[:, 2] = data.iloc[:, i]

        tempdataset = pd.concat(
            [pd.DataFrame({"Sample": tempcolumn}), tempdata],
            axis=1
        )
        outputdataset = pd.concat([outputdataset, tempdataset])

    isocor_file = os.path.join(workdir, "input_isocor.txt")
    outputdataset.to_csv(isocor_file, sep="\t", index=False, header=False)

    # ======================================================
    # COPIA .dat
    # ======================================================
    for f in os.listdir("./data"):
        shutil.copy(os.path.join("data", f), workdir)

    # ======================================================
    # ISOCOR (ANTIGO)
    # ======================================================
    try:
        result = subprocess.run(
            ["python", "../IsoCor.py"],
            cwd=workdir,
            capture_output=True,
            text=True
        )
        
        st.text(result.stdout)
        st.text(result.stderr)
        
        if result.returncode != 0:
            st.error("IsoCor failed")
            st.stop()
    except Exception as e:
        st.error(f"IsoCor failed: {e}")
        st.stop()

    res_file = isocor_file.replace("_isocor.txt", "_isocor_res.txt")

    # ======================================================
    # AFTER
    # ======================================================
    corrected = pd.read_csv(res_file, sep="\t")

    mult = corrected.shape[0] // len(samplenames)
    tempoutput = corrected.iloc[:mult, [1, 2, 4]].copy()

    for j in range(2, len(samplenames)+1):
        start = (j-1)*mult
        end = j*mult
        col = corrected.iloc[start:end, 4].reset_index(drop=True)
        tempoutput = pd.concat([tempoutput, col], axis=1)

    final = pd.concat([
        tempoutput.iloc[:, :2],
        dataset.iloc[:, 1:2],
        tempoutput.iloc[:, 2:]
    ], axis=1)

    final.columns = list(final.columns[:3]) + samplenames

    output_file = os.path.join(workdir, "result.csv")
    final.to_csv(output_file, index=False)

    st.success("✅ Done!")

    with open(output_file, "rb") as f:
        st.download_button("📥 Download result", f, file_name="result.csv")
