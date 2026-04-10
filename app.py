import streamlit as st
import pandas as pd
import subprocess
import os
import uuid
import shutil

st.set_page_config(page_title="IsoCor Online", layout="centered")
st.title("🧪 IsoCor Online Processor")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

run = st.button("▶ Run")

if uploaded_file and run:

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    ISOCOR_ORIG = os.path.join(BASE_DIR, "IsoCor.py")

    # ======================================================
    # CREATE WORKDIR
    # ======================================================
    run_id = str(uuid.uuid4())
    workdir = os.path.join(BASE_DIR, f"run_{run_id}")
    os.makedirs(workdir, exist_ok=True)

    csv_path = os.path.join(workdir, "input.csv")

    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("⚙️ Running pipeline...")

    # ======================================================
    # BEFORE (igual ao seu)
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
    # COPY .dat FILES
    # ======================================================
    for f in os.listdir(DATA_DIR):
        shutil.copy(os.path.join(DATA_DIR, f), workdir)

    # ======================================================
    # PATCH IsoCor (REMOVE WX + GUI)
    # ======================================================
    patched_path = os.path.join(workdir, "IsoCor_headless.py")

    with open(ISOCOR_ORIG, "r") as f:
        code = f.read()

    # REMOVE WX IMPORT
    code = code.replace("import wx, re, numpy", "import re, numpy")

    # REMOVE GUI EXECUTION
    code = code.replace("if __name__ == '__main__':\n    gui()", 
                        "if __name__ == '__main__':\n    print('Headless mode')")

    with open(patched_path, "w") as f:
        f.write(code)

    # ======================================================
    # RUN IsoCor (HEADLESS)
    # ======================================================
    result = subprocess.run(
        ["python", "IsoCor_headless.py"],
        cwd=workdir,
        capture_output=True,
        text=True
    )

    st.text(result.stdout)
    st.text(result.stderr)

    if result.returncode != 0:
        st.error("❌ IsoCor failed")
        st.stop()

    # ======================================================
    # CHECK OUTPUT
    # ======================================================
    res_file = isocor_file.replace("_isocor.txt", "_isocor_res.txt")

    if not os.path.exists(res_file):
        st.error("❌ Output not generated")
        st.stop()

    # ======================================================
    # AFTER (igual ao seu)
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
