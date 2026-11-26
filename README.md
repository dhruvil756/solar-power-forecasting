Setup

Requires Python 3.9 or newer. Run these commands from the project root.

Windows (PowerShell or Command Prompt)

python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt

macOS / Linux

python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

Place Solar_Weather_Data_Processed.csv in data/raw/.

How to run

1. Notebooks (run in order)

Open each notebook in notebooks/ and run all cells from top to bottom: 01, then 02, then 03. Each notebook starts with its own setup cell, so it runs on its own once the data file is in place. Notebook 03 saves models/hourly_lookup.joblib and models/metrics.csv.

2. Check the prediction function

bash
python -c "from src.prediction import predict_power; print(predict_power(0), predict_power(11))"

Expected output: 0.0 for hour 0 (night) and a large positive value for hour 11 (midday).

3. Dashboard

streamlit run app.py

Open the Local URL that Streamlit prints