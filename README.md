# Urban Traffic Zone Analyser

Streamlit app for traffic zone clustering and anomaly detection on Bangalore city traffic data.

## Features
- Exploratory data analysis dashboard
- K-Means clustering for zone segmentation
- DBSCAN clustering for anomaly/hotspot detection
- PCA-based 2D and 3D visualizations
- Model quality metrics comparison

## Project Structure
- app.py: Streamlit application
- requirements.txt: Python dependencies
- data/Banglore_traffic_Dataset.csv: Input dataset
- Urban_Traffic_Zone_Analyser.ipynb: Notebook analysis workflow
- DATASET_INSTRUCTIONS.md: Dataset guidance

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Run the app (recommended on Windows):

   python -m streamlit run app.py

## Notes
- If `streamlit` command is not recognized on Windows, use `python -m streamlit`.
- Verify dataset license terms before public redistribution of the CSV.
