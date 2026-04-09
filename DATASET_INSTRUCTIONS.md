# Dataset Download Instructions

## Bangalore City Traffic Dataset

**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/preethamgouda/banglore-city-traffic-dataset  

### Steps to Download

1. Go to https://www.kaggle.com/datasets/preethamgouda/banglore-city-traffic-dataset
2. Click the **Download** button (you may need a free Kaggle account).
3. Extract the downloaded zip file.
4. Place the file `Banglore_traffic_Dataset.csv` inside the `data/` folder of this project.

### Dataset Overview

- **Rows:** 8,936
- **Columns:** 16
- **Areas Covered:** Indiranagar, Whitefield, Koramangala, M.G. Road, Jayanagar, Hebbal, Yeshwanthpur, Electronic City
- **Features:** Traffic Volume, Average Speed, Travel Time Index, Congestion Level, Road Capacity Utilization, Incident Reports, Environmental Impact, Public Transport Usage, Traffic Signal Compliance, Parking Usage, Pedestrian and Cyclist Count, Weather Conditions, Roadwork and Construction Activity

### File Structure After Setup

```
ML_project/
├── app.py
├── requirements.txt
├── DATASET_INSTRUCTIONS.md
├── data/
│   └── Banglore_traffic_Dataset.csv   <-- Place the CSV here
└── Urban_Traffic_Zone_Analyser.ipynb
```
