# Viva Short Answers

## 1) What we are doing?

We are building an Urban Traffic Zone Analyser for Bangalore city traffic data.
The project has two goals:
- Segment traffic observations into meaningful zones such as Critical, High, Moderate, and Free Flow.
- Detect unusual traffic patterns (anomalies/hotspots) that may indicate abnormal congestion conditions.

In simple terms, we are using data-driven clustering to understand city traffic behavior and highlight risky situations.

## 2) What algorithm we are using and why?

We are using two unsupervised machine learning algorithms:

- K-Means clustering:
  - Used for traffic zone segmentation.
  - Chosen because it efficiently groups similar traffic records into structured clusters.
  - We evaluate cluster quality using metrics such as Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.

- DBSCAN clustering:
  - Used for anomaly/hotspot detection.
  - Chosen because it identifies dense regions and automatically marks sparse/unusual points as noise (label = -1).
  - This is useful for detecting abnormal traffic behavior without pre-labeled anomaly data.

Why both:
- K-Means gives clear zone partitioning.
- DBSCAN gives outlier awareness.
- Together, they provide better operational insight than using only one method.

## 3) Which dataset we are using?

We are using the Bangalore City Traffic Dataset from Kaggle:
- Source: https://www.kaggle.com/datasets/preethamgouda/banglore-city-traffic-dataset
- Local file used in project: data/Banglore_traffic_Dataset.csv

Dataset highlights:
- Around 8,936 records and 16 columns.
- Covers multiple Bangalore areas.
- Includes traffic and context features such as Traffic Volume, Average Speed, Congestion Level, Road Capacity Utilization, Travel Time Index, Incident Reports, Weather Conditions, Public Transport Usage, and Roadwork Activity.

## 4) What conclusion we reached?

Key conclusions from the project:
- K-Means successfully segments traffic patterns into interpretable zones.
- Zone labels derived from congestion profile make results easier for practical decision-making.
- DBSCAN identifies anomaly points that represent unusual traffic states and potential hotspots.
- Combining EDA + clustering + visual analytics gives a clearer city-level traffic understanding than raw tables.
- The system can support traffic monitoring, hotspot investigation, and planning interventions.

Overall conclusion:
The project demonstrates that unsupervised learning can effectively convert unlabeled traffic data into actionable zone intelligence and anomaly alerts.
