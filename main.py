import src.load_data as load_data
import src.anomaly_detection as anomaly_detection
import src.correlation_check as correlation_check
import src.anomaly_summaries as anomaly_summaries
import src.visualisation as visualisation
import src.inject_attacks as inject_attacks
import src.time_segmentation as time_segmentation

df = load_data.load_data()

# Further steps change dataframe, copy made
df_copy = load_data.load_data()
# df_copy = inject_attacks.inject_attacks(df_copy) UNCOMMENT FOR ADDED ATTACKS

df_copy = time_segmentation.add_time_features(df_copy)
df_copy = anomaly_detection.find_all_anomalies(df_copy)
df_copy = correlation_check.correlation_check(df_copy)
anomaly_summaries.generate_anomaly_summary(df_copy)
df_copy = visualisation.plot_anomalies(df_copy)
