import pandas as pd
import matplotlib.pyplot as plt

# Load tracking statistics
df = pd.read_csv("tracking_stats.csv")

# === Plot 1: Mean Optical Flow Tracking Error ===
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df['Frame'], df['MeanError'], marker='o', linestyle='-', color='royalblue', label='Mean Error')
ax1.set_title("Mean Optical Flow Tracking Error per Frame", fontsize=14)
ax1.set_xlabel("Frame Index", fontsize=12)
ax1.set_ylabel("Mean Error (pixels)", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()
fig1.tight_layout()
fig1.savefig("mean_tracking_error_graph.png")
plt.show()
plt.close(fig1)  #If the figure is closed

# === Plot 2: Tracked / Lost / Added Points Over Time ===
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df['Frame'], df['Tracked'], label='Tracked', color='seagreen', linewidth=2)
ax2.plot(df['Frame'], df['Lost'], label='Lost', color='crimson', linewidth=2)
ax2.plot(df['Frame'], df['Added'], label='Added', color='darkorange', linewidth=2)
ax2.set_title("Feature Point Tracking Dynamics per Frame", fontsize=14)
ax2.set_xlabel("Frame Index", fontsize=12)
ax2.set_ylabel("Number of Points", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()
fig2.tight_layout()
fig2.savefig("tracking_point_dynamics_graph.png")
plt.show()
plt.close(fig2)
