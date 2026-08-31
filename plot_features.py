import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf


cfg = OmegaConf.load("config.yaml")
df = pd.read_csv(cfg.path)

plt.hist(df['equipment'], bins=50)
plt.savefig("equipment_histogram.png", dpi=300, bbox_inches="tight")
plt.close()

ax = plt.scatter(df['distance'], df['posted_rate'], s=4, alpha=0.3)
plt.xlabel("Distance", fontsize=12, fontweight="bold")
plt.ylabel("Posted rate", fontsize=12, fontweight="bold")
fig = ax.get_figure()
fig.savefig("distance_x_rate.png", dpi=300, bbox_inches="tight")
plt.close()

ax = plt.scatter(df["quote_signal"], df['posted_rate'], s=4, alpha=0.3)
plt.xlabel("Quote signal", fontsize=12, fontweight="bold")
plt.ylabel("Posted rate", fontsize=12, fontweight="bold")
fig = ax.get_figure()
fig.savefig("posted_rate_x_quote_signal.png", dpi=300, bbox_inches="tight")
plt.close()

ax = plt.scatter(df['distance'], df['quote_signal'], s=4, alpha=0.3)
plt.xlabel("Distance", fontsize=12, fontweight="bold")
plt.ylabel("Quote signal", fontsize=12, fontweight="bold")
fig = ax.get_figure()
fig.savefig("distance_x_quote_signal.png", dpi=300, bbox_inches="tight")
plt.close()

ax = plt.scatter(df['distance'], df['market_index'], s=4, alpha=0.3)
plt.xlabel("Distance", fontsize=12, fontweight="bold")
plt.ylabel("Market index", fontsize=12, fontweight="bold")
fig = ax.get_figure()
fig.savefig("distance_x_market_index.png", dpi=300, bbox_inches="tight")
plt.close()

ax = plt.scatter(df['distance'], df['weight'], s=4, alpha=0.3)
plt.xlabel("Distance", fontsize=12, fontweight="bold")
plt.ylabel("Weight", fontsize=12, fontweight="bold")
fig = ax.get_figure()
fig.savefig("distance_x_weight.png", dpi=300, bbox_inches="tight")
plt.close()

ax = df.boxplot(column='posted_rate', by='equipment')
fig = ax.get_figure()
fig.savefig("rate_x_equipment.png", dpi=300, bbox_inches="tight")
plt.close()

ax = sns.heatmap(df.corr(numeric_only=True), annot=True, annot_kws={"size": 8}, cmap='coolwarm')
fig = ax.get_figure()
fig.savefig("heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


