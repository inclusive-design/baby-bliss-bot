# python plot_trend.py <csv_filename>
# python plot_trend.py ./test_results/PCA_cosine_similarities_trend_24852.csv

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python PCA_cosine_similarities_trend.py <csv_filename>")
    sys.exit(1)

csv_filename = sys.argv[1]

data = pd.read_csv(csv_filename)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(data["num_components"], data["cosine_similarity"], marker="o", linestyle="-")

# Add titles and labels for clarity
plt.title("Similarity Trend vs. Number of Principal Components")
plt.xlabel("Number of Averaged Principal Components (PC-1 to PC-i)")
plt.ylabel("Cosine Similarity with Direct Average")
plt.grid(True)
plt.xticks(range(1, len(data) + 1))  # Ensure integer ticks on the x-axis

# Save the plot to a file
plot_filename = os.path.splitext(csv_filename)[0] + ".png"
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")

# Display the plot
plt.show()
