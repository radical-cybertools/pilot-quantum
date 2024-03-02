import matplotlib.pyplot as plt
import pandas as pd

data = [
    ["qubits", "GPUs", "Pilot-startup(p95)", "Total time to completion"],
    [36, 32, 5, 67.03901567],
    [36, 64, 5, 41.87906662],
    [36, 128, 5, 25.16715938],
    [36, 256, 5, 14.89548963]
]

# Convert the data into a DataFrame
df = pd.DataFrame(data[1:], columns=data[0])

# Calculate the total time (Pilot-startup + Total time to completion)
df['Total'] = df['Pilot-startup(p95)'] + df['Total time to completion']

# Plotting the stacked bar chart without legend and with horizontal x-axis labels
ax = df.plot(x='GPUs', y=['Pilot-startup(p95)', 'Total time to completion'], kind='bar', stacked=True, 
    xlabel='GPUs', ylabel='Time')
ax.set_xticklabels(df['GPUs'], rotation=0)  # Rotate x-axis labels to be horizontal
plt.show()
