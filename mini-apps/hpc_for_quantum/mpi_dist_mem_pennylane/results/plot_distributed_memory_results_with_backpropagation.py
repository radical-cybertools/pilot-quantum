import matplotlib.pyplot as plt
import pandas as pd

data = [
    ["qubits", "GPUs", "Pilot-startup(p95)", "Total time to completion"],
    [30, 32, 4.6, 145.09246],
    [30, 64, 5, 96.27447878],
    [30, 128, 5, 66.28230513],
    [30, 256, 5, 48.09477112]
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
