import matplotlib.pyplot as plt
import numpy as np

# Extracted data
gpus = [4, 8, 16, 32, 64]

gpu_compute_times = [
    [15.850442, 14.9242788600922, 15.069414138794],
    [9.3063237667084, 7.9129476547241, 8.088312625885, 8.64621901512146, 8.762521982193],
    [3.5559196472168, 3.15093898773193, 3.8808422088623, 3.28382635116577, 3.15042901039124],
    [2.5669808387756, 2.66861066818237, 2.56411309242249, 2.56728110313416, 2.54063400268555],
    [2.25913485527039, 2.26628518104553, 2.2688813209534, 2.1850224494934, 2.3332438468933]
]

cpu_compute_times = [[900.1030270571786, 880.3735793771265, 890.0386085060064],
 [600.9127419139897, 550.3059343321102, 530.0076610351986, 510.2886920568808, 520.0091368095989],
 [400.8621408108934, 420.39069087587717, 410.20483835486147, 430.8350331477297, 405.2290254534384],
 [300.1226348788603, 280.0042488150073, 290.42344982508183, 270.0469708810977, 260.4461020771275],
 [250.6348747548712, 270.21937487504255, 240.5596002987042, 260.81324009050485, 255.09837312646323]]

# CPU counts as multiples of 128
cpu_counts = [128, 256, 512, 1024, 2048]

# Calculate the mean and standard deviation for compute times
mean_gpu_compute_times = [np.mean(times) for times in gpu_compute_times]
std_dev_gpu = [np.std(times) for times in gpu_compute_times]

mean_cpu_compute_times = [np.mean(times) for times in cpu_compute_times]
std_dev_cpu = [np.std(times) for times in cpu_compute_times]

# Create the bar chart
x = np.arange(len(gpus))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, mean_gpu_compute_times, width, yerr=std_dev_gpu, label='GPU Compute Time')
rects2 = ax.bar(x + width / 2, mean_cpu_compute_times, width, yerr=std_dev_cpu, label='CPU Compute Time')

# Set x-axis labels as GPU/CPU format
x_labels = [f'{gpu}/{cpu}' for i, (gpu, cpu) in enumerate(zip(gpus, cpu_counts))]
ax.set_xticks(x)
ax.set_xticklabels(x_labels)

ax.set_xlabel('GPUs / CPUs')
ax.set_ylabel('Compute Time (s)')
ax.set_title('Compute Time by GPU and CPU')
ax.legend()

plt.tight_layout()
plt.show()
