import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def firing_pattern(x, y, scale, field_radius, orientation):
    # Create a hexagonal grid of firing field centers
    x_coords = np.arange(-scale, scale + 1)
    y_coords = np.arange(-scale, scale + 1)
    xx, yy = np.meshgrid(x_coords, y_coords)
    centers = np.column_stack((xx.flatten(), yy.flatten())).astype(np.float64) * scale
    centers[:, 0] += 0.5 * (centers[:, 1] % 2) * scale

    # Apply the rotation matrix to the centers
    rotation_matrix = np.array([[np.cos(orientation), -np.sin(orientation)],
                                [np.sin(orientation), np.cos(orientation)]])
    centers = centers @ rotation_matrix

    # Calculate the firing rate at each (x, y) position
    firing_rate = np.zeros_like(x)
    for center in centers:
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        firing_rate += np.exp(-distance ** 2 / (2 * field_radius ** 2))
    return firing_rate


# Define the parameters for the grid cell
scale = 5
field_radius = 0.8
orientation = np.pi / 4  # 30 degrees

# Create a grid of x and y coordinates
x = np.linspace(-scale * 2, scale * 2, 1000)
y = np.linspace(-scale * 2, scale * 2, 1000)
xx, yy = np.meshgrid(x, y)

# Compute the firing pattern of the grid cell
firing_rate = firing_pattern(xx, yy, scale, field_radius, orientation)

# Apply Gaussian smoothing to the firing rate data
sigma = 30
smoothed_firing_rate = gaussian_filter(firing_rate, sigma)


# Normalization function
def normalize_data(data, min_value, max_value):
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data * (max_value - min_value) + min_value


# Apply normalization to the smoothed firing rate data
min_value = 0
max_value = 1
normalized_smoothed_firing_rate = normalize_data(smoothed_firing_rate, min_value, max_value)
# Plot the firing pattern
plt.figure(figsize=(8, 8), dpi=300, frameon=False)
plt.pcolormesh(x, y, normalized_smoothed_firing_rate, cmap='jet', shading='auto', vmax=1.3, vmin=-0.5)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.title("Grid cell firing pattern")
plt.rcParams.update({'font.size': 7})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig("firing_pattern.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# Define the x-axis values (in radians)
x = np.linspace(0, 2*np.pi, 361)

# Define the amplitudes of the two cosine curves
amp = 0.3

# Generate the two cosine curves
y = amp * np.cos(6*x)

# Plot the two curves
plt.figure(figsize=(5, 1), dpi=300, frameon=False)
plt.plot(x, y,color='steelblue', linewidth=2)

# Set the x-tick labels
xticks = np.linspace(0, 2*np.pi, 3)
xtick_labels = [r'$\varphi$',  r'$\varphi+\pi$', r'$\varphi+2\pi$']
plt.xticks(xticks, xtick_labels)

# Add a red transparent rectangle shade
shade_min = np.pi/3 - np.pi/12
shade_max = np.pi/3 + np.pi/12
shade1 = plt.axvspan(shade_min, shade_max, facecolor='red', alpha=0.3)

# Add a grey transparent rectangle shade
shade_min = np.pi/3 + np.pi/12
shade_max = np.pi/3 + 3*np.pi/12
shade2 = plt.axvspan(shade_min, shade_max, facecolor='grey', alpha=0.3)

# Remove the y-tick labels
plt.yticks([])

# Set the plot title, labels, and legend
plt.xlabel('Angle (radians)')
plt.ylabel('Amplitude')

plt.rcParams.update({'font.size': 7})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig("hexagonal_modulation.png", dpi=300, bbox_inches='tight', transparent=True)
# Display the plot
plt.show()