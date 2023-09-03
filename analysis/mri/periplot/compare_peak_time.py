import numpy as np
from scipy.stats import ttest_ind

adults_peak_points = np.load(r'/mnt/data/DCM/derivatives/peri_event_analysis/distancne_adults_peak_times.npy')
adolescents_peak_points = np.load(r'/mnt/data/DCM/derivatives/peri_event_analysis/distance_adolescents_peak_times.npy')

print(adults_peak_points.mean())

print(adolescents_peak_points.mean())

t, p = ttest_ind(adults_peak_points, adolescents_peak_points)
print(t, p)