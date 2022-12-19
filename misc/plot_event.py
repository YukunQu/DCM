import pandas as pd
import matplotlib.pyplot as plt
from nilearn.plotting import plot_event


events = pd.read_csv(r'/mnt/workdir/DCM/BIDS/derivatives/Events/game1/cv_train1/sub-180/6fold/sub-180_task-game1_run-1_events.tsv',sep='\t')
plot_event(events, figsize=(15, 5))
plt.show()