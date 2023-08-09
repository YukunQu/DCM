# RSA simulation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


# Set the default plots style
sns.set_theme(style="white")

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.left'] = True

mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['xtick.top'] = False
mpl.rcParams['xtick.direction'] = 'out'

mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = False
mpl.rcParams['ytick.direction'] = 'out'


# sampling angle pairs
def angle_dissimilarity(theta,phi=0):
    deltaTheta = np.arange(0,61,1)
    theta2 = theta+deltaTheta

    # get angle's activity
    a1 = np.cos(6*(np.deg2rad(theta)+phi))
    a2 = np.cos(6*(np.deg2rad(theta2)+phi))
    # calculate the activity dissimilarity of angle paris
    dissimilarity = np.abs(a1 - a2)
    return deltaTheta,dissimilarity

theta_list = np.arange(0, 60, 0.5)
deltaTheta_array = np.zeros(61)
dissimilarity_array = np.zeros((len(theta_list),61))

for i,theta in enumerate(theta_list):
    deltaTheta,dissimilarity = angle_dissimilarity(theta,phi=0)
    dissimilarity_array[i,:] = 1 - dissimilarity

mean_dissimilarity = np.mean(dissimilarity_array,axis=0)
std_dissimilarity = np.std(dissimilarity_array,axis=0)
se_dissimilarity = np.std(dissimilarity_array, axis=0) / np.sqrt(dissimilarity_array.shape[0])

# Plotting line plot
fig,ax = plt.subplots(figsize=(5,5))
plt.errorbar(deltaTheta, mean_dissimilarity, capsize=8, fmt='-', linewidth=3, color='#333333', alpha=0.9)
plt.xticks(np.arange(0, 361, 60))
plt.xlabel('Angle difference (degree)')
plt.ylabel('Pattern similarity')

# Add a red transparent rectangle shade to 0-15 and 45 ~60 of x-axis
plt.axvspan(0, 15, facecolor='#D9291D', alpha=0.5)
plt.axvspan(45, 60, facecolor='#D9291D', alpha=0.5)
# Add a grey transparent rectangle shade to 15-45 of x-axis
plt.axvspan(15, 45, facecolor='#898989', alpha=0.5)
plt.xticks([0,30,60],[0,30,60])
plt.yticks([],[])
plt.savefig('/mnt/workdir/DCM/ResulsuNMANMAt/paper/sf/sf2/RSA/RSA_angle_similarity_simulation.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set the default plots style
sns.set_theme(style="white")

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.left'] = True

mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['xtick.top'] = False
mpl.rcParams['xtick.direction'] = 'out'

mpl.rcParams['ytick.left'] = True
mpl.rcParams['ytick.right'] = False
mpl.rcParams['ytick.direction'] = 'out'

# sampling angle pairs
def angle_dissimilarity(theta, phi=0):
    deltaTheta = np.arange(0,60,3)
    theta2 = theta+deltaTheta

    # get angle's activity
    a1 = np.cos(6*(np.deg2rad(theta)+phi))
    a2 = np.cos(6*(np.deg2rad(theta2)+phi))
    # calculate the activity dissimilarity of angle paris
    dissimilarity = np.abs(a1 - a2)
    return deltaTheta,dissimilarity

theta_list = np.arange(0, 60, 0.5)
phi_values = np.arange(0, 61, 30)  # Phi values from 0 to 60
phi_values = [0]
fig, axs = plt.subplots(1, 1, figsize=(6, 6), sharey=True,sharex=True)  # Create 1x4 subplots

deltaTheta_array = np.zeros(21)
dissimilarity_array = np.zeros((len(theta_list),21))
for ax, phi in zip(axs, phi_values):
    for i,theta in enumerate(theta_list):
        deltaTheta,dissimilarity = angle_dissimilarity(theta,phi=phi)
        dissimilarity_array[i,:] = dissimilarity

    mean_dissimilarity = np.mean(dissimilarity_array,axis=0)
    std_dissimilarity = np.std(dissimilarity_array,axis=0)
    se_dissimilarity = np.std(dissimilarity_array, axis=0) / np.sqrt(dissimilarity_array.shape[0])

    # Plotting line plot
    ax.errorbar(deltaTheta, mean_dissimilarity,fmt='-', linewidth=4,color='cornflowerblue',alpha=0.6)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.axvspan(0, 15, facecolor='red', alpha=0.3)
    ax.axvspan(45, 60, facecolor='red', alpha=0.3)
    ax.axvspan(15, 45, facecolor='lightgrey', alpha=0.6)
    ax.set_title(f'Phi = {phi}')
axs[-1].set_xlabel('Angle difference (degree)')
# Add a title to all subplot's y-axis
fig.text(0.04, 0.5, 'Neural dissimilarity for grid like coding', va='center', rotation='vertical')
#plt.subplots_adjust(hspace=0.5)  # Adjust the spacing between subplots
#plt.savefig('/mnt/workdir/DCM/Result/paper/sf/sf2/RSA_angle_dissimilarity_simulation.pdf', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0)
plt.show()