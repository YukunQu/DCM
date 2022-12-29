# Plot results
sns.displot(eval_score)
plt.title('Distributions of correlations', size=18)
plt.ylabel('Occurance', size=18)
plt.xlabel('Spearmann correlation', size=18)
sns.despine()
plt.show()

# let's plot the voxels above the 99th percentile
threshold = np.percentile(eval_score, 99)
plot_img = new_img_like(tmp_img, RDM_brain)

cmap = RDMcolormapObject()

coords = range(-20, 40, 5)
fig = plt.figure(figsize=(12, 3))

display = plotting.plot_stat_map(
    plot_img, colorbar=True, cut_coords=coords,threshold=threshold,
    display_mode='z', draw_cross=False, figure=fig,
    title=f'Animal model evaluation', cmap=cmap,
    black_bg=False, annotate=False)
plt.show()