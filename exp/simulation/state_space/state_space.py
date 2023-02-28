import random
import numpy as np
import matplotlib.pyplot as plt

def line_through_points(x1, y1, x2, y2):
    # Calculate slope
    m = (y2 - y1) / (x2 - x1)
    # Calculate intercept
    b = y1 - m * x1
    # Return slope and intercept as tuple
    return m, b


def count_nodes_throgh_line(point1,point2):
    # calculate the line
    if point1[0] == point2[0]:
        nodes_num = abs(point2[1] - point1[1]) - 1
    elif point1[1] == point2[1]:
        nodes_num = abs(point2[0] - point1[0]) - 1
    else:
        k, b = line_through_points(*point1, *point2)
        nodes_num = 0
        for node in grid:
            x_bound_max = max(point1[0], point2[0])
            x_bound_min = min(point1[0], point2[0])

            y_bound_max = max(point1[1], point2[1])
            y_bound_min = min(point1[1], point2[1])
            if (x_bound_min < node[0] < x_bound_max) and (y_bound_min < node[1] < y_bound_max):
                y_pre = k * node[0] + b
                if y_pre == node[1]:
                    nodes_num += 1
    nodes_num += 2  # add the start point and ending point
    return nodes_num


# generate grid
grid_size = 5
grid = [(i, j) for i in range(grid_size) for j in range(grid_size)]

# randomly sample two points
point1, point2 = random.sample(grid, 2)
angle = np.rad2deg(np.arctan2(point2[1]-point1[1],
                              point2[0]-point1[0]))
nodes_num = count_nodes_throgh_line(point1,point2)
print("The number of nodes through the line:", nodes_num)
print("The angle:",angle)


#%%
grid_size = 5
# Define the grid
x = range(grid_size)
y = range(grid_size)

# Define the arrow endpoints
arrow1_start = point1
arrow1_end = point2
arrow2_start = [0, 0]
arrow2_end = [1, 1]

# Create the plot
fig, ax = plt.subplots()

# Plot the grid
for i in range(len(x)):
    for j in range(len(y)):
        ax.plot(x[i], y[j], 'o', color='white', markersize=5, mec='black', mew=2)

# Add the arrows to the plot
#ax.arrow(arrow1_start[0], arrow1_start[1], arrow1_end[0] - arrow1_start[0], arrow1_end[1] - arrow1_start[1],
#         head_width=0.2, head_length=0.2, fc='red', ec='red')
#ax.arrow(arrow2_start[0], arrow2_start[1], arrow2_end[0]-arrow2_start[0], arrow2_end[1]-arrow2_start[1],
#         head_width=0.2, head_length=0.2, fc='blue', ec='blue')

# Set the axis range and gridlines
ax.set_xlim([-0.5, 4.5])
ax.set_ylim([-0.5, 4.5])
ax.set_xticks(x)
ax.set_yticks(y)
ax.grid(True)

# Set the axis labels
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')

# Set the plot title
ax.set_title('5x5 Square Grid with Arrows')

plt.savefig("/home/dell/clash/demo.png", dpi=300)
# Show the plot
plt.show()

#%%
grid_size = 5
grid = [(i, j) for i in range(grid_size) for j in range(grid_size)]
ntrials = 10000
# randomly sample two points

ntrials_nodes_num = []
ntrials_angle = []

for n in range(ntrials):
    point1, point2 = random.sample(grid, 2)
    ntrials_angle.append(np.rad2deg(np.arctan2(point2[1]-point1[1],
                                  point2[0]-point1[0])))
    ntrials_nodes_num.append(count_nodes_throgh_line(point1,point2))


# sort the data by angles
angles_sorted, y_true_sorted = zip(*sorted(zip(ntrials_angle, ntrials_nodes_num)))
y_8fold = [np.cos(np.deg2rad(8*(angle-8))) for angle in angles_sorted]

# create the plot
fig, ax = plt.subplots()
plt.plot(angles_sorted, y_true_sorted, '-',label='state_space')
plt.plot(angles_sorted, y_8fold,label='8fold')

# set the x-axis label
plt.xlabel('Angles')
# set the y-axis label
plt.ylabel('number')

# set the x-axis tick labels to be in 45-degree increments
x_ticks = np.arange(-180, 181, 45)
x_ticklabels = [str(x) + '°' for x in x_ticks]
plt.xticks(x_ticks, x_ticklabels)

# move the legend outside the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2,fontsize=12)

# save the figure
plt.savefig("/mnt/workdir/DCM/tmp/8fold&state_space.png",dpi=300)

# show the plot
plt.show()
