import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from numpy.polynomial.legendre import Legendre
from numpy import polynomial as P


def task_space(t):
    mag = np.linalg.norm(t)
    t = t/mag
    return np.append(t, mag)

parser = argparse.ArgumentParser(description="Run the analysis script with customizable parameters.")
# Add arguments
parser.add_argument("--experiment", type=str, default="legeval", choices=["legeval", "legood", "OOD", "legtrain"], help="Specify the experiment name (default: OOD).")
parser.add_argument("--num_experiments", type=int, default=1, help="Specify the number of experiments")
parser.add_argument("--generate_figure", type=bool, default=True, help="Specify whether to generate figure")
args = parser.parse_args()

output_dir = f"runs/experiments/{args.experiment}"

agents = [entry.name for entry in os.scandir(output_dir) if entry.is_dir()]


figsize_per_plot = [8, 8]
N = args.num_experiments
# Calculate the grid size
cols = int(np.ceil(np.sqrt(N)))
rows = int(np.ceil(N / cols))

# Determine the figure size based on the grid and figsize_per_plot
figsize = (figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
fig, axes = plt.subplots(rows, cols, figsize=figsize)

# Flatten the axes for easy iteration (handles both 1D and 2D cases)
axes = axes.flatten() if N > 1 else [axes]

lines = []
labels = []
poly_crashed = []
poly_survived = [] 
for agent in agents:
    agent_type = agent.split('-')[0]
    agent_results = torch.load(f"{output_dir}/{agent}/results.pth")
    print(f"NUM_ENVS: {len(agent_results.keys())}")
    assert len(agent_results.keys()) >= N 
    MSE = []
    for trial in tqdm(range(1, N + 1)):
        gt = agent_results[trial]['trajectory'].cpu().numpy()
        time_alive = agent_results[trial]['time_alive']
        pose = agent_results[trial]['pose'].cpu().numpy()
        axes[trial - 1].plot(gt[:, 0], gt[:, 1], color='black', linewidth=2)
        line, = axes[trial - 1].plot(pose[:time_alive, 0], pose[:time_alive, 1], label=agent_type)
        axes[trial - 1].grid(True)
        if agent_results[trial]['crashed']:
            axes[trial - 1].plot(pose[time_alive, 0], pose[time_alive, 1], marker='x', color='red', linewidth=3)
            for spine in axes[trial - 1].spines.values():
                spine.set_edgecolor('red')  # Set the box color
                spine.set_linewidth(4)
            axes[trial - 1].set_facecolor((1, 0, 0, 0.1))
            
            if agent_type == 'CTRLSAC':
                poly_crashed.append(task_space(agent_results[trial]['trajectory_legendre'].coef))
        else:
            MSE.append(agent_results[trial]['MSE'].cpu().numpy())
            print(agent_results[trial]['MSE'].cpu().numpy())
            if agent_type == 'CTRLSAC':
                poly_survived.append(task_space(agent_results[trial]['trajectory_legendre'].coef))
        
        if trial == 1:    
            lines.append(line)
            labels.append(agent_type)
    MSE = np.array(MSE)
    print(f"{agent_type}_MSE: {MSE.mean(axis=1)}")
    
for i in range(N, len(axes)):
    axes[i].axis("off")

fig.legend(
    lines, labels,
    loc='lower right',  # Position at the bottom right of the figure
    fontsize=50,
    bbox_to_anchor=(0.95, 0.05),  # Adjust the position of the legend
    bbox_transform=fig.transFigure,
    frameon=True  # Add a border around the legend
)
plt.tight_layout()
if args.generate_figure:
    plt.savefig(f"{output_dir}/plot.png")


