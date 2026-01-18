import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
USER_MAIN_PATH = "user_mainstream.npy"
USER_ACTIVE_PATH = "user_activeness.npy"
ITEM_MAIN_PATH = "item_mainstream.npy"
ITEM_POP_PATH = "item_popularity.npy"
TRAIN_DATA_PATH = "data/recsys_data/amazon_cds/user_train_like.npy"

def generate_figure_1():
    print("Loading data...")
    
    # 1. Load Attributes
    if not all(os.path.exists(p) for p in [USER_MAIN_PATH, USER_ACTIVE_PATH, ITEM_MAIN_PATH, ITEM_POP_PATH, TRAIN_DATA_PATH]):
        print("Error: Some .npy files are missing. Did you run generate_missing_files.py?")
        return

    u_mainstream = np.load(USER_MAIN_PATH, allow_pickle=True)
    u_activeness = np.load(USER_ACTIVE_PATH, allow_pickle=True)
    i_mainstream = np.load(ITEM_MAIN_PATH, allow_pickle=True)
    i_popularity = np.load(ITEM_POP_PATH, allow_pickle=True)
    
    # 2. Calculate Data for Plot (c) - User's Average Item Popularity
    print("Calculating User Average Item Popularity for Plot (c)...")
    train_data = np.load(TRAIN_DATA_PATH, allow_pickle=True)
    
    u_avg_item_pop = []
    
    # train_data is an array of lists: index=user_id, value=[item_id1, item_id2...]
    for user_interactions in train_data:
        if len(user_interactions) > 0:
            # Get popularity scores for all items this user interacted with
            pops = i_popularity[user_interactions]
            avg_pop = np.mean(pops)
            u_avg_item_pop.append(avg_pop)
        else:
            u_avg_item_pop.append(0)
    
    u_avg_item_pop = np.array(u_avg_item_pop)

    # 3. Plotting
    print("Generating Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot settings to match paper style (Scatter dot size and transparency)
    dot_size = 10
    alpha = 0.6
    color = '#1f77b4' # Standard Matplotlib blue

    # --- Plot (a): User Mainstream vs Activeness ---
    axes[0].scatter(u_mainstream, u_activeness, s=dot_size, alpha=alpha, c=color)
    axes[0].set_xlabel("User Mainstreamness", fontsize=14)
    axes[0].set_ylabel("User Activeness", fontsize=14)
    axes[0].set_title("(a)", y=-0.2, fontsize=16, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # --- Plot (b): Item Mainstream vs Popularity ---
    axes[1].scatter(i_mainstream, i_popularity, s=dot_size, alpha=alpha, c=color)
    axes[1].set_xlabel("Item Mainstreamness", fontsize=14)
    axes[1].set_ylabel("Item Popularity", fontsize=14)
    axes[1].set_title("(b)", y=-0.2, fontsize=16, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    # --- Plot (c): User Mainstream vs Avg Popularity ---
    axes[2].scatter(u_mainstream, u_avg_item_pop, s=dot_size, alpha=alpha, c=color)
    axes[2].set_xlabel("User Mainstreamness", fontsize=14)
    axes[2].set_ylabel("Item Avg Popularity", fontsize=14)
    axes[2].set_title("(c)", y=-0.2, fontsize=16, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    
    output_file = "figure_1_repro.png"
    plt.savefig(output_file, dpi=300)
    print(f"Success! Figure saved to {output_file}")

if __name__ == "__main__":
    generate_figure_1()