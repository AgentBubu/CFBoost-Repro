import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_BINS = 60  # The paper uses 60 subgroups

# Attributes
USER_MAIN = "user_mainstream.npy"
USER_ACT = "user_activeness.npy"
ITEM_MAIN = "item_mainstream.npy"
ITEM_POP = "item_popularity.npy"

# Models to Compare (Update paths!)
MODELS = {
    "MF": {
        "user": r"data/recsys_data/amazon_cds/bias_scores/MF_scores/20260114-1514_mf_scores.npy",
        "item": r"data/recsys_data/amazon_cds/bias_scores/MF_scores/20260114-1514_ITEM_scores.npy"
    },
    "CFBoost": {
        "user": r"data/recsys_data/amazon_cds/bias_scores/MF_adaboost_scores/20260114-1515_boost_scores.npy",
        "item": r"data/recsys_data/amazon_cds/bias_scores/MF_adaboost_scores/20260114-1515_ITEM_scores.npy"
    },
    "CFAdaBoost": {
        "user": r"data/recsys_data/amazon_cds/bias_scores/MF_CFadaboost_scores/20260114-1526_boost_scores.npy",
        "item": r"data/recsys_data/amazon_cds/bias_scores/MF_CFadaboost_scores/20260114-1526_ITEM_scores.npy"
    }
}

# ==============================================================================
# LOGIC
# ==============================================================================

def load_data(path, key="NDCG@20"):
    """Generic loader for User Dicts or Item Arrays"""
    if not os.path.exists(path): return None
    try:
        data = np.load(path, allow_pickle=True)
        # If Item array (flat scores)
        if isinstance(data, np.ndarray) and data.dtype == float:
            return data
        # If User Dictionary
        if data.shape == (): data = data.item()
        return np.array(data[key])
    except: return None

def get_binned_scores(attr_path, score_path, is_user=True):
    """Sorts data by attribute, splits into 60 bins, returns avg score per bin"""
    # 1. Load Attribute
    if not os.path.exists(attr_path): return None
    attr = np.load(attr_path, allow_pickle=True)
    if len(attr.shape) > 1: attr = attr.flatten()
    
    # 2. Load Scores
    scores = load_data(score_path, "NDCG@20" if is_user else None)
    if scores is None: return None
    
    # Sanity check lengths
    min_len = min(len(attr), len(scores))
    attr = attr[:min_len]
    scores = scores[:min_len]

    # 3. Sort and Split
    sorted_idx = np.argsort(attr)
    bins = np.array_split(sorted_idx, NUM_BINS)
    
    # 4. Calculate Averages
    bin_avgs = []
    for b in bins:
        vals = scores[b]
        # Remove NaNs (crucial for Item MDG)
        vals = vals[~np.isnan(vals)]
        avg = np.mean(vals) if len(vals) > 0 else 0.0
        bin_avgs.append(avg)
        
    return np.array(bin_avgs)

def plot_figure_4():
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # Configuration for the 4 subplots
    plot_configs = [
        {"title": "(a) User Mainstream", "attr": USER_MAIN, "is_user": True, "ylabel": "NDCG@20"},
        {"title": "(b) User Activeness", "attr": USER_ACT, "is_user": True, "ylabel": "NDCG@20"},
        {"title": "(c) Item Mainstream", "attr": ITEM_MAIN, "is_user": False, "ylabel": "MDG@20"},
        {"title": "(d) Item Popularity", "attr": ITEM_POP, "is_user": False, "ylabel": "MDG@20"},
    ]

    # Styles
    styles = {"MF": "bo-", "CFBoost": "m^-", "CFAdaBoost": "gx-"} # Blue circle, Magenta triangle

    for ax_idx, config in enumerate(plot_configs):
        ax = axes[ax_idx]
        
        # Calculate scores for all models first to determine "Wins"
        model_results = {}
        for name, files in MODELS.items():
            path = files["user"] if config["is_user"] else files["item"]
            y_vals = get_binned_scores(config["attr"], path, config["is_user"])
            if y_vals is not None:
                model_results[name] = y_vals

        # Calculate Wins (Who had the highest score in each bin?)
        if model_results:
            # Create matrix: rows=models, cols=bins
            names = list(model_results.keys())
            matrix = np.array([model_results[n] for n in names])
            
            # Argmax along axis 0 gives index of winner for each bin
            winners = np.argmax(matrix, axis=0)
            
            # Count wins
            win_counts = {n: 0 for n in names}
            for w in winners:
                win_counts[names[w]] += 1
        
        # Plotting
        for name, y_vals in model_results.items():
            count = win_counts.get(name, 0)
            label = f"{name}({count})" # e.g., CFBoost(44)
            style = styles.get(name, "k.-")
            
            ax.plot(range(NUM_BINS), y_vals, style, markersize=4, linewidth=1, label=label)

        ax.set_title(config["title"], y=-0.2, fontsize=14, fontweight='bold')
        ax.set_ylabel(config["ylabel"], fontsize=12)
        ax.set_xlabel("Groups (Sorted Low -> High)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    output_file = "figure_4_repro.png"
    plt.savefig(output_file, dpi=300)
    print(f"Success! Saved to {output_file}")

if __name__ == "__main__":
    plot_figure_4()