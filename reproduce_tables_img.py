import numpy as np
import os
import matplotlib.pyplot as plt

# ==============================================================================
# 1. ATTRIBUTE FILES 
# ==============================================================================
USER_ACTIVENESS_PATH = "user_activeness.npy"
USER_MAINSTREAM_PATH = "user_mainstream.npy"
ITEM_POPULARITY_PATH = "item_popularity.npy"
ITEM_MAINSTREAM_PATH = "item_mainstream.npy"

# ==============================================================================
# 2. MODEL RESULTS CONFIGURATION
# ==============================================================================
MODEL_FILES = {
    "MF": {
        "user": r"data/recsys_data/amazon_cds/bias_scores/MF_scores/20260114-1514_mf_scores.npy",
        "item": r"data/recsys_data/amazon_cds/bias_scores/MF_scores/20260114-1514_ITEM_scores.npy" 
    },
    "CFAdaBoost": {
        "user": r"data/recsys_data/amazon_cds/bias_scores/MF_CFadaboost_scores/20260114-1526_boost_scores.npy",
        "item": r"data/recsys_data/amazon_cds/bias_scores/MF_CFadaboost_scores/20260114-1526_ITEM_scores.npy"
    },
    "CFBoost": {
        "user": r"data/recsys_data/amazon_cds/bias_scores/MF_adaboost_scores/20260114-1515_boost_scores.npy",
        "item": r"data/recsys_data/amazon_cds/bias_scores/MF_adaboost_scores/20260114-1515_ITEM_scores.npy"
    }
}

# ==============================================================================
# DATA LOADING HELPERS
# ==============================================================================

def load_user_ndcg(file_path):
    if not os.path.exists(file_path): return None
    try:
        data = np.load(file_path, allow_pickle=True)
        if data.shape == (): data = data.item()
        keys = list(data.keys())
        target_key = next((k for k in keys if "NDCG" in k and "20" in k), None)
        return np.array(data[target_key]) if target_key else None
    except: return None

def load_item_mdg(file_path):
    if not os.path.exists(file_path): return None
    try:
        data = np.load(file_path, allow_pickle=True)
        return data 
    except: return None

# ==============================================================================
# TABLE GENERATION LOGIC
# ==============================================================================

def calculate_table_data(attribute_path, models_dict, metric_type="user"):
    """
    Calculates the raw data for the table.
    Returns: (column_labels, row_data)
    """
    if not os.path.exists(attribute_path):
        print(f"[Error] Attribute file not found: {attribute_path}")
        return [], []

    # 1. Load Attribute and Sort
    attr_scores = np.load(attribute_path, allow_pickle=True)
    if len(attr_scores.shape) > 1: attr_scores = attr_scores.flatten()
    sorted_indices = np.argsort(attr_scores)
    
    # 2. Split into 5 Groups
    grouped_indices = np.array_split(sorted_indices, 5)
    
    # 3. Define Columns
    col_labels = ["Model", "Low", "Med-Low", "Medium", "Med-High", "High"]
    table_rows = []

    # 4. Process Each Model
    for model_name, paths in models_dict.items():
        score_path = paths.get(metric_type)
        
        if metric_type == "user":
            scores = load_user_ndcg(score_path)
        else:
            scores = load_item_mdg(score_path)

        row_data = [model_name]

        if scores is None:
            row_data.extend(["MISSING"] * 5)
        else:
            # Calculate average for each group
            for group_idx in grouped_indices:
                valid_idx = group_idx[group_idx < len(scores)]
                if len(valid_idx) == 0:
                    row_data.append("0.0000")
                else:
                    group_vals = scores[valid_idx]
                    group_vals = group_vals[~np.isnan(group_vals)]
                    avg = np.mean(group_vals) if len(group_vals) > 0 else 0.0
                    row_data.append(f"{avg:.4f}")
        
        table_rows.append(row_data)
        
    return col_labels, table_rows

def save_table_image(title, col_labels, row_data, filename):
    """
    Renders the calculated data as a PNG image using Matplotlib.
    """
    if not row_data:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, len(row_data) * 0.8 + 1)) 
    
    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table
    the_table = ax.table(cellText=row_data,
                         colLabels=col_labels,
                         loc='center',
                         cellLoc='center')

    # Styling
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 1.8) # Adjust scale for padding

    # Bold the headers
    for (row, col), cell in the_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e') # Dark blue header
        elif col == 0:
            cell.set_text_props(weight='bold') # Bold model names
            cell.set_facecolor('#f2f2f2') # Light gray model column
        
        cell.set_edgecolor('white') # White grid lines

    # Add Title
    plt.title(title, y=0.95, fontsize=14, weight='bold')

    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Success! Saved image to: {filename}")

if __name__ == "__main__":
    print("Generating Table Images...")

    # --- TABLE 1: USER MAINSTREAM BIAS ---
    cols, rows = calculate_table_data(USER_MAINSTREAM_PATH, MODEL_FILES, metric_type="user")
    save_table_image("Table 1: User Mainstream Bias (NDCG@20)", cols, rows, "table_1_user_mainstream.png")
    
    # --- TABLE 2: USER ACTIVENESS BIAS ---
    cols, rows = calculate_table_data(USER_ACTIVENESS_PATH, MODEL_FILES, metric_type="user")
    save_table_image("Table 2: User Activeness Bias (NDCG@20)", cols, rows, "table_2_user_activeness.png")

    # --- TABLE 3: ITEM MAINSTREAM BIAS ---
    cols, rows = calculate_table_data(ITEM_MAINSTREAM_PATH, MODEL_FILES, metric_type="item")
    save_table_image("Table 3: Item Mainstream Bias (MDG@20)", cols, rows, "table_3_item_mainstream.png")

    # --- TABLE 4: ITEM POPULARITY BIAS ---
    cols, rows = calculate_table_data(ITEM_POPULARITY_PATH, MODEL_FILES, metric_type="item")
    save_table_image("Table 4: Item Popularity Bias (MDG@20)", cols, rows, "table_4_item_popularity.png")