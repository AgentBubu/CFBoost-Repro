import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

# CONFIGURATION
TRAIN_FILE = 'data/recsys_data/amazon_cds/user_train_like.npy'
TEST_FILE = 'data/recsys_data/amazon_cds/user_test_like.npy'
VAL_FILE = 'data/recsys_data/amazon_cds/user_vali_like.npy'

def generate_files():
    print(f"Loading {TRAIN_FILE}...")
    train_data = np.load(TRAIN_FILE, allow_pickle=True)
    
    num_users = len(train_data)
    
    max_item_id = 0
    for items in train_data:
        if len(items) > 0:
            max_item_id = max(max_item_id, max(items))
    num_items = max_item_id + 1
    
    print(f"Detected {num_users} users and {num_items} items.")

    # ==========================================
    # 1. GENERATE USER ACTIVENESS (Section 3.2)
    # ==========================================
    print("Generating User Activeness...")
    # Activeness = number of items a user interacted with in the training set
    user_activeness = np.array([len(items) for items in train_data], dtype=np.float32)
    np.save('user_activeness.npy', user_activeness)
    print("Saved 'user_activeness.npy'")

    # ==========================================
    # 2. GENERATE ITEM POPULARITY (Section 3.2)
    # ==========================================
    print("Generating Item Popularity...")
    # Popularity = number of times an item appears across all users
    item_popularity = np.zeros(num_items, dtype=np.float32)
    
    for items in train_data:
        for item_id in items:
            item_popularity[item_id] += 1
            
    np.save('item_popularity.npy', item_popularity)
    print("Saved 'item_popularity.npy'")

    # ==========================================
    # 3. GENERATE MAINSTREAMNESS (Appendix B)
    # ==========================================
    # The paper defines User Mainstreamness as the average Jaccard similarity 
    # to all other users. This is computationally heavy (O(N^2)).
    # I use Sparse Matrices to speed this up.
    
    print("Generating User/Item Mainstream scores (This might take a moment)...")
    
    rows = []
    cols = []
    for u_id, items in enumerate(train_data):
        for i_id in items:
            rows.append(u_id)
            cols.append(i_id)
            
    # Create CSR Matrix
    vals = np.ones(len(rows))
    interaction_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(num_users, num_items))
    
    # --- A. User Mainstreamness ---
    # Intersection count (A dot A.T)
    print("  - Calculating User-User Similarity...")
    intersection = interaction_matrix.dot(interaction_matrix.T)
    
    # Union = |A| + |B| - |A intersection B|
    # Get user lengths (diagonal of intersection or pre-calculated activeness)
    user_lens = user_activeness
    
    # Calculate Jaccard for users
    # I just need the average Jaccard for every user
    user_mainstream = np.zeros(num_users)
    
    # I iterate to save memory (full dense matrix would be too big)
    # Using a sample of users to estimate mainstreamness is standard practice if N is huge,
    # but for Amazon CDs (12k users), I can try to do it fully or by chunks.
    
    # OPTIMIZATION: Only calculate average overlap with the "Global Average User" 
    # or iterate. Let's do a proper calculation using sparse operations.
    
    # To avoid MemoryError on creating a 12000x12000 dense matrix, I calculate row by row
    for u in tqdm(range(num_users), desc="User Mainstreamness"):
        if user_lens[u] == 0:
            continue
            
        # Get one row of intersections (Sparse)
        u_intersections = intersection.getrow(u).toarray().flatten()
        
        # Calculate Union for this user vs all others
        # Union[v] = len[u] + len[v] - intersection[u, v]
        unions = user_lens[u] + user_lens - u_intersections
        
        # Jaccard = Intersection / Union
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            jaccards = u_intersections / unions
            jaccards[unions == 0] = 0
        
        # Formula (Eq 9 in Appendix): Average Jaccard sum / (N-1)
        # I exclude the self-similarity (index u) where jaccard is 1.0
        total_jaccard = np.sum(jaccards) - 1.0 # Remove self
        avg_jaccard = total_jaccard / (num_users - 1)
        user_mainstream[u] = avg_jaccard

    np.save('user_mainstream.npy', user_mainstream)
    print("Saved 'user_mainstream.npy'")

    # --- B. Item Mainstreamness ---
    # Similar logic but transposed (Item-Item similarity)
    print("  - Calculating Item-Item Similarity...")
    
    # Transpose matrix to be Item x User
    item_matrix = interaction_matrix.T
    item_intersection = item_matrix.dot(item_matrix.T)
    item_lens = item_popularity
    
    item_mainstream = np.zeros(num_items)
    
    for i in tqdm(range(num_items), desc="Item Mainstreamness"):
        if item_lens[i] == 0:
            continue
        
        i_intersections = item_intersection.getrow(i).toarray().flatten()
        unions = item_lens[i] + item_lens - i_intersections
        
        with np.errstate(divide='ignore', invalid='ignore'):
            jaccards = i_intersections / unions
            jaccards[unions == 0] = 0
            
        total_jaccard = np.sum(jaccards) - 1.0
        avg_jaccard = total_jaccard / (num_items - 1)
        item_mainstream[i] = avg_jaccard
        
    np.save('item_mainstream.npy', item_mainstream)
    print("Saved 'item_mainstream.npy'")

if __name__ == "__main__":
    generate_files()