# utils.py
import cupy as cp
import numpy as np
from typing import Tuple, Optional


def read_features(feature_path: str) -> Optional[Tuple[np.ndarray, cp.ndarray]]:
    """
    Read face features from NPZ file and transfer them to GPU memory.
    
    Args:
        feature_path (str): Path to the NPZ file containing features
        
    Returns:
        tuple: (images_name, images_emb) where images_emb is on GPU
        or None if file cannot be read
    """
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        images_name = data["images_name"]
        # Transfer embeddings to GPU
        images_emb = cp.array(data["images_emb"])
        return images_name, images_emb
    except Exception as e:
        print(f"Error reading features: {e}")
        return None


def compare_encodings(encoding: cp.ndarray, encodings: cp.ndarray) -> Tuple[float, int]:
    """
    Compare face encodings using GPU-accelerated dot product.
    
    Args:
        encoding (cp.ndarray): Query face encoding
        encodings (cp.ndarray): Database of face encodings
        
    Returns:
        tuple: (best_score, best_match_index)
    """
    # Ensure inputs are on GPU
    if isinstance(encoding, np.ndarray):
        encoding = cp.array(encoding)
    if isinstance(encodings, np.ndarray):
        encodings = cp.array(encodings)
        
    # Compute similarities using GPU
    sims = cp.dot(encodings, encoding.T)
    pare_index = int(cp.argmax(sims).get())
    score = float(sims[pare_index].get())
    
    return score, pare_index