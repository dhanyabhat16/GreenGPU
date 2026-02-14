"""
GreenGPU – IMDB Test Dataset Semantic De-duplication Engine

This module performs semantic redundancy removal on the IMDB test dataset.
It integrates into the GreenGPU sustainability optimization pipeline with
memory-conscious and efficient implementation.

Key features:
- Semantic duplicate detection using transformer embeddings
- FAISS-based k-NN search (avoids O(N²) memory)
- Connected component clustering
- Class distribution preservation
- Safety guardrails
"""

import os
from typing import List, Dict, Tuple, Set
from collections import defaultdict, deque
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class IMDBTextDeduplicator:
    """
    Semantic de-duplication engine for IMDB test dataset.
    
    Removes semantically redundant reviews using transformer embeddings
    and FAISS similarity search, while preserving class distribution.
    """
    
    MIN_DATASET_SIZE = 200
    MAX_REMOVAL_PERCENTAGE = 0.80
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_THRESHOLD = 0.80
    BATCH_SIZE = 32
    K_NEIGHBORS = 40
    
    def __init__(self, dataset_path: str, threshold: float = 0.80):
        """
        Initialize the deduplicator.
        
        Args:
            dataset_path: Path to dataset root (e.g., "dataset_v1/dataset/test")
            threshold: Cosine similarity threshold for duplicate detection (0.0-1.0)
        
        Raises:
            ValueError: If threshold is not in valid range [0, 1]
            FileNotFoundError: If dataset path does not exist
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        self.dataset_path = dataset_path
        self.threshold = threshold
        self.model = None
        self.texts: List[str] = []
        self.labels: List[int] = []
        
    def _load_test_dataset(self) -> None:
        """
        Load all .txt files from test/pos and test/neg directories.
        Prints progress every 1000 files.
        """
        pos_dir = os.path.join(self.dataset_path, "pos")
        neg_dir = os.path.join(self.dataset_path, "neg")

        file_counter = 0

        # ---------------- POSITIVE ----------------
        if os.path.isdir(pos_dir):
            print("\nLoading POSITIVE reviews...")
            for filename in sorted(os.listdir(pos_dir)):
                if filename.endswith(".txt"):
                    filepath = os.path.join(pos_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read().strip()
                            if text:
                                self.texts.append(text)
                                self.labels.append(1)

                        file_counter += 1
                        if file_counter % 1000 == 0:
                            print(f"  Processed {file_counter} files (last type: POS)")

                    except Exception as e:
                        print(f"Warning: Could not read {filepath}: {e}")

        # ---------------- NEGATIVE ----------------
        if os.path.isdir(neg_dir):
            print("\nLoading NEGATIVE reviews...")
            for filename in sorted(os.listdir(neg_dir)):
                if filename.endswith(".txt"):
                    filepath = os.path.join(neg_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read().strip()
                            if text:
                                self.texts.append(text)
                                self.labels.append(0)

                        file_counter += 1
                        if file_counter % 1000 == 0:
                            print(f"  Processed {file_counter} files (last type: NEG)")

                    except Exception as e:
                        print(f"Warning: Could not read {filepath}: {e}")

        print(f"\nFinished loading. Total files processed: {file_counter}")
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using sentence-transformers model.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            numpy array of shape (len(texts), embedding_dim) with L2-normalized embeddings
        """
        if self.model is None:
            print(f"Loading model: {self.DEFAULT_MODEL}")
            self.model = SentenceTransformer(self.DEFAULT_MODEL)
        
        print(f"Generating embeddings for {len(texts)} texts (batch_size={self.BATCH_SIZE})...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # L2 normalize for cosine similarity
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        return embeddings
    
    def _build_similarity_graph(self, embeddings: np.ndarray) -> Dict[int, Set[int]]:
        """
        Build undirected similarity graph using FAISS k-NN search.
        
        Uses IndexFlatIP (inner product) for cosine similarity on normalized embeddings.
        Adds edges when cosine_similarity(i, j) > threshold.
        
        Args:
            embeddings: numpy array of L2-normalized embeddings (N x D)
        
        Returns:
            Adjacency list: dict[int, set[int]] representing undirected graph
        """
        n_samples = embeddings.shape[0]
        print(f"\nBuilding similarity graph with k={self.K_NEIGHBORS}, threshold={self.threshold}...")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))
        
        # Initialize adjacency list
        graph: Dict[int, Set[int]] = defaultdict(set)
        
        # Search k-nearest neighbors
        distances, indices = index.search(embeddings.astype(np.float32), self.K_NEIGHBORS + 1)
        
        # Build graph from similarity results
        for i in range(n_samples):
            for dist, j in zip(distances[i], indices[i]):
                if i != j and dist > self.threshold:  # Exclude self-loops and below-threshold
                    # Add undirected edge (avoid duplicates by checking both directions)
                    if j not in graph[i]:
                        graph[i].add(j)
                        graph[j].add(i)
        
        n_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
        print(f"Graph built: {len(graph)} nodes with edges, {n_edges} total edges")
        
        return graph
    
    def _find_connected_components(self, graph: Dict[int, Set[int]], 
                                   n_total: int) -> List[List[int]]:
        """
        Find connected components in the similarity graph using BFS.
        
        Args:
            graph: Adjacency list representation of the graph
            n_total: Total number of samples (includes isolated nodes)
        
        Returns:
            List of components, where each component is a list of node indices
        """
        visited = set()
        components = []
        
        for start in range(n_total):
            if start in visited:
                continue
            
            # BFS to find component
            component = []
            queue = deque([start])
            visited.add(start)
            
            while queue:
                node = queue.popleft()
                component.append(node)
                
                # Explore neighbors (if node has any)
                if node in graph:
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            components.append(sorted(component))
        
        print(f"Found {len(components)} connected components")
        
        return components
    
    def _select_representatives(self, components: List[List[int]]) -> Set[int]:
        """
        Select representative indices for each component.
        
        For each component, selects the smallest index as the representative.
        All other indices in the component are marked for removal.
        
        Args:
            components: List of connected components
        
        Returns:
            Set of representative indices to keep
        """
        representatives = set()
        removed_count = 0
        
        for component in components:
            # Select smallest index as representative
            representative = min(component)
            representatives.add(representative)
            removed_count += len(component) - 1
        
        print(f"Selected {len(representatives)} representatives, marking {removed_count} for removal")
        
        return representatives
    
    def _compute_distribution_shift(self, labels: List[int], 
                                   selected_indices: Set[int]) -> float:
        """
        Compute maximum class distribution shift after deduplication.
        
        Calculates percentage shift for each class:
            shift = abs(original_count - new_count) / original_count
        
        Args:
            labels: Original labels list
            selected_indices: Indices of samples to keep
        
        Returns:
            Maximum percentage shift across all classes
        """
        original_counts = defaultdict(int)
        new_counts = defaultdict(int)
        
        for i, label in enumerate(labels):
            original_counts[label] += 1
            if i in selected_indices:
                new_counts[label] += 1
        
        max_shift = 0.0
        for label in original_counts:
            shift = abs(original_counts[label] - new_counts[label]) / original_counts[label]
            max_shift = max(max_shift, shift)
            print(f"  Class {label}: {original_counts[label]} ->> {new_counts[label]} "
                  f"(shift: {shift*100:.2f}%)")
        
        return max_shift
    
    def deduplicate(self) -> Dict:
        """
        Perform semantic de-duplication on the test dataset.
        
        Process:
        1. Load dataset from pos/ and neg/ directories
        2. Check safety constraints (min size, max removal)
        3. Generate embeddings
        4. Build similarity graph
        5. Find connected components
        6. Select representatives
        7. Compute distribution shift
        
        Returns:
            Dictionary with deduplication results:
            {
                "original_size": int,
                "reduced_size": int,
                "reduction_percentage": float,
                "max_class_shift": float,
                "threshold_used": float,
                "description": str
            }
        """
        print("=" * 70)
        print("IMDB Test Dataset Semantic De-duplication")
        print("=" * 70)
        
        # Step 1: Load dataset
        print("\n[Step 1] Loading dataset...")
        self._load_test_dataset()
        original_size = len(self.texts)
        print(f"Loaded {original_size} samples")
        
        if original_size == 0:
            raise ValueError("No data found in dataset directories")
        
        # Step 2: Check safety constraints
        print(f"\n[Step 2] Safety checks...")
        if original_size < self.MIN_DATASET_SIZE:
            print(f"Dataset size ({original_size}) < minimum ({self.MIN_DATASET_SIZE})")
            print("Skipping deduplication to preserve data integrity")
            return {
                "original_size": original_size,
                "reduced_size": original_size,
                "reduction_percentage": 0.0,
                "max_class_shift": 0.0,
                "threshold_used": self.threshold,
                "description": f"Dataset too small ({original_size} < {self.MIN_DATASET_SIZE}). "
                               "Deduplication skipped."
            }
        
        # Step 3: Generate embeddings
        print(f"\n[Step 3] Generating embeddings...")
        embeddings = self._generate_embeddings(self.texts)
        
        # Step 4: Build similarity graph
        print(f"\n[Step 4] Building similarity graph...")
        graph = self._build_similarity_graph(embeddings)
        
        # Step 5: Find connected components
        print(f"\n[Step 5] Finding connected components...")
        components = self._find_connected_components(graph, original_size)
        
        # Step 6: Select representatives
        print(f"\n[Step 6] Selecting representatives...")
        representatives = self._select_representatives(components)
        
        # Calculate removal statistics
        removal_count = original_size - len(representatives)
        removal_percentage = removal_count / original_size
        
        # Step 7: Check removal cap
        print(f"\n[Step 7] Checking removal cap...")
        if removal_percentage > self.MAX_REMOVAL_PERCENTAGE:
            print(f"Removal ({removal_percentage*100:.2f}%) exceeds cap ({self.MAX_REMOVAL_PERCENTAGE*100:.1f}%)")
            print("Aborting to preserve data integrity")
            return {
                "original_size": original_size,
                "reduced_size": original_size,
                "reduction_percentage": 0.0,
                "max_class_shift": 0.0,
                "threshold_used": self.threshold,
                "description": f"Proposed removal ({removal_percentage*100:.1f}%) exceeds safety cap "
                               f"({self.MAX_REMOVAL_PERCENTAGE*100:.1f}%). Deduplication aborted."
            }
        
        # Step 8: Compute distribution shift
        print(f"\n[Step 8] Computing class distribution shift...")
        max_class_shift = self._compute_distribution_shift(self.labels, representatives)

        # Step 9: Save reduced dataset
        print(f"\n[Step 9] Saving deduplicated dataset...")
        self._save_reduced_dataset(representatives)

        
        print("\n" + "=" * 70)
        print("DEDUPLICATION COMPLETE")
        print("=" * 70)
        print(f"Original size: {original_size}")
        print(f"Reduced size: {len(representatives)}")
        print(f"Removed: {removal_count} ({removal_percentage*100:.2f}%)")
        print(f"Max class shift: {max_class_shift*100:.2f}%")
        print(f"Threshold used: {self.threshold}")

        # Gemini AI explanation (if GEMINI_API_KEY set)
        ai_explanation = ""
        try:
            try:
                from .gemini_explainer import explain_removed_test_cases
            except ImportError:
                from gemini_explainer import explain_removed_test_cases
            summary = {
                "removed_count": removal_count,
                "original_size": original_size,
                "reduced_size": len(representatives),
                "similarity_threshold": self.threshold,
                "reduction_percentage": removal_percentage * 100,
                "max_class_shift_percent": max_class_shift * 100,
            }
            ai_explanation = explain_removed_test_cases(summary)
            print(f"\nAI Explanation: {ai_explanation}")
        except Exception:
            pass

        # Generate description
        description = (
            f"Removed {removal_percentage*100:.1f}% semantically redundant reviews using "
            f"cosine threshold {self.threshold}. "
            f"Class distribution preserved with max shift {max_class_shift*100:.2f}%. "
            f"This reduces evaluation cost and energy usage while maintaining coverage."
        )
        
        result = {
            "original_size": original_size,
            "reduced_size": len(representatives),
            "reduction_percentage": removal_percentage,
            "max_class_shift": max_class_shift,
            "threshold_used": self.threshold,
            "description": description
        }
        if ai_explanation:
            result["ai_explanation"] = ai_explanation
        return result
    def _save_reduced_dataset(self, selected_indices: Set[int]) -> None:
        """
        Save deduplicated dataset to new directory: dataset/test_deduplicated
        """
        output_root =os.path.join(os.path.dirname(self.dataset_path), "test_deduplicated")
        pos_out = os.path.join(output_root, "pos")
        neg_out = os.path.join(output_root, "neg")

        os.makedirs(pos_out, exist_ok=True)
        os.makedirs(neg_out, exist_ok=True)

        counter_pos = 0
        counter_neg = 0

        for idx in selected_indices:
            text = self.texts[idx]
            label = self.labels[idx]

            if label == 1:
                filepath = os.path.join(pos_out, f"{counter_pos}.txt")
                counter_pos += 1
            else:
                filepath = os.path.join(neg_out, f"{counter_neg}.txt")
                counter_neg += 1

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

        print(f"Saved deduplicated dataset to {output_root}")



if __name__ == "__main__":
    # Example usage
    dataset_path = "dataset/test"
    deduplicator = IMDBTextDeduplicator(dataset_path, threshold=0.80)
    result = deduplicator.deduplicate()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("\nDescription:")
    print(result["description"])
