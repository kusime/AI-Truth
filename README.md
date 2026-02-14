# AI Intuition Verification Project

This project contains a series of Python scripts designed to verify deep learning intuitions using synthetic data and geometric visualizations.

## Project Structure

- **`src/`**: Python source code for proof-of-concept scripts.
- **`output/`**: Generated HTML visualizations (Plotly) and results.
- **`docs/`**: Reference documents (e.g., `new.html` - The User's Manifesto).
- **`venv/`**: (Optional) Python virtual environment.

## Verified Intuitions

1.  **Resolution vs. Structure** (`intuition_proof.py`)
    -   Proves that sufficient width (resolution) enables universal approximation regardless of depth/structure.
    -   Visualizes high-dimensional manifold unrolling.

2.  **High-Dimensional Sparsity** (`sparsity_proof.py`)
    -   Proves *Cover's Theorem*: High-dimensional space makes data sparse and linearly separable.
    -   Shows "Blessing of Dimensionality".

3.  **Linear Independence & Static Points** (`independence_proof.py`)
    -   Proves that in high dimensions ($D \ge N$), random vectors become orthogonal and linearly independent ("Static Points").

4.  **Neural Collapse & Simplex Geometry** (`collapse_proof.py`)
    -   Proves that at terminal training phase, data collapses to static class means which form an encoded orthogonal Simplex.
    -   Relates to stable diffusion as "Manifold Restoration".

5.  **Dimensional Collision & Shadow Problem** (`collision_proof.py`)
    -   Proves that projecting high-dimensional unique data to low dimensions causes inevitable information loss (collisions).
    -   Explains why Multi-Head Attention is needed (to avoid single-view collisions).

## How to Run

1.  Activate virtual environment (if applicable):
    ```bash
    source venv/bin/activate
    ```

2.  Run any script from the project root:
    ```bash
    python src/intuition_proof.py
    python src/collision_proof.py
    # ... etc
    ```

3.  Open the generated HTML files in `output/` to see the visualizations.
