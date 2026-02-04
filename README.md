# ABC Dataset for DeepSDF

This branch (`dataset_ABC`) integrates the **ABC dataset**—a large-scale CAD model collection designed for geometric deep learning—into the [DeepSDF](https://github.com/facebookresearch/DeepSDF) framework. It provides tools, preprocessed data, and documentation to enable training and evaluation using high-quality, watertight CAD models alongside or in comparison with ShapeNet.

## Overview

The **ABC dataset** (*A Big CAD Model Dataset for Geometric Deep Learning*) is a large-scale repository of one million Computer-Aided Design (CAD) models, created and maintained by the [deep-geometry](https://deep-geometry.github.io/abc-dataset/) team. Unlike scan-based datasets, ABC consists of **parametrically defined, ground-truth CAD geometries**, offering precise differential quantities, patch segmentation, and feature annotations ideal for geometric learning tasks.

> **Reference**:  
> Koch, S., Matveev, A., Jiang, Z., et al. (2019). *ABC: A Big CAD Model Dataset For Geometric Deep Learning*. CVPR 2019.  
> [Dataset Website](https://deep-geometry.github.io/abc-dataset) | [NYU Archive](https://archive.nyu.edu/handle/2451/44309)

## Key Features

- **Large Scale**: Contains millions of real-world CAD models, with this branch focusing on a curated subset.
- **High Quality**: Models are derived from explicit parametric representations, ensuring geometric fidelity.
- **Diversity**: Encompasses a wide variety of mechanical and industrial part designs with rich geometric complexity.
- **Standardized Format**: Uniform preprocessing and export formats (e.g., `.obj`, `.step`) facilitate reproducible research.

## Repository Structure

This repository includes:
- Documentation for using the ABC dataset within DeepSDF.
- Scripts to filter, preprocess, and validate watertight models.
- Instructions for generating SDF samples compatible with DeepSDF's training pipeline.
- Evaluation tools to verify SDF quality and consistency.

## Accessing the ABC Dataset

The full ABC dataset is publicly available via the NYU Digital Archive:

🔗 [https://archive.nyu.edu/handle/2451/44309](https://archive.nyu.edu/handle/2451/44309)

> ⚠️ Note: The complete dataset is ~1.8 TB in size. This branch focuses on a **small, curated subset** of 50 watertight models suitable for prototyping and benchmarking in DeepSDF.

## Use Cases

The ABC dataset is particularly well-suited for:
- **3D shape representation and reconstruction** using implicit functions (e.g., SDFs).
- **Training geometric deep learning models** that require high-fidelity ground-truth geometry.
- **Shape analysis**, including symmetry detection, segmentation, and feature recognition.
- **Comparative studies** with other datasets like ShapeNet, especially in terms of geometric precision and topology.

## Comparison with ShapeNet

| Aspect                | ABC Dataset                          | ShapeNet                             |
|-----------------------|--------------------------------------|--------------------------------------|
| **Source**            | Parametric CAD models (synthetic)    | Human-created 3D scans (reconstructed) |
| **Geometry Quality**  | Exact, differentiable, watertight    | Often noisy, non-watertight, holes   |
| **Topology**          | Clean, manifold, consistent normals  | Irregular, may require repair        |
| **Use in DeepSDF**    | Ideal for precise SDF supervision    | Requires careful preprocessing       |

While both datasets support 3D learning research, **ABC provides superior geometric consistency**, making it especially valuable for tasks requiring accurate signed distance fields—such as those in DeepSDF.

## Extraction Script (via Docker)

The following Docker command scans the raw ABC download directory and copies the first 50 watertight models to a new folder:

```bash
docker run --rm \
  -v /mnt/c/Users/Administrator/Downloads:/downloads \
  -v /mnt/c/Users/Administrator/Desktop:/desktop \
  deepsdf:latest \
  python3 -c "
import trimesh, os, shutil
src = '/downloads/abc_0000_obj_v00'
dst = '/desktop/ABC_50_Watertight_Models'
os.makedirs(dst, exist_ok=True)
count = 0
for name in sorted(os.listdir(src)):
    if count >= 50: break
    folder = os.path.join(src, name)
    if not os.path.isdir(folder): continue
    objs = [f for f in os.listdir(folder) if f.endswith('.obj')]
    if not objs: continue
    try:
        mesh = trimesh.load(os.path.join(folder, objs[0]))
        if mesh.is_watertight:
            count += 1
            new_name = f'{count:04d}.obj'
            shutil.copy2(os.path.join(folder, objs[0]), os.path.join(dst, new_name))
            print(f'[OK] {count}: {name}/{objs[0]} -> {new_name}')
        else:
            print(f'[SKIP] Non-watertight: {name}')
    except Exception as e:
        print(f'[ERROR] {name} - {e}')
print(f'\n[INFO] Found {count} watertight models → saved to /desktop/ABC_50_Watertight_Models')
"
```

## Dataset Split

The curated set is divided as follows for training and evaluation:

| Split | Models               | Count |
|-------|----------------------|-------|
| Train | 0001.obj – 0040.obj  | 40    |
| Test  | 0041.obj – 0050.obj  | 10    |

This split enables standard supervised learning protocols within DeepSDF.

## SDF Preprocessing

Each .obj model is preprocessed into a signed distance field (SDF) sample file (.npz) using DeepSDF's PreprocessMesh tool:

```bash
cd /root/DEEPSDF/DeepSDF/bin
./PreprocessMesh -m dataset_ABC_test/0001.obj -o dataset_ABC_test/0001.npz --var 0.005
```

### Key Parameters
- `--var 0.005`: Controls noise level for near-surface sampling.
- Output includes:
  - `pos`: points inside the shape (SDF < 0)
  - `neg`: points outside the shape (SDF > 0)
- Coordinates are normalized to approximately `[-0.95, 0.95]^3`.
- The corresponding normalized mesh is saved as `0001_norm.obj`.

💡 Despite an OpenGL warning during execution, the preprocessing completes successfully and produces valid SDF samples.

## SDF Quality Verification

We validate the correctness and quality of the generated SDF data using a custom script (`sdf_quality_check_final.py`). For model 0001, the results are:

> Summary Report (Model 0001)
> - Total Samples: 499,462 
> - Internal Points (SDF < 0): 42.59% 
> - Symbol Convention Compliance: 100% (matches DeepSDF spec) 
> - SDF–Mesh Consistency:
>   - Sign Agreement: 100.00% ✓ 
>   - MAE vs. Ground Truth: 0.000020 ✓
> - Gradient Norm (∥∇SDF∥): Median = 1.0000 ✓ 
> - Surface Sampling Density: 92.2% of points within 1σ of surface ✓
> - Overall Score: 10.0 / 10 — Excellent quality, ready for training.

🔍 Full logs confirm that ABC models—being watertight and precisely defined—yield significantly higher SDF fidelity than typical ShapeNet models (which often suffer from topology errors).

## ABC vs. ShapeNet: SDF Quality Comparison

We evaluated SDF samples generated from both datasets using identical DeepSDF preprocessing and validation protocols. The results highlight a significant advantage of ABC due to its watertight geometry and precise CAD origins.

| Metric / Category          | ABC Dataset                          | ShapeNet Dataset                      | Interpretation                                                                 |
|----------------------------|--------------------------------------|---------------------------------------|--------------------------------------------------------------------------------|
| Mesh Watertightness        | ✓ True                               | ✗ False                               | Non-watertight meshes in ShapeNet cause unreliable inside/outside classification. |
| Vertices per Mesh          | 15,262                               | 10,453                                | ABC models are geometrically richer.                                            |
| Total SDF Samples          | 499,462                              | 454,857                               | Comparable scale.                                                              |
| Internal Point Ratio       | 42.59%                               | 56.97%                                | ShapeNet has more interior volume (possibly due to mesh artifacts).             |
| SDF Std. Dev.              | 0.1173                               | 0.1325                                | ABC SDF values are slightly more stable.                                       |
| Symbol Compliance          | 100% ✓                               | 100% ✓                                | Both follow DeepSDF's convention: pos=SDF<0, neg=SDF>0.                        |
| Sign Consistency (critical)| 100.00% ✓                            | 43.50% ✗                              | ABC perfectly aligns with ground-truth SDF; ShapeNet fails frequently.          |
| MAE (vs. True SDF)         | 0.000020 ✓                           | 0.033618 ✗                            | ABC is ~1,680× more accurate.                                                  |
| Gradient Median (∥∇SDF∥)   | 1.0000 ✓                             | 1.0000 ✓                              | Both maintain correct Eikonal property.                                        |
| Surface Sampling           | 92.2% within 1σ ✓                    | 93.9% within 1σ ✓                     | Both sample surfaces well.                                                      |
| Overall Score              | 10.0 / 10 ✓                          | 7.0 / 10                              | ABC is superior for high-fidelity SDF learning.                                |

### Conclusion:
Under the same DeepSDF pipeline, ABC produces near-perfect SDF supervision, while ShapeNet's reconstruction artifacts lead to inconsistent sign assignments and higher numerical error. For tasks requiring geometric precision (e.g., reconstruction, interpolation), ABC is strongly preferred.

## SDF Data Generation & Validation Pipeline

### 1. Preprocessing Workflow (PreprocessMesh)
| Step | Operation          | Details                                                                 |
|------|--------------------|-------------------------------------------------------------------------|
| 1    | Load mesh          | 0001.obj (15,262 vertices, 30,520 faces, watertight)                   |
| 2    | Normalize          | Bounding cube normalization → fits in `[-1, 1]^3`                       |
| 3    | Save normalized mesh| → 0001_norm.obj (vertex range: ±0.7165)                                |
| 4    | Surface sampling   | High-density sampling with multi-scale noise                            |
| 5    | SDF sampling       | 94% near surface + 6% uniform in `[-0.9, 0.9]^3`                        |
| 6    | Compute SDF        | Exact Euclidean signed distance (not approximate)                       |
| 7    | Organize data      | Split into `pos` (SDF < 0) and `neg` (SDF > 0)                          |
| 8    | Export             | → 0001.npz (499k samples, coord range: ±0.95)                           |

### 2. Output Format
- `0001.npz`
  - `pos`: (N₁, 4) array → [x, y, z, sdf], all sdf < 0
  - `neg`: (N₂, 4) array → [x, y, z, sdf], all sdf > 0
- `0001_norm.obj`
  - Normalized, watertight mesh for visualization and validation.

### 3. Validation Logic
- Reference SDF: Computed independently using `trimesh.proximity.signed_distance`.
- Sign Correction: `trimesh` uses opposite sign convention → we flip signs to match DeepSDF.
- Metrics:
  - Sign consistency rate
  - MAE / RMSE against reference SDF
  - Gradient norm via finite differences (∥∇SDF∥ ≈ 1)
  - Surface sampling density
  - Internal/external balance

### 4. Key Technical Insights
- Sign Convention Mismatch: `trimesh` defines inside as SDF > 0; DeepSDF uses SDF < 0. Always invert when validating.
- Validation Upgrade: Replaced heuristic checks with rigorous finite-difference gradient and sign-consistency tests.
- Geometric Fidelity Matters: Only watertight meshes guarantee correct SDF signs—this is why ABC outperforms ShapeNet.

### 5. Best Practices Summary
- Always use watertight input meshes.
- Normalize geometry to `[-1, 1]^3`; sample SDF in `[-0.95, 0.95]^3` to avoid boundary effects.
- Use dense near-surface sampling (≥90%) to capture fine details.
- Validate SDF quality before training—especially sign consistency.
