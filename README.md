# Part 1 - Feature Tracking

## Features Used
Shi-Tomasi corner detection for identifying good initial features to track.

Lucas-Kanade Optical Flow (Pyramidal) for tracking features between consecutive frames.

## Algorithm
For the first frame, detect Shi-Tomasi corners using goodFeaturesToTrack.

For each new frame, track features from the previous frame using calcOpticalFlowPyrLK.

Filter matches based on displacement and optical flow error thresholds.

Draw lines and circles to visualize tracked features and motion vectors.

Replenish lost features if the count falls below a threshold.

Annotate and save visualizations as single-frame and dual-view videos.

Log tracking statistics (mean error, tracked/lost/added features) to a CSV file.

# Part 2 – Visual Odometry & Real‑Time 3‑D Reconstruction  

## Additions on top of Part 1
- **Re‑use of Part 1 tracks**  
  *Shi–Tomasi* + *pyramidal Lucas‑Kanade* still deliver 2‑D corner trajectories.  
  After getting **(x, y)** in both frames *Part 1*, we now estimate the **z‑coordinate** by triangulation.
- **Robust camera motion**  
  1. **Fundamental → Essential** matrix with RANSAC (five‑point).  
  2. `recoverPose` → rotation **R** and translation **t** per frame.  
  3. Global pose is chained:  `T_wc ← T_wc · T_ct`.
- **Linear triangulation** creates a rolling 3‑D point cloud (latest 2 k pts).  
  Points with 0.1 m < Z < 1 km are kept.
- **Auto‑scaled 2‑D map**  
  *Bottom panel* shows orthographic X‑Z projection:  
  cyan 3‑px poly‑line (camera centres) + light‑cyan dots (triangulated points).
- **Split‑screen H.264 output**  
  `HW2_sample_result_video.mp4` – top = RGB + tracks, bottom = map.  
  Extra artefacts: `traj_out.avi`, `cloud_out.avi`, `trajectory.txt`, `cloud.ply`.

---

## Methodology
|-------|----------------------|
| **Corner detection** | Shi–Tomasi:  λ<sub>min</sub> > *q*·λ<sub>max</sub> |
| **Tracking** | Iterative LK minimises Σ[I<sub>t</sub>(p+Δx) − I<sub>t‑1</sub>(p)]² |
| **Essential matrix** | E = Kᵀ F K (five‑point + RANSAC) |
| **Pose** | SVD of E, choose (R,t) that puts all points in front of both cams |
| **Triangulation** | DLT, normalise X ← X/X₄ |
| **Trajectory integration** | R<sub>wc</sub> ← R<sub>wc</sub>·R<sub>ct</sub>, t<sub>wc</sub> ← t<sub>wc</sub> + R<sub>wc</sub>·t<sub>ct</sub> |
| **Map scaling** | s = 0.9 · min((W−1)/Δx, (H−1)/Δz) |

---

## How to build & run on WSL (Ubuntu ≥ 20.04)

```bash
# 1.  enter project folder
cd /mnt/c/Users/mm17889/feature_tracking

# 2.  one‑time dependencies
sudo apt update
sudo apt install build-essential pkg-config libopencv-dev

# 3.  compile
g++ -std=c++17 -O3 feature_odometry_pointcloud.cpp \ $(pkg-config --cflags --libs opencv4) -o part2

# 4.  run
./part2

# 5.  open outputs in Windows
explorer.exe .
