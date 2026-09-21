# 3D Camera RGB-D Body Model

Research and engineering package for Orbbec Gemini 2 L RGB-D human-region
localization and body-scale estimation.

## Included capabilities

- RTMDet person detection and RTMW-X 17-point pose estimation.
- RGB-D body-region inference for 12 fixed regions.
- Camera-space 3D region centers using aligned depth and camera intrinsics.
- Depth-quality flags, confidence, visibility, and metric-source tracing.
- Nearest-person tracking, short occlusion recovery, and temporal filtering.
- Gemini RGB-D standing-height estimation and lying body-length estimation.
- A local HTTP inference service for the desktop capture application.

## Repository layout

```text
app/local_live_inference_server.py   Local HTTP service
app/project/body_model/              Body model, geometry, tracking, and pose code
app/project/configs/                 RTMW-X and detector configurations
app/height/work/                     Gemini height estimation code
app/vendor/                          Pinned lightweight runtime dependencies
models/body_regions/v12/best.pt      V12 body-region checkpoint
models/body_regions/v5/best.pt       Historical V5 body-region checkpoint
models/pose/                          RTMW-X checkpoint
models/detector/                      RTMDet-nano checkpoint
tests/                                Package tests, when present
```

## Start the local service

The service expects Python 3.11, PyTorch with CUDA support, OpenCV, Flask,
NumPy, MMPose-compatible dependencies, and an NVIDIA GPU. The checked-in
`runtime/` directory is intentionally excluded; install the environment
separately for the target workstation.

From the repository root:

```powershell
python app/local_live_inference_server.py --device cuda:0 --port 18766
```

Health check:

```text
http://127.0.0.1:18766/health
```

The desktop capture application sends aligned RGB and uint16 depth frames to
`/infer`. The depth scale and camera intrinsics are supplied per request.

## Model notes

The V12 network is a project-specific RGB-D body-region model. It uses RTMW-X
keypoints as a pose front end, local metric depth geometry, region queries,
multi-task heads, and bounded prior residuals. Reliable measured geometry is
kept separate from learned fallback predictions.

The height module is a geometry-based estimator. It combines RGB crown
localization, aligned depth, camera intrinsics, support-plane estimation, and
multi-frame robust aggregation. It does not require a manually entered person
height.

## Data and safety

Raw RGB-D, IR, point-cloud, IMU, patient, and experiment data are deliberately
excluded from this public repository. Keep those files under the local
`F:\Gemini2L_Captures` workflow. The outputs are research/engineering results,
require human review, and are not clinical decisions.

Model checkpoints are large files tracked with Git LFS. Review the licenses of
the upstream RTMW-X, RTMDet, MMPose, MMDetection, and vendored dependencies
before redistribution.

