# Architecture and roadmap

## Implemented starter

FastAPI → simulated robot records → deterministic health score → advisory API → HTML dashboard.

Repair cost endpoint uses user-supplied parts, labor, and downtime estimates. No real parts catalog or computer vision inspection is integrated.

## Planned modules (not implemented)

- `perception/`: camera, depth, LiDAR, object detection and segmentation
- `world_model/`: dynamics prediction, imagined rollouts and uncertainty
- `navigation/`: mapping, SLAM, path planning and obstacle avoidance
- `manipulation/`: grasping, motion planning and simulation
- `vla/`: language-conditioned robot policies
- `rl/`: offline and simulation-based policy learning
- `agents/`: supervisor, perception, navigation, maintenance, safety
- `maintenance/`: anomaly detection, fault classification and RUL
- `inspection/`: robot part photo analysis and repair cost intelligence
- `evaluation/`: navigation success, safety events, perception metrics, drift
- `mlops/`: model registry, reproducibility, telemetry monitoring

Safety principle: AI is advisory; independently certified controllers handle safety-critical actions.
