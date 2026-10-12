# 🦾 muarm

Robot arm kinematics and manipulation learning in MuJoCo physics simulation, featuring **Pinocchio-based FK/IK**, **Catmull-Rom trajectory planning**, and a **multi-robot RL/IL training framework** supporting PPO, SAC, TD3, DDPG and Behavior Cloning.

| Trajectory | Impedence Control |
|:---:|:---:|
| <img src="media/ik_panda.gif" width="280"> | <img src="media/impedence_panda.gif" width="260"> |
---



## 📁 Project Structure

```
muarm/
├── models/
│   ├── franka_emika_panda/      # Franka Panda MJCF model + meshes
│   └── trs_so_arm100/           # TRS SO-ARM100 MJCF model + meshes
├── kinematic/
│   ├── panda_kinematics.py      # PandaKinematics — FK / IK / Jacobian (Pinocchio)
│   ├── trajectory.py            # TrajectoryGenerator — cubic, Catmull-Rom, Cartesian arc
│   ├── run_fk.py                # Forward kinematics demo
│   ├── run_ik.py                # Inverse kinematics demo
│   └── run_trajectory.py        # Figure-8 Lissajous trajectory demo
├── dynamics/
│   ├── impedance_controller.py  # Task-space impedance control (τ = J^T K e + τ_bias)
│   └── admittance_controller.py # Task-space admittance control (virtual ODE + inner PD)
├── learning/
│   ├── envs/
│   │   ├── base_env.py          # MuJocoRobotEnv — robot-agnostic Gymnasium base class
│   │   ├── registry.py          # make_env(robot, task)
│   │   └── tasks/
│   │       ├── base_task.py     # BaseTask interface
│   │       ├── reach.py
│   │       ├── reach_obstacles.py
│   │       ├── reach_obstacles_2.py
│   │       ├── push.py
│   │       └── pick_place.py
│   ├── cdf/                     # Pretrained Panda configuration-space distance field
│   ├── cdflow/                  # Scene-conditioned boundary data, CNF, and field inference
│   ├── robots/
│   │   ├── panda.py             # FrankaPandaEnv (Cartesian or joint-delta control)
│   │   └── so_arm.py            # SoArm100Env (6D joint velocity)
│   ├── algos/
│   │   ├── rl_trainer.py        # train_rl — PPO/SAC/TD3/DDPG + SuccessRateCallback
│   │   ├── cdf_mppi.py          # One-step CDF-MPPI planner + checkpoint preparation
│   │   ├── cdflow_projection.py # CDFlow direct whole-body contact projection
│   │   ├── cdflow_mppi.py       # Optional CDFlow + MPPI extension
│   │   ├── star_worlds.py       # Reactive point-goal planning with link avoidance
│   │   ├── differentiable_mpc.py # Point-goal trajectory optimization
│   │   ├── ik_rrt_connect.py    # Collision-valid IK and joint-space RRT-Connect
│   │   ├── planning.py          # Lazy planner registry
│   │   └── il/
│   │       └── bc.py            # Behavior Cloning (MLP + CosineAnnealingLR)
│   ├── utils/
│   │   └── visualize.py         # MuJoCo overlay helpers
│   ├── train.py                 # Unified training entry point
│   └── play.py                  # Real-time policy playback
├── scripts/
│   ├── generate_cdflow_data.py # Multi-start MuJoCo collision-boundary dataset
│   ├── train_cdflow.py         # Offline CDFlow model fitting
│   ├── eval_cdflow_field.py    # Held-out distance/direction and boundary audit
│   ├── eval_cdflow_mppi.py     # Paired CDF/CDFlow MPPI comparison
│   └── eval_cdflow_contact.py  # CDFlow whole-arm button-contact evaluation
├── requirements.txt
└── pyproject.toml
```

---

## 🔧 Installation

```bash
cd muarm
uv venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

The requirements include `torchdiffeq` for CDFlow's differentiable neural ODE.

---

## 🚀 Usage

### ▶️ Kinematics

```bash
source .venv/bin/activate

# Forward kinematics
python kinematic/run_fk.py

# Inverse kinematics
python kinematic/run_ik.py

# Figure-8 continuous trajectory with live EE trail
python kinematic/run_trajectory.py
```

### ▶️ Dynamics — Impedance & Admittance Control

Both demos use torque-controlled actuators (`panda_motor.xml` + `scene_torque.xml`).
Interact via **Ctrl + drag** in the MuJoCo viewer.

```bash
source .venv/bin/activate

# Impedance control — drag and release, arm springs back
python dynamics/impedance_controller.py

# Admittance control — arm compliantly follows your applied force
python dynamics/admittance_controller.py
```

**Control law summary:**

| | Impedance | Admittance |
|---|---|---|
| Input | Displacement `e` | External force `F_ext` |
| Output | Joint torques directly | Virtual reference → inner PD |
| Feel | Stiff spring | Soft, mass-damper compliant |
| Equation | `τ = J^T(Kp·e − Kd·ẋ) + τ_bias` | `M_d·ẍ_v = F_ext − D_d·ẋ_v − K_d·(x_v−x_eq)` |

Tune the parameters at the top of each file:

| Parameter | File | Effect |
|-----------|------|--------|
| `KP / KD` | `impedance_controller.py` | Spring stiffness / damping |
| `M_D` | `admittance_controller.py` | Lower → faster response |
| `D_D` | `admittance_controller.py` | Higher → smoother motion |
| `K_D` | `admittance_controller.py` | 0 = free drift, >0 = spring to eq |

### ▶️ RL Training

```bash
source .venv/bin/activate

# Headless (recommended for speed)
python learning/train.py --robot panda --task reach     --algo sac  --timesteps 200000
python learning/train.py --robot panda --task reach_obstacles --algo sac --timesteps 200000
python learning/train.py --robot panda --task reach_obstacles_2 --algo sac --timesteps 200000
python learning/train.py --robot panda --task push      --algo td3  --timesteps 300000
python learning/train.py --robot panda --task pick_place --algo sac --timesteps 500000

# Enable real-time viewer
python learning/train.py --robot panda --task reach --algo sac --timesteps 200000 --render

# Parallel environments (headless only)
python learning/train.py --robot panda --task reach --algo sac --timesteps 200000 --n-envs 4

# TensorBoard logging
python learning/train.py --robot panda --task reach --algo sac --timesteps 200000 --tensorboard
tensorboard --logdir learning/runs/tb

# SO-ARM100
python learning/train.py --robot so_arm --task reach --algo sac --timesteps 100000

# UFACTORY xArm7 (with / without hand)
python learning/train.py --robot xarm7        --task reach --algo sac --timesteps 200000
python learning/train.py --robot xarm7_nohand --task reach --algo sac --timesteps 200000
```

The obstacle tasks are Panda-only and keep the normal Reach reward, horizon,
and training settings. `reach_obstacles` has its own collision-free start pose
and goal region for its vertically stacked spheres. Its three 0.12 m radius
spheres are centered at `(0.60, -0.16, 0.14)`, `(0.60, -0.16, 0.42)`, and
`(0.60, -0.16, 0.70)` metres, leaving 0.04 m gaps and spanning 0.02–0.82 m
above the floor. Goals are sampled on the far side of the stack from the arm,
around `(0.58, 0.10, 0.38)` m with half-range `(0.02, 0.02, 0.02)` m, and
rejected if inside an obstacle.
`reach_obstacles_2` retains its original goal region and adapts the
cross-shaped Franka benchmark from
[CDF-MPPI](https://github.com/Rin-Li/cdf_mppi) to the Reach workspace.
Their SAC policies directly control seven joint-position deltas;
the other Panda tasks retain their Cartesian action and IK control. The observation
uses the existing padding for CDF clearance and seven normalized joint gradients from
the [MIT-licensed CDF model](https://github.com/idiap/cdf). Each control step in
contact with an obstacle receives an additional `-1.0` reward.

### ▶️ CDF-MPPI Planning

CDF-MPPI supports Panda `reach_obstacles` and `reach_obstacles_2`. The source
Franka experiment uses a pretrained CDF and requires no offline policy
training. `train.py` saves its weights and planner settings to a `.pt`
checkpoint; `--timesteps` applies only to RL. Preparation uses one environment
and no TensorBoard logging. `reach_obstacles_2` has the Reach-workspace cross;
`reach_obstacles` has three vertically stacked spheres.

```bash
python learning/train.py --robot panda --task reach_obstacles --algo cdf_mppi
python learning/play.py --robot panda --task reach_obstacles --algo cdf_mppi \
    --model learning/runs/panda_reach_obstacles_cdf_mppi.pt --episodes 5

python learning/train.py --robot panda --task reach_obstacles_2 --algo cdf_mppi
python learning/play.py --robot panda --task reach_obstacles_2 --algo cdf_mppi \
    --model learning/runs/panda_reach_obstacles_2_cdf_mppi.pt --episodes 5
```

Inference uses the source's fixed joint start, a 0.01-second integration step,
and horizon-one MPPI. At the start of each episode, the planner keeps the first
IK goal if it passes MuJoCo collision checks; otherwise it tries deterministic
additional joint seeds and targets inside the task's Cartesian success radius.
It does not move the displayed target or obstacles. Each
iteration samples 200 clipped joint velocities, scores their angle to the joint
goal and (when active) the CDF gradient, updates the Gaussian mean and
covariance, then projects only the weighted velocity to joint limits. It
proposes `q_next = q + 0.01 * u` and checks sampled poses along the
interpolated joint-space edge in MuJoCo before hard-setting the endpoint. A
negative obstacle, floor, or self-contact gap aborts before accepting the pose. Otherwise
playback displays it and stops when the hand enters the
task's Cartesian target tolerance or after 500 steps. CDF queries the task's
existing obstacle points, and playback leaves its obstacle geometry and sampled
goal unchanged. The printed end-effector error measures distance to the target
shown in the viewer.

IK runs only at episode setup; the MPPI loop does not simulate Panda servo
dynamics or apply damping or recovery actions. The
checkpoint format also loads older CDF-MPPI checkpoints, but runs them with
this task-goal mapping. MuJoCo supplies the collision geometry, so numerical
gaps can differ from the source's PyBullet model. A sampled hand target can
still be blocked for the full arm; those episodes report `unreachable_goal`.
The planner also stops on a near-obstacle field with no usable outward
direction. Regression checks can be run with
`python -m unittest learning.algos.test_cdf_mppi`.

### ▶️ CDFlow: boundary data, field training, and contact projection

[CDFlow](https://arxiv.org/html/2509.13771v1) is the scene-conditioned
collision-boundary distribution and distance field. Its adaptive data sampler
finds near-minimum collision-boundary configurations through global restarts
and local refinement. Training fits a conditional continuous normalizing flow
and a distance head with likelihood, distance, direction, Eikonal, and tension
losses. The paper's button experiment then repeatedly projects the robot
joints along the **negative** expected outward direction until any arm body
touches a 0.01 m sphere. The primary `cdflow` path implements this projection;
it does not use MPPI or an end-effector position goal.

Generate varied, reachable button scenes and train the field:

```bash
python scripts/generate_cdflow_data.py \
    --task reach_obstacles --button-scenes \
    --num-scenes 100 --queries-per-scene 128 \
    --global-starts 16 --local-starts 4 --seed 0 \
    --output learning/runs/cdflow_button.npz
python learning/train.py --robot panda --task cdflow_button --algo cdflow \
    --dataset learning/runs/cdflow_button.npz \
    --cdflow-epochs 100 --cdflow-batch-size 32 --cdflow-device cpu
```

The checkpoint is `learning/runs/panda_cdflow_button_cdflow.pt`. Button scene
generation keeps one sphere active, places its center at a collision-checked
contact location, and records the positions and witnesses in the NPZ metadata.
The dataset splits by scene. `scripts/train_cdflow.py` exposes architecture,
ODE solver, and loss-weight options. The example settings are practical run
settings; the paper's training scale is larger.

Run the paper-style contact controller on one recorded scene:

```bash
python learning/play.py --robot panda --task cdflow_button --algo cdflow \
    --model learning/runs/panda_cdflow_button_cdflow.pt \
    --cdflow-scene-data learning/runs/cdflow_button.npz \
    --cdflow-scene-id 0 --episodes 10 --headless
```

This route uses Monte Carlo boundary samples for the expected outward
direction by default (`--cdflow-mode mc --cdflow-samples 64`).
`--cdflow-mode regression` uses the paper's faster distance-head gradient.
Each trial has a 3 s wall-clock limit. MuJoCo contact between the selected
sphere and any Panda arm, hand, or finger collision geom defines success;
joint limits, floor and self-contact constraints are checked along each step.
Projection step size and edge resolution are implementation settings because
the paper does not specify them. This geometric controller is headless; the
`--headless` flag keeps the command's intent explicit.

To audit field quality on held-out scenes and get a detailed contact report:

```bash
python scripts/eval_cdflow_field.py \
    --model learning/runs/panda_cdflow_button_cdflow.pt \
    --data learning/runs/cdflow_button.npz --split test \
    --mc-samples 64 --boundary-samples 16 \
    --output learning/runs/cdflow_field_eval.json
python scripts/eval_cdflow_contact.py \
    --checkpoint learning/runs/panda_cdflow_button_cdflow.pt \
    --scene-data learning/runs/cdflow_button.npz \
    --split test --trials-per-scene 1 --mode mc \
    --output learning/runs/cdflow_contact.json
```

The contact script evaluates each distinct held-out button position and
reports both trial and scene success rates. `--split all` includes every
recorded position; `--trials-per-scene` repeats starts per position. Without
`--scene-data`, it runs at one fixed center, optionally set with
`--sphere-center X Y Z --sphere-radius 0.01`. The report checks whether each
selected geometry was in the checkpoint's training split. Starts come from
recorded free queries or perturbations of a verified contact witness; this
scene sampler and start distribution are muarm choices, so the rate is not a
direct reproduction of the paper's 1,000-position benchmark. To run one recorded
scene, select it in `play.py` with `--cdflow-scene-data` and
`--cdflow-scene-id`.
A one-epoch or tiny diagnostic checkpoint only proves that the pipeline runs;
it does not establish contact performance.

`cdflow_mppi` and `scripts/eval_cdflow_mppi.py` remain as optional muarm
experiments that use the learned field inside MPPI for end-effector reaching.
They are not the CDFlow algorithm in the paper. The paper's separate
gradient-based motion planning study does not specify enough optimizer detail
for an exact planner reproduction here.

Focused checks:

```bash
python -m unittest learning.cdflow.test_data learning.cdflow.test_model learning.cdflow.test_contact learning.test_cdflow_entrypoints learning.algos.test_cdf_mppi learning.algos.test_cdflow_mppi
```

### ▶️ Point-target obstacle planners

Three additional planners accept the task's **end-effector XYZ goal** and run
on Panda `reach_obstacles` or `reach_obstacles_2`. They each return normalized
joint-delta actions through the existing Franka environment. None needs
offline training or a model checkpoint: selecting one in `train.py` prints a
notice and exits. `play.py` constructs the planner for each run.

| `--algo` | Planning pipeline | Reference implementation |
|----------|-------------------|--------------------------|
| `star_worlds` | Obstacle proxies → modulated tip velocity → position IK → null-space link avoidance → collision-checked joint action | [Star-Worlds Franka demo](https://github.com/saayuj/star-worlds/blob/main/examples/franka_pybullet_demo.py) |
| `differentiable_mpc` | Joint acceleration horizon → differentiable FK point cost and collision constraints → first joint target → replan | [Differentiable MPC](https://github.com/GiantMGG/robot-optimal-trajectory-planning) |
| `ik_rrt_connect` | Position IK with collision-valid candidates → joint-space RRT-Connect → checked, shortened path → waypoint tracking | [CoppeliaSim IK/RRT-Connect example](https://github.com/SarathKumarKomathukattil/robot-motion-planning-coppeliasim/blob/main/scripts/motion_planner.lua) |

```bash
# All three train commands only report that training is unnecessary.
python learning/train.py --robot panda --task reach_obstacles --algo star_worlds

# Playback opens the viewer by default. No --model argument is used.
python learning/play.py --robot panda --task reach_obstacles --algo star_worlds --episodes 1
python learning/play.py --robot panda --task reach_obstacles --algo differentiable_mpc --episodes 1
python learning/play.py --robot panda --task reach_obstacles --algo ik_rrt_connect --episodes 1

# Reproducible, viewer-free verification
python learning/play.py --robot panda --task reach_obstacles --algo ik_rrt_connect \
    --episodes 1 --seed 0 --headless

# Save the same inference metrics to TensorBoard for any play.py algorithm.
python learning/play.py --robot panda --task reach_obstacles --algo ik_rrt_connect \
    --episodes 5 --seed 0 --headless --tensorboard
```

The plain `play.py` examples use task defaults. Use the scripts below for the
audited reference settings and reproducible seed.

For 100 reproducible `reach_obstacles` inference episodes per algorithm, run
`bash scripts/infer_star_worlds.sh`, `bash scripts/infer_differentiable_mpc.sh`,
or `bash scripts/infer_ik_rrt_connect.sh` from the repository root. Each script
has one playback command, uses seeds 0–99, and enables TensorBoard logging.
Star-Worlds uses the reference demo's 240 Hz physics, 40 Hz control, 0.02 m
goal tolerance, and 60 s trial limit. MPC uses 100 MuJoCo substeps per action
to match its 0.2 s model step. RRT-Connect uses the task's default 0.04 s
control step. The MPC and RRT-Connect scripts limit episodes to 20 s.

Selected numeric settings come from the reference implementations:

| Planner | Reference settings used in muarm | Adaptations |
|---------|----------------------------------|-------------|
| Star-Worlds | Franka demo: 0.30 m/s tip speed, 0.05 m obstacle inflation, modulation gain 1, weighting exponent 2, 0.01 m safety radius, 1.6 link activation, 0.30/0.08 avoidance/posture gains, 0.10 smoothing, 0.07 rad null-space cap, 40 Hz control, 0.02 m goal tolerance, 60 s trial limit | The paper states 0.10 m inflation, unlike its demo. The paper's obstacles are visual-only; muarm's are physical collision geoms. |
| Differentiable MPC | Official repository defaults: 8-step model horizon, 0.2 s model/control step, position weight 100 at every step, velocity weight 0.1, acceleration weight 0.02, linear slack weight 50,000, acceleration bound 4π rad/s² | The paper evaluates a 10-step horizon and 20 Hz controller. This point-goal Panda task has no orientation reference; it retains Panda joint limits, MuJoCo collision safeguards, and a 0.6 s solver limit. |
| IK → RRT-Connect | CoppeliaSim code: 1,000 IK trials, 1.5 m pre-IK tip-to-target distance limit, first IK solution, 6 s RRT search limit, 0.01 m minimum tip height, robot-obstacle/floor checks | The source requests at least 1,000 interpolated path states; muarm uses servo-sized waypoints for its finite MuJoCo episode. Its RRT step size and iteration cap are implementation choices because the source does not specify equivalents. |

The source algorithms run with different robots, simulation steps, geometry,
and target definitions. Matching their shared numeric settings does not make
these `reach_obstacles` runs identical to the published experiments.

The MuJoCo implementations use the cited code as algorithm references and
adapt the robot geometry and action interface to muarm. Star-Worlds adds an
exact MuJoCo robot-obstacle check before issuing an action. MPC uses Panda
link capsules in its optimizer, soft collision slack, and an exact MuJoCo
check of the first proposed action; its horizon model approximates the
environment's position servo, even when the model and control steps match.
Star-Worlds and MPC set a planner-local,
collision-free start posture before each episode; RRT-Connect uses the task
keyframe. RRT-Connect validates interpolated joint-space edges and rechecks
each tracked action. The task reports actual obstacle
contacts during playback; `play.py` prints whether any contact occurred and
the final tip-to-goal error.

`play.py` also prints end-effector trajectory length (meters) and cumulative
policy computation time (seconds) per episode, plus the final success rate.
With `--tensorboard`, it writes `rollout/computation_time_s` (episode setup and
policy decisions only) and `rollout/trajectory_length_m` (sum of executed
end-effector displacements sampled once per control step). Success rate remains
in the console output. Computation time excludes MuJoCo stepping, rendering,
and playback sleeps. Logs go to a new run under
`learning/runs/tb/inference/`; `--tensorboard-dir` changes the parent directory.
These single-seed scripts show trial outcomes, not estimated success rates.
Their whole-episode computation times are not directly comparable with the
per-update times reported in the Star-Worlds paper.
The existing `time/fps` TensorBoard tag comes from RL training and measures
environment steps per wall-clock second, not planner computation time.

### ▶️ IL — Behavior Cloning

```bash
source .venv/bin/activate

# Collect heuristic demos + train BC policy
python learning/train.py --robot panda --task reach --algo bc

# Visualize demo collection
python learning/train.py --robot panda --task reach --algo bc --render
```

### ▶️ Play — Real-time Policy Rollout

```bash
source .venv/bin/activate

# RL policy (SAC reach)
python learning/play.py --robot panda --task reach --algo sac \
    --model learning/runs/panda_reach_sac.zip --episodes 5

# Obstacle reach with CDF observations
python learning/play.py --robot panda --task reach_obstacles --algo sac \
    --model learning/runs/panda_reach_obstacles_sac.zip --episodes 5
python learning/play.py --robot panda --task reach_obstacles_2 --algo sac \
    --model learning/runs/panda_reach_obstacles_2_sac.zip --episodes 5

# BC policy
python learning/play.py --robot panda --task reach --algo bc \
    --model learning/runs/panda_reach_bc.pt --episodes 5

# Push / Pick-Place
python learning/play.py --robot panda --task push       --algo sac \
    --model learning/runs/panda_push_sac.zip
python learning/play.py --robot panda --task pick_place --algo sac \
    --model learning/runs/panda_pick_place_sac.zip

# SO-ARM100
python learning/play.py --robot so_arm --task reach --algo sac \
    --model learning/runs/so_arm_reach_sac.zip

# UFACTORY xArm7
python learning/play.py --robot xarm7 --task reach --algo sac \
    --model learning/runs/xarm7_reach_sac.zip
python learning/play.py --robot xarm7_nohand --task reach --algo sac \
    --model learning/runs/xarm7_nohand_reach_sac.zip
```

---

## 🎛️ CLI Arguments — `train.py` / `play.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--robot` | `panda` | `panda` \| `so_arm` \| `xarm7` \| `xarm7_nohand` |
| `--task` | `reach` | `reach` \| `reach_obstacles` \| `reach_obstacles_2` \| `cdflow_button` \| `push` \| `pick_place` |
| `--algo` | `sac` | RL/BC/planners, plus `cdflow` for direct button contact; `cdflow_mppi` is a play-only extension |
| `--family` | `rl` | `rl` \| `il` \| `planning` (auto-detected for BC/planners) |
| `--timesteps` | `200000` | Total RL training steps |
| `--n-envs` | `1` | Number of parallel environments |
| `--render` | `False` | Enable MuJoCo real-time viewer |
| `--tensorboard` | `False` | Enable TensorBoard logging |
| `--save-dir` | `learning/runs` | Model save directory |
| `--dataset` | — | Generated boundary NPZ, required for CDFlow training |
| `--cdflow-epochs` / `--cdflow-batch-size` | `100` / `32` | Offline CDFlow training settings |
| `--cdflow-device` | `cpu` | CDFlow training or inference PyTorch device |
| `--model` | — | Required saved model path for RL, BC, CDF-MPPI, and CDFlow playback |
| `--cdflow-mode` / `--cdflow-samples` | `mc` / `64` | CDFlow direction and boundary sample count; `regression` is the faster alternative |
| `--cdflow-scene-data` / `--cdflow-scene-id` | — / first | Select a recorded single-button scene for contact playback |
| `--cdflow-button-center` / `--cdflow-button-radius` | model center / `0.01` | Supply button geometry without a scene dataset |
| `--cdflow-time-limit-s` | `3.0` | Wall-clock seconds per contact trial |
| `--episodes` | `5` | Rollout episodes (play only) |
| `--headless` | `False` | Run play without viewer or real-time sleeps |
| `--max-steps` | — | Optional play step limit per episode |
| `--seed` | — | Seed the sampled play goals and RRT-Connect search |

---

## 🗂️ Model Save Paths

```
learning/runs/{robot}_{task}_{algo}.zip   ← RL  (Stable-Baselines3)
learning/runs/{robot}_{task}_{algo}.pt    ← IL  (Behavior Cloning)
learning/runs/{robot}_{task}_{algo}.pt    ← CDF-MPPI preparation or CDFlow training
```

---

## 🔌 Adding a New Robot

1. Create `learning/robots/my_robot.py` inheriting `MuJocoRobotEnv`
2. Implement `_build_spaces`, `_get_obs`, `_apply_action`, `get_ee_pos`, `get_ee_body`
3. Register it in `learning/envs/registry.py` under `make_env`
4. Use the same `train.py` / `play.py` commands with `--robot my_robot`

---

## 📝 TODO
