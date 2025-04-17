import genesis as gs
import torch
import numpy as np
gs.init(backend=gs.cuda)

# ───────────────────────────────────────── Scene & Robot ─────────────────────────────────────────
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
    dt=0.02, 
    substeps=4),
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    show_FPS=False,
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,        # Enable collision
        enable_self_collision=False,   # Enable self-collision
        enable_joint_limit=True,      # Enable joint limits
    ),
    renderer=gs.renderers.Rasterizer(),   
)

robot_path = "resources/" + "robots/flamingo_light/" + "xml/" + "flamingo_light_v1"+ ".xml"
urdf_path = "resources/" + "plane/" + "plane.urdf"

scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True),)


robot = scene.add_entity(
    gs.morphs.MJCF(
        file=robot_path,

    ),
    vis_mode="visual"
)

cam = scene.add_camera(
    res=(640, 480), pos=(3.5, 0.0, 2.5), lookat=(0, 0, 0.5), fov=30, GUI=False
)

scene.build(n_envs=1)      # 1 환경이라도 내부 인터페이스는 (B, N_dof)

# ───────────────────────────────────────── DOF 분리 ─────────────────────────────────────────
pos_joints = ["left_shoulder_joint", "right_shoulder_joint"]   # 위치 제어
vel_joints = ["left_wheel_joint",    "right_wheel_joint"]      # 속도 제어

pos_idx = [robot.get_joint(n).dof_idx_local for n in pos_joints]
vel_idx = [robot.get_joint(n).dof_idx_local for n in vel_joints]

print(f"pos_idx: {pos_idx}")
print(f"vel_idx: {vel_idx}")

# ────────────────────── 0) 파라미터 ──────────────────────
POS_MIN   = -1.25
POS_MAX   =  0.0
STEP_SIZE =  0.01          # 0.01 rad/step ≈ 0.6 rad/s (dt=0.0166 s일 때)

# ────────────────────── 1) 드라이브 모드 & 토크 한계 ──────────────────────

# ────────────────────── 2) 게인 튜닝 ──────────────────────
robot.set_dofs_kp(np.array([30.0, 30.0]), dofs_idx_local=pos_idx)  # ↑추종성 향상
robot.set_dofs_kv(np.array([1.0, 1.0]), dofs_idx_local=pos_idx)
robot.set_dofs_kp(np.array([0.0, 0.0]), dofs_idx_local=vel_idx)  # ↑추종성 향상
robot.set_dofs_kv(np.array([0.0, 0.0]), dofs_idx_local=vel_idx)

# ────────────────────── 3) 목표 값 초기화 ──────────────────────
target_pos = np.zeros((1, 2), dtype=np.float32)
target_vel = np.zeros((1, 2), dtype=np.float32)

current   = POS_MAX   # 0.0
direction = -1        # 감소부터 시작

scene.step()  # 초기 스텝

# ────────────────────── 4) 제어 루프 ──────────────────────
while True:
    # ① 삼각파 생성
    current += direction * STEP_SIZE
    if current <= POS_MIN:
        current, direction = POS_MIN, +1
    elif current >= POS_MAX:
        current, direction = POS_MAX, -1

    target_pos[0, :] = current   # 두 어깨 동일 각도

    # ② 명령 전송
    robot.control_dofs_position(target_pos, dofs_idx_local=pos_idx)
    robot.control_dofs_velocity(target_vel, dofs_idx_local=vel_idx)

    # ③ 시뮬레이션 스텝
    scene.step()
    cam.render()

    # ④ 모니터링
    # print(f"current target: {current:.3f} | measured q: {robot.get_dofs_position(pos_idx).cpu().numpy()}")

    # ⑤ 종료 조건
    geom_names_for_termination = [
    "base_collision",
    "left_foot_collision",
    "right_foot_collision",
    ]


