import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter

# TBD 주석 달린 파라미터 위주로 수정 하시오

# ── 센서 외부 파라미터 (실로봇 정본) ─────────────────────────────────────────
# 정본은 이 패키지 안: `resource/sensor_configuration.yaml` →
# 설치되면 `share/gdm_planning/resource/sensor_configuration.yaml`.
# 패키지에 들어 있으므로 실로봇 docker 의 마운트 구성이 달라도 그대로 동작한다.
# sim(sim_docker/workspace/scripts/run_rbq10_sim.py) 은 **같은 파일**을 심볼릭 링크
# (`sim_docker/sensor_configuration.yaml`, sim_docker/run.sh 가 이 정본으로 건다) 경유
# bind-mount 로 읽어 USD 센서 prim pose 와 static TF 를 만든다 — 그래서 실로봇 장착을
# 바꾸면 sim 이 같이 따라온다. gdm_docker 없이 sim 만 클론한 경우 run.sh 는 repo 사본
# `sim_docker/_sensor_configuration.yaml` 로 링크하므로, 정본을 바꾸면 그 사본도 갱신할 것.
# 센서 오프셋 숫자를 이 런치에 다시 적지 말 것 (그렇게 갈라진 사본 4개를 합친 결과다).
SENSOR_CONFIG_NAME = 'sensor_configuration.yaml'

# MID360 emergency 기둥의 천장 — **지면 기준** 높이 [m]. 이 위는 로봇이 부딪힐 수
# 없거나(로봇 최고점이 지면 위 0.75m 부근) 통과해야 할 구조물이다.
# 1.45 인 이유: 높이 1.5m 문의 상인방 **아래** 5cm 여유로 지나가기 위해서다 —
# 1.5 로 두면 상인방 모서리가 경계에 걸려 문 앞에서 정지한다.
# 노드 파라미터 livox_emergency_z_max 는 여기서 livox 지면高를 뺀 파생값이다.
EMERGENCY_CEILING_AGL = 1.45


def _default_sensor_config_path():
    """GUIDEDOG_SENSOR_CONFIG 우선, 없으면 설치된 패키지 리소스."""
    override = os.environ.get('GUIDEDOG_SENSOR_CONFIG')
    if override:
        return override
    return os.path.join(get_package_share_directory('gdm_planning'),
                        'resource', SENSOR_CONFIG_NAME)


def _load_sensor_config(path=None):
    """센서 extrinsics 로드. 없거나 깨지면 즉시 실패 — 조용한 기본값으로 폴백하면
    실로봇과 sim 이 갈라지는데 그걸 막는 게 이 파일의 존재 이유다."""
    path = path or _default_sensor_config_path()
    if not os.path.exists(path):
        raise RuntimeError(
            f'센서 설정 {path} 없음. 정본은 gdm_planning/resource/{SENSOR_CONFIG_NAME} '
            '이며 setup.py 가 share/gdm_planning/resource/ 로 설치한다 — '
            '`cbuild gdm_planning` 을 실행했는지 확인하라. '
            '다른 경로를 쓰려면 GUIDEDOG_SENSOR_CONFIG 로 지정.')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    tf = {}
    for child, tr in cfg['transforms'].items():
        xyz, quat = tuple(tr['xyz']), tuple(tr['quat_xyzw'])
        n = math.sqrt(sum(c * c for c in quat))
        if len(xyz) != 3 or len(quat) != 4 or abs(n - 1.0) > 1e-3:
            raise RuntimeError(f'{path}: {child} 의 xyz/quat_xyzw 가 잘못됐다 (|q|={n:.6f})')
        # 정규화 — sim 쪽 로더와 동일 처리 (실측 calib 값이 |q| 1e-7 수준으로 어긋나 있다).
        tf[child] = (tr['parent'], xyz, tuple(c / n for c in quat))
    for required in ('os_sensor', 'camera_link', 'livox_frame'):
        if required not in tf:
            raise RuntimeError(f'{path}: transforms 에 {required} 가 없다')
    for child in ('livox_frame', 'camera_link'):
        if tf[child][0] != 'os_sensor':
            raise RuntimeError(f"{path}: {child} 의 parent 는 os_sensor 여야 한다 "
                               f"(현재 {tf[child][0]}) — 노드 파라미터를 base_link "
                               '기준으로 접는 계산이 그걸 전제한다')
    return cfg, tf


# self-hit 섹터 1개를 이루는 키 — 노드 파라미터로 넘길 때의 순서이기도 하다.
SELF_HIT_KEYS = ('bearing_deg', 'half_angle_deg', 'elev_min_deg', 'elev_max_deg', 'max_range')


def _self_hit_sectors(cfg, frame, path=None):
    """cfg['self_hits'][frame] → 노드 파라미터용 평탄화 double 배열.

    [bearing_deg, half_angle_deg, elev_min_deg, elev_max_deg, max_range] × N.
    절이 없으면 빈 배열(= self-hit 마스크 없음)이 아니라 **실패**시킨다 — 조용히
    비우면 자기 몸체 반사가 상시 emergency 로 굳는다. 일부러 끄려면 빈 리스트를
    명시하라 (`self_hits: {livox_frame: []}`).
    """
    path = path or _default_sensor_config_path()
    try:
        sectors = cfg['self_hits'][frame]
    except (KeyError, TypeError):
        raise RuntimeError(
            f'{path}: self_hits.{frame} 절이 없다. 실측 self-hit 은 extrinsic 과 같은 '
            '장착 속성이라 이 파일이 정본이다 — scripts/measure_livox_self_hit.py 로 '
            '재서 채워라 (끄려면 빈 리스트를 명시).')
    flat = []
    for s in sectors or []:
        missing = [k for k in SELF_HIT_KEYS if k not in s]
        if missing:
            raise RuntimeError(f'{path}: self_hits.{frame} 항목 '
                               f'{s.get("name", "?")} 에 {missing} 가 없다')
        if s['half_angle_deg'] <= 0.0 or s['max_range'] <= 0.0 \
                or s['elev_min_deg'] >= s['elev_max_deg']:
            raise RuntimeError(f'{path}: self_hits.{frame} 항목 '
                               f'{s.get("name", "?")} 의 범위가 비어 있다')
        flat += [float(s[k]) for k in SELF_HIT_KEYS]
    return flat


def _rot_apply(q, v):
    """ROS 쿼터니언 q 로 벡터 v 회전 (R @ v)."""
    x, y, z, w = q
    return (
        (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - z * w) * v[1] + 2 * (x * z + y * w) * v[2],
        2 * (x * y + z * w) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - x * w) * v[2],
        2 * (x * z - y * w) * v[0] + 2 * (y * z + x * w) * v[1] + (1 - 2 * (x * x + y * y)) * v[2],
    )


def generate_launch_description():
    sensor_cfg, sensor_tf = _load_sensor_config()
    print(f'[rbq] 센서 extrinsics: {_default_sensor_config_path()}')
    base_height = float(sensor_cfg['base_height'])
    os_xyz, os_quat = sensor_tf['os_sensor'][1], sensor_tf['os_sensor'][2]
    # livox·camera 는 os_sensor 기준 → base_link 기준으로 접어 넣는다 (노드 파라미터가 body 기준).
    _lv_off = _rot_apply(os_quat, sensor_tf['livox_frame'][1])
    livox_xyz = tuple(os_xyz[i] + _lv_off[i] for i in range(3))
    _cam_off = _rot_apply(os_quat, sensor_tf['camera_link'][1])
    cam_xyz = tuple(os_xyz[i] + _cam_off[i] for i in range(3))
    # self-hit 은 센서 프레임 그대로 넘긴다 (노드도 livox_frame 에서 판정한다).
    livox_self_hits = _self_hit_sectors(sensor_cfg, 'livox_frame')
    print(f'[rbq] livox self-hit 섹터 '
          f'{len(livox_self_hits) // len(SELF_HIT_KEYS)}개: {livox_self_hits}')
    # emergency 최소 기둥(정지 시) — 정본은 sensor_configuration.yaml `emergency.base_link`.
    # 없으면 즉시 실패 (transforms·self_hits 와 같은 정책 — 조용한 기본값 금지).
    try:
        _em = sensor_cfg['emergency']['base_link']
        em_front_min = float(_em['front_min'])
        em_back = float(_em['back'])
        em_side = float(_em['side'])
        em_nose_x = float(_em['nose_x'])
    except (KeyError, TypeError):
        raise RuntimeError(
            f'{_default_sensor_config_path()}: emergency.base_link 절이 없다 — '
            '정지 시 최소 emergency 기둥(front_min/back/side)·앞코 nose_x 의 정본이 이 파일이다.')
    print(f'[rbq] emergency 최소 기둥(base_link): '
          f'-{em_back} <= x <= {em_front_min}, |y| <= {em_side}, 앞코 nose_x={em_nose_x}')
    # emergency 기둥 천장을 지면 기준 → livox_frame 기준으로 옮긴다 (장착 따라 자동 추종).
    livox_ground_z = base_height + livox_xyz[2]          # livox 지면高 [m]
    emergency_z_max = round(EMERGENCY_CEILING_AGL - livox_ground_z, 4)
    print(f'[rbq] livox 지면高 {livox_ground_z:.4f}m → emergency 기둥 천장 '
          f'z_max {emergency_z_max}m (지면 기준 {EMERGENCY_CEILING_AGL}m)')
    # sim(run_rbq10_sim.py) 또는 실로봇 odometry 가 발행하는 프레임 이름과 같아야 한다.
    sensor_tf_nodes = [
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'{parent}_to_{child}',
            arguments=['--x', str(xyz[0]), '--y', str(xyz[1]), '--z', str(xyz[2]),
                       '--qx', str(q[0]), '--qy', str(q[1]),
                       '--qz', str(q[2]), '--qw', str(q[3]),
                       '--frame-id', parent, '--child-frame-id', child],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            condition=IfCondition(LaunchConfiguration('sensor_tf')),
        )
        for child, (parent, xyz, q) in sensor_tf.items()
    ]
    # Isaac Sim 과 함께 돌릴 때만 true 로 override: `use_sim_time:=true`.
    # 실로봇에서는 /clock 발행자가 없으므로 기본값 false 를 유지해야 노드들이 wall clock 으로 정상 동작.
    use_sim_time = LaunchConfiguration('use_sim_time')

    # GridMap source selection.
    # Default: gdm_clustering3d  →  topic:=/gd/mobility_map        layer:=traversable  invert:=false  trav_threshold:=0.0
    # trav_map use:              →  topic:=/trav_map/trav_geo_map  layer:=erosion      invert:=true   trav_threshold:=0.1~0.5
    mobility_map_topic = LaunchConfiguration('mobility_map_topic')
    mobility_map_layer = LaunchConfiguration('mobility_map_layer')
    mobility_map_invert = LaunchConfiguration('mobility_map_invert')
    mobility_map_trav_threshold = LaunchConfiguration('mobility_map_trav_threshold')
    mobility_map_cost_weight = LaunchConfiguration('mobility_map_cost_weight')
    elevation_step_weight = LaunchConfiguration('elevation_step_weight')
    elevation_max_step = LaunchConfiguration('elevation_max_step')
    inflation_radius = LaunchConfiguration('inflation_radius')
    # 동적 장애물 회피
    dyn_obs_enable = LaunchConfiguration('dyn_obs_enable')
    stair_climb_speed = LaunchConfiguration('stair_climb_speed')
    stair_align_enable = LaunchConfiguration('stair_align_enable')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic')
    plan_dt = LaunchConfiguration('plan_dt')
    plan_timeout_s = LaunchConfiguration('plan_timeout_s')
    replan_static_dt = LaunchConfiguration('replan_static_dt')
    # LiDAR 지면 blind ring 바닥 강제 통과
    blind_floor_enable = LaunchConfiguration('blind_floor_enable')
    # RAPP 추종기 동역학 (하네스 사용자 편의 — 급가감속·급조향·제자리선회 완화)
    tracker_max_lin_acc = LaunchConfiguration('tracker_max_lin_acc')
    rapp_decel = LaunchConfiguration('rapp_decel')
    rapp_w_slew = LaunchConfiguration('rapp_w_slew')
    rapp_rotate_to_heading_deg = LaunchConfiguration('rapp_rotate_to_heading_deg')
    tracker_goal_pos_tol = LaunchConfiguration('tracker_goal_pos_tol')
    rapp_slow_horizon = LaunchConfiguration('rapp_slow_horizon')
    rapp_vx_min_speed = LaunchConfiguration('rapp_vx_min_speed')
    vy_min_speed = LaunchConfiguration('vy_min_speed')
    align_w_max = LaunchConfiguration('align_w_max')
    align_w_min = LaunchConfiguration('align_w_min')
    sd_rotate_wait_s = LaunchConfiguration('sd_rotate_wait_s')
    rapp_approach_ld_min = LaunchConfiguration('rapp_approach_ld_min')
    rapp_goal_line_lat = LaunchConfiguration('rapp_goal_line_lat')
    rapp_approach_lat_engage = LaunchConfiguration('rapp_approach_lat_engage')
    livox_emergency_front = LaunchConfiguration('livox_emergency_front')
    livox_emergency_ttc = LaunchConfiguration('livox_emergency_ttc')
    livox_emergency_back = LaunchConfiguration('livox_emergency_back')
    livox_emergency_side = LaunchConfiguration('livox_emergency_side')
    livox_emergency_z_max = LaunchConfiguration('livox_emergency_z_max')
    livox_emergency_margin_side = LaunchConfiguration('livox_emergency_margin_side')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use /clock (sim time) if true. Set true only when running with Isaac Sim.'),
        DeclareLaunchArgument(
            'mobility_map_topic', default_value='/gd/mobility_map',
            description='GridMap topic to subscribe for path planning. '
                        'Use /trav_map/trav_geo_map when using trav_map instead of gdm_clustering3d.'),
        DeclareLaunchArgument(
            'mobility_map_layer', default_value='traversable',
            description='Layer name inside the GridMap message to use as occupancy. '
                        'clustering3d: traversable  |  trav_map: erosion'),
        DeclareLaunchArgument(
            'mobility_map_invert', default_value='false',
            description='Set true when high layer value means FREE (e.g. trav_map erosion layer: 1.0=traversable). '
                        'Inverts values so that the planner treats high values as occupied.'),
        DeclareLaunchArgument(
            'mobility_map_trav_threshold', default_value='0.0',
            description='Minimum traversability to treat a cell as free [0.0-1.0]. '
                        'Lower = easier planning, higher = stricter. '
                        'clustering3d (invert=false): use 0.0. '
                        'trav_map erosion (invert=true): easy=0.1~0.3, moderate=0.3~0.6, strict=0.6~0.9'),
        DeclareLaunchArgument(
            'mobility_map_cost_weight', default_value='3.0',
            description='연속 traversability 비용 가중치. 0.0=기존 binary(threshold) 만, '
                        '>0=통과 가능 셀의 edge cost *= (1 + w*trav_cost) 로 저주행성 영역 회피. '
                        '하드 장애물/unknown 은 threshold 로 여전히 차단. 권장 1.0~5.0.'),
        DeclareLaunchArgument(
            'elevation_step_weight', default_value='2.0',
            description='elevation 경사 soft cost 가중치. >0 이면 인접 셀 높이차 |Δz| 에 비례해 '
                        'edge cost += w*|Δz| (평탄 경로 선호, 급경사 회피). 0=미사용. 권장 1.0~4.0.'),
        DeclareLaunchArgument(
            'elevation_max_step', default_value='0.2',
            description='단차 하드 차단 높이(m). 인접 셀 높이차 |Δz| 가 이 값 초과면 그 edge 통과 불가. '
                        '0=차단 안 함. 계단/연석 차단용. 로봇이 오를 수 있는 단차에 맞춰 조정.'),
        DeclareLaunchArgument(
            'inflation_radius', default_value='0.4',
            description='장애물 팽창 반경(m). 벽 셀로부터 이 거리 이내 통과가능 셀을 장애물로 팽창 → '
                        '경로가 벽에서 최소 이만큼 떨어진다(로봇 크기 충돌 방지). 0=팽창 안 함. '
                        'RBQ10: 몸통 0.8x0.20m, 다리 스탠스 전폭 ~0.44m → 반폭 0.22m + 여유 0.18m. '
                        '좁은 통로 검증(test/test_gate_clearance.py, 격자 위상 전수 스윕): '
                        '0.70~1.20m 통로와 개찰구 0.90m 레인 모두 통과하며, 0.3 으로 낮춰도 결과가 '
                        '같고 오히려 자유폭이 넓어져 경로가 중앙을 벗어난다 → 0.4 유지.'),
        DeclareLaunchArgument(
            'dyn_obs_enable', default_value='false',
            description='동적 장애물 검출·추적(gdm_obstacle_tracker_node) 및 회피 계획 활성화. '
                        '2026-10-07 기본 off — false 면 tracker 노드를 띄우지 않고 플래너도 '
                        '트랙을 구독하지 않는다(동적 스탬프·재계획 주기·dyn_scale 감속 전부 무효).'),
        DeclareLaunchArgument(
            'stair_align_enable', default_value='true',
            description='계단 각도 정렬(계단면 수직 방향 연속 heading 보정) 활성화. '
                        'false 면 자체 정렬을 끄고 외부 모듈/센서가 정렬을 담당한다 '
                        '(lpp-* 스크립트는 false 로 띄운다). 계단 폭 기반 lateral '
                        '횡보정은 이 인자와 무관하게 항상 동작.'),
        DeclareLaunchArgument(
            'stair_climb_speed', default_value='0.6',
            description='계단 모드(등반/하강/평탄 전진) 선속 [m/s] — 계단 vx 의 유일한 '
                        '상한이다 (2026-08-05 지시: 계단 0.6 제한). 이 값이 계단 중 '
                        'emergency 동적 전방 상한에도 들어간다 '
                        '(0.6 × livox_emergency_ttc 1.5s = 0.9 m → stair_emg_front_max 0.5 m 로 캡). 정렬 게이트는 '
                        '히스테리시스(이탈 2×임계)로 채터링을 막고, vx 는 '
                        'tracker_max_lin_acc 로 램프된다 — 계단 모드는 cmd_vel 을 '
                        '직접 발행해 RAPP rate limiter 를 우회하기 때문.'),
        DeclareLaunchArgument(
            'pointcloud_topic', default_value='/ouster/points',
            description='장애물 추적기가 구독할 3D 포인트클라우드 토픽.'),
        DeclareLaunchArgument(
            'plan_dt', default_value='0.5',
            description='(재)계획 주기 [s]. 동적 회피 반응성을 위해 낮게 유지.'),
        DeclareLaunchArgument(
            'plan_timeout_s', default_value='0.45',
            description='1회 계획 허용 시간 [s]. plan_dt 이하 권장.'),
        DeclareLaunchArgument(
            'replan_static_dt', default_value='2.0',
            description='동적 장애물이 없을 때의 추종 중 재계획 주기 [s]. plan_dt 보다 '
                        '크게 — 잦은 경로 교체로 인한 좌우 움찔거림 방지. 동적 장애물 '
                        '감지·재계획 실패 시엔 plan_dt 주기로 자동 복귀.'),
        DeclareLaunchArgument(
            'blind_floor_enable', default_value='true',
            description='LiDAR 하방 FoV 밖 지면 blind ring(OS1-64 0.9m/22.5° ≈ 2.17m) 안의 '
                        '미관측 "바닥" 셀을 free 로 강제. elevation 은 불변, D455 가시 wedge 는 제외. '
                        'unknown→obstacle 처리로 로봇이 blind ring 에 갇혀 계획 불가능해지는 것 방지.'),
        DeclareLaunchArgument(
            'sensor_tf', default_value='true',
            description='sensor_configuration.yaml 의 센서 extrinsic static TF '
                        '(base_link→os_sensor / os_sensor→camera_link / os_sensor→livox_frame) '
                        '를 이 런치가 발행한다. 다른 발행자가 있으면 false 로 끌 것 — '
                        'Isaac Sim(run_rbq10_sim.py) 과 trav_map_bag.launch.py 는 '
                        '자체 발행하므로 lpp-sim / lpp-bag 이 false 를 넘긴다.'),
        # ── RAPP 추종기 동역학 (하네스 사용자 편의) ──────────────────────────
        # 낮출수록 부드럽지만 반응성↓. emergency 정지는 zero Twist 직접 발행이라 무관.
        DeclareLaunchArgument(
            'tracker_max_lin_acc', default_value='0.8',
            description='선가속 한계 [m/s^2]. RAPP 가 vx 를 이 기울기로 rate-limit — 재계획으로 '
                        '목표속도가 바뀌거나 가감속 시 홱 당김 방지. 4.0(≈0.1s 전속)은 하네스 충격.'),
        DeclareLaunchArgument(
            'rapp_decel', default_value='0.8',
            description='감속 한계 [m/s^2]. 도착·재계획 감속 램프(v=√(2·decel·d))를 부드럽게 — 급제동 충격 방지.'),
        DeclareLaunchArgument(
            'rapp_w_slew', default_value='2.0',
            description='각가속 한계 [rad/s^2]. 낮출수록 코너 조향이 완만하게 붙고 재계획 직후 w '
                        '계단점프를 램프로 흡수 → 하네스가 옆으로 홱 당겨지지 않는다.'),
        DeclareLaunchArgument(
            'rapp_rotate_to_heading_deg', default_value='100.0',
            description='제자리 선회 진입 각 [deg]. 이 각 미만 코너는 원호로 통과, 이상만 정지 후 회전. '
                        '높일수록 "정지 후 회전" 끊김이 줄지만 하이핀에서 회전반경↑.'),
        DeclareLaunchArgument(
            'tracker_goal_pos_tol', default_value='0.4',
            description='도착 판정 반경 [m]. GP 에서 이 거리 이내면 완료·정지, 접근 감속도 이 지점에서 '
                        'v→0. GP 돌파(관성 코스트로 지나침) 시 키워 더 일찍 정지. '
                        '노드 arrive_tol 은 이 값 +0.15 로 자동 추종 (full path 부분경로 오인 방지).'),
        DeclareLaunchArgument(
            'rapp_slow_horizon', default_value='1.0',
            description='시간 기반 접근 감속 지평 [s]. GP 도달 약 이 시간 전부터 v ≤ d/T 로 서서히 '
                        '감속해 관성 돌파를 방지. 계단 GP·좁은 틈 통과 중에는 자동 제외. 0=off.'),
        DeclareLaunchArgument(
            'rapp_vx_min_speed', default_value='0.3',
            description='보행모델 vx dead zone [m/s] — 이 미만 선속 명령 무반응 (2026-08-07 실측). '
                        '접근 감속은 도착 판정까지 이 값 위로 유지하다 판정 순간 0 으로 끊는다. 0=off.'),
        DeclareLaunchArgument(
            'vy_min_speed', default_value='0.2',
            description='vx=0(제자리 게걸음) 일 때 vy 하한 [m/s] — 0 이 아닌 모든 vy 출력'
                        '(추종·계단 횡보정)이 발행 직전 이 값으로 스냅된다 (emergency 탈출은 '
                        '자체 밴드). 0.3 → 0.2 (2026-09-18 사용자 확인).'),
        DeclareLaunchArgument(
            'rapp_approach_ld_min', default_value='0.5',
            description='접근 구간 lookahead 하한 [m]. 잔여 거리가 짧으면 Ld 를 여기까지 줄여 '
                        '선회 진입 코너 컷(GP 옆으로 스침 → 돌파)을 없앤다.'),
        DeclareLaunchArgument(
            'rapp_goal_line_lat', default_value='0.7',
            description='goal-line 도착 corridor 반폭 [m]. GP 를 지나는 최종 진행방향 수선을 이 '
                        '횡오프셋 안에서 넘으면 도착 인정 — 톨 원을 스쳐도 돌파로 새지 않음. 0=off.'),
        DeclareLaunchArgument(
            'rapp_approach_lat_engage', default_value='0.15',
            description='접근 구간(잔여 2m) vy 개입 문턱 [m] (평시 rapp_lat_engage=0.25 대신) — '
                        '코너 컷 횡편차를 일찍 게걸음으로 지운다.'),
        DeclareLaunchArgument(
            'align_w_max', default_value='0.7',
            description='제자리 회전(시작/도착 정렬, rotate-to-heading) 순항 각속 [rad/s].'),
        DeclareLaunchArgument(
            # ★ 라이브(실로봇) 정본값. 스크린도어 탑승 후 하차 회전 예고→대기 시간.
            #   sim/bag 테스트는 lpp-sim, lpp-sim-travmap, lpp-bag 이 15.0 을 넘긴다.
            #   60 → 30 (2026-09-17) → 10 (2026-09-18) → 30 → 20 (2026-09-28 지시) → 60 (2026-09-29 지시) → 30 (2026-10-05, 실차). 예고 발화도 이 값으로 말하고,
            #   10초 단위 예고("이십초 전" 등) 후 마지막 10초는 1초마다 카운트다운 발화.
            'sd_rotate_wait_s', default_value='30.0',
            description='스크린도어 탑승 완료 후 하차 회전까지 대기 [s] (라이브 30, sim/bag 15).'),
        DeclareLaunchArgument(
            'align_w_min', default_value='0.4',
            description='제자리 회전 각속 하한 [rad/s] = yaw dead zone (실측: 그 밑 무반응). '
                        '남은 각 비례 감속이 이 값에서 멈추고 tol 도달 시 0 컷.'),
        # ── MID360 emergency 사각기둥 (x,y 는 base_link 기준, z 는 livox_frame 기준) ──────
        # 센서가 base_link 보다 0.296m 앞이므로 body 기준으로는 x -0.70~+0.80 이다.
        # 판정 전에 실측 self-hit(sensor_configuration.yaml `self_hits.livox_frame`)만
        # 빼고, 후방 섹터(하네스 사용자) 마스크는 **적용하지 않는다** — 후방 1m 를
        # 실제로 보기 위해서다. 즉 뒤를 바짝 따라오는 사용자는 정지를 유발할 수 있다.
        DeclareLaunchArgument(
            'livox_emergency_front', default_value=str(em_front_min),
            description='MID360 emergency 기둥 전방(+x) **최소** 깊이 [m], base_link 기준 '
                        '— 정지 상태에서 보는 범위. 주행 중에는 '
                        'max(이 값, v_fwd × livox_emergency_ttc) 로 동적 확장된다. '
                        '기본값 정본은 sensor_configuration.yaml `emergency.base_link.front_min` '
                        '— 수치가 side 와 같지만 별개 노브다.'),
        DeclareLaunchArgument(
            'livox_emergency_ttc', default_value='1.5',
            description='emergency 전방 TTC [s] — 실측 전진 속도로 이 시간 안에 닿을 정면 '
                        '점이면 emergency (조이스틱·외부 컨트롤러 포함). 거리는 로봇 앞코'
                        '(base_link + nose_x, yaml 정본)에서 잰다. 사용 값: odom 토픽 twist 를 '
                        'pose yaw 로 전방 투영한 v_fwd — twist 가 world frame 이라 linear.x '
                        '단독은 heading 에 따라 부호가 바뀐다. 정책 상한 0.8 m/s 에서 전방 1.6 m, 설계속도 '
                        '2.0 m/s 까지 캡 없이 비례. 0 이면 동적 확장 off (고정 front).'),
        DeclareLaunchArgument(
            'livox_emergency_back', default_value=str(em_back),
            description='MID360 emergency 기둥 후방(-x) 깊이 [m], base_link 기준. '
                        '기본값 정본은 sensor_configuration.yaml `emergency.base_link.back`. '
                        '**후방 섹터(livox_user_sector_deg 150°) 제외가 먼저 걸리므로 실제 '
                        '감시 깊이는 이 값이 아니라 섹터 경계에서 끊긴다** — 경계가 '
                        '|y|=tan75°·(livox 뒤 거리) 라 |y|<=side 면 livox 뒤 0.08m 부터 전부 섹터 안이다. '
                        '뒤를 더 깊게 보려면 이 값이 아니라 livox_user_sector_deg 를 줄여라.'),
        DeclareLaunchArgument(
            'livox_emergency_side', default_value=str(em_side),
            description='MID360 emergency 기둥 좌우(±y) 반폭 [m], base_link 기준. '
                        '기본값 정본은 sensor_configuration.yaml `emergency.base_link.side`. '
                        '0.90m 통로(개찰구 레인, 반폭 0.45m)에서는 횡편차 0.15m 부터 측벽이 '
                        '기둥에 들지만 좁은 틈 안 emergency 는 무시된다.'),
        DeclareLaunchArgument(
            'livox_emergency_z_max', default_value=str(emergency_z_max),
            description='MID360 emergency 기둥 상단 [m], livox_frame 기준(0=센서 수평면). '
                        f'기본값은 파생값이다: 지면 기준 천장 EMERGENCY_CEILING_AGL '
                        f'({EMERGENCY_CEILING_AGL}m) - livox 지면高 ({livox_ground_z:.4f}m) '
                        f'= {emergency_z_max}m. 높이 1.5m 문의 상인방 아래로만 보므로 문을 '
                        '통과한다. 바꾸려면 이 인자 대신 런치의 EMERGENCY_CEILING_AGL 을 '
                        '고쳐라 — 지면 기준이라야 장착이 바뀌어도 뜻이 유지된다.'),
        DeclareLaunchArgument(
            'livox_emergency_margin_side', default_value='0.05',
            description='좁은 틈 측벽 여유 [m] (footprint 밖). 벽이 로봇 반폭 0.22 + 이 값 '
                        '이내로 붙으면 정지 없이 vy 중앙 복귀(_gap_wall_repel_vy)를 낸다. '
                        '0.90m 레인의 횡편차 예산 0.45-0.22=0.23m 중 0.05 는 0.18m 에서 '
                        '발동해 22%만 잠식한다 (구 0.10 은 43%). '
                        '2026-08-05 이전엔 이 값이 emergency 사각형의 측면 반폭도 겸했다.'),
        SetParameter(name='use_sim_time', value=use_sim_time),
        Node(
            package='gdm_planning',
            executable='gdm_obstacle_tracker_node',
            name='gdm_obstacle_tracker',
            condition=IfCondition(dyn_obs_enable),
            parameters=[
                {'pointcloud_topic': pointcloud_topic},
                {'odom_topic': '/gdq/msg/gdq_odom'},
                {'tracks_topic': '/gdm/tracked_objects'},
                {'marker_topic': '/gdm/obstacle_markers'},
                {'world_frame_id': 'map'},
                # base_link → os_sensor 오프셋 — sensor_configuration.yaml 정본
                {'sensor_mount_x': float(os_xyz[0])},
                {'sensor_mount_y': float(os_xyz[1])},
                {'sensor_mount_z': float(os_xyz[2])},
                # 지면 제거: 지면 z = base_link z - base_height, 그 위 [z_min, z_max] 만 사용
                {'base_height': base_height},
                {'z_min': 0.15},
                {'z_max': 1.9},
                {'min_range': 0.5},
                {'max_range': 12.0},
                {'robot_exclude_radius': 0.8},
                # 클러스터링/추적
                {'cluster_cell_size': 0.15},
                {'cluster_min_points': 5},
                {'cluster_max_radius': 2.5},
                {'dyn_on_speed': 0.4},
                {'dyn_off_speed': 0.2},
                {'process_max_hz': 10.0},
            ],
            output='screen',
            emulate_tty=True,
        ),
        # map → odom identity. SLAM 보정이 필요하면 이 노드 대신 localization 이
        # 동적으로 발행한다. odom → base_link 는 sim(run_rbq10_sim.py) 또는 실로봇
        # odometry 가 발행하므로 여기서 건드리지 않는다.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom',
            arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
            output='screen',
        ),
        Node(
            package='gdm_planning',
            executable='gp_simulator',
            name='gdm_planning_gp_sim',
            parameters=[
            	{'position_topic': '/gdg/data/gp'},
                {'odometry_topic': '/gdq/msg/gdq_odom'},
            	{'goal_pose_topic': '/goal_pose'},
            ],
            output='screen',
            emulate_tty=True,  # 터미널 입력 활성화(for debug)
        ),
        Node(
            package='gdm_planning',
            executable='gdm_path_planning_node',
            name='gdm_planning_main',
            parameters=[
                {'world_frame_id': 'map'},
        		{'topic_local_plan': 'gdm/local_plan'},
        		{'topic_local_plan_map': 'gdm/local_plan_map'},
                {'topic_received_gp_marker': '/gdm/received_gp_marker'},
        		{'topic_cmd_vel': '/gdq/msg/cmd_vel'},
        		{'mobility_map_topic': mobility_map_topic},
                {'mobility_map_layer': mobility_map_layer},
                {'mobility_map_invert': mobility_map_invert},
                {'mobility_map_trav_threshold': mobility_map_trav_threshold},
                {'mobility_map_cost_weight': mobility_map_cost_weight},
                {'elevation_step_weight': elevation_step_weight},
                {'elevation_max_step': elevation_max_step},
                {'inflation_radius': inflation_radius},
        		{'gdg_gp_topic': '/gdg/data/gp'},
        		{'odom_topic': '/gdq/msg/gdq_odom'},
                {'sensor_offset_x': float(os_xyz[0])},  # base_link → os_sensor 전방 오프셋 (m, blind spot·계단 점군용; 계획/추종 원점은 base_link) — sensor_configuration.yaml 정본
                {'rapp_ctrl_point_x': float(os_xyz[0])},  # RAPP 조향 제어점 = 센서 위치 (조향 안정성; 도착 판정은 base_link)
                # MID360 장착 (base_link 기준) — 계단 검출 캐시가 livox 점군을 body 로 옮길 때 사용
                {'livox_mount_x': float(livox_xyz[0])},
                {'livox_mount_y': float(livox_xyz[1])},
                {'stair_livox_z': base_height + float(livox_xyz[2])},  # livox 지면高 (m)
                {'stair_ouster_z': base_height + float(os_xyz[2])},    # ouster 지면高 (m)
                {'sensor_blind_radius': 0.4},  # LiDAR 최소 감지 거리 이내 blind spot 반경 (m). 0.0 이면 비활성화.
                # LiDAR 지면 blind ring 바닥 강제 통과 (ring 반경 = height/tan(down_fov) ≈ 2.17m)
                {'blind_floor_enable': blind_floor_enable},
                {'blind_sensor_height': 0.9},   # OS1-64 지면 높이 (m) — 마진 포함 장착 가정
                {'blind_down_fov_deg': 22.5},   # OS1-64 하방 FoV (deg)
                {'blind_floor_max_step': 0.15}, # m. elevation 관측 셀의 "바닥" 인정 높이차
                {'blind_cam_fov_deg': 87.0},    # D455 depth 수평 FoV (deg) — 이 wedge 는 강제 제외
                {'blind_cam_min_dist': 0.9},    # m. D455 가 바닥을 보기 시작하는 거리 (pitch 14.8°↓, vFoV 58°)
                {'blind_cam_offset_x': float(cam_xyz[0])},  # m. base_link 기준 camera_link 전방 오프셋
                # (os_sensor 기준 camera_link 를 base_link 로 접은 값 — sensor_configuration.yaml 정본)
                # 개활지(일반 주행) 전진 속도 상한 [m/s] — 2026-08-05 지시로 1.0 → 0.8.
                # 좁은 통로(narrow_vx_max 0.5)·계단(stair_climb_speed 0.6)은 각자 더
                # 낮은 상한이 min 으로 겹쳐 걸린다. 전방 콘 1m 기준 정지 여유 1.25s.
                {'max_linear_speed': 1.0},
        		{'max_angular_speed': 1.0}, # 최대 각속도 (rad/s),  0.6 max speed in joystick's cmd_vel(from experiment)
                {'max_lateral_speed': 0.4}, # 최대 횡속 (m/s) — 홀로노믹 게걸음 보정(rapp 전용), vx 후진은 항상 금지
                # vx=0 일 때 vy 하한 (설명은 vy_min_speed 선언부) — 기본 0.2.
                {'vy_min_speed': vy_min_speed},
                # Planning
                {'plan_dt': plan_dt},           # sec. 재계획 주기 (낮출수록 반응성 ↑, CPU ↑)
                {'plan_timeout_s': plan_timeout_s},  # sec. 1회 계획 허용 시간 (plan_dt 이하 권장)
                # 동적 장애물 회피
                {'dyn_obs_enable': dyn_obs_enable},
                {'stair_climb_speed': stair_climb_speed},     # m/s. 계단 모드 선속
                {'stair_align_enable': stair_align_enable},   # 계단 각도 정렬 (lateral 횡보정은 항상 on)
                {'tracked_objects_topic': '/gdm/tracked_objects'},
                {'dyn_speed_thresh': 0.3},      # m/s. 이 이상이면 dynamic 으로 회피 계획에 반영
                {'dyn_obs_inflation': 0.3},     # m. 장애물 반경 안전 여유
                {'dyn_track_timeout': 1.5},     # s. 트랙 스탬프 유효 시간
                {'dyn_monitor_range': 3.0},     # m. 로봇 전방 이 거리 이내 트랙만 계획에 반영 (0=무제한)
                {'dyn_stop_distance': 0.9},     # m. 안전층 감속 하한 도달 거리 (정지는 MID360 emergency 전용)
                {'dyn_slow_distance': 2.5},     # m. 경로 침범 장애물 감속 시작 거리
                {'dyn_check_path_ahead': 3.0},  # m. 경로 전방 충돌 검사 길이
                {'dyn_robot_radius': 0.45},     # m. 안전층 로봇 반경 — RBQ10 외접 반경(몸통 반길이 0.40, 반폭 0.22 → ~0.46)
                {'replan_while_following': True},  # 추종 중 재계획 (동적: plan_dt / 정적: replan_static_dt 주기)
                {'replan_static_dt': replan_static_dt},  # sec. 동적 장애물 없을 때 재계획 주기 (움찔거림 방지)
                {'replan_fail_warn_count': 5},  # 재계획 연속 실패 경고 임계
                # 경로 스무딩 (좌우 움찔거림 완화)
                {'path_smooth_window': 7},   # 경로 이동평균 윈도(포인트, 0.1m 간격 ≈ 0.7m 창)
                {'flow_smooth_iters': 2},    # flow 플래너 Chaikin 코너 라운딩 반복 (충돌 검증 포함)
                {'flow_prefer_straight': True},  # 장애물 없고 직선이 자유로우면 직선 경로 (선형 주행)
                {'flow_trav_safe': 0.8},     # 이 traversability 이상은 비용 0 → 안전 영역에선 최단 경로
                {'flow_center_weight': 1.0}, # 길 가운데 선호 — 벽 근처(ref_dist 이내)만
                {'flow_center_ref_dist': 0.6},
                # RAPP 추종기 동역학 (하네스 사용자 편의 — 급가감속·급조향·제자리선회 완화)
                {'tracker_max_lin_acc': tracker_max_lin_acc},
                {'rapp_decel': rapp_decel},
                {'rapp_w_slew': rapp_w_slew},
                {'rapp_rotate_to_heading_deg': rapp_rotate_to_heading_deg},
                {'tracker_goal_pos_tol': tracker_goal_pos_tol},
                {'rapp_slow_horizon': rapp_slow_horizon},
                {'rapp_vx_min_speed': rapp_vx_min_speed},
                {'align_w_max': align_w_max},
                {'align_w_min': align_w_min},
                {'sd_rotate_wait_s': sd_rotate_wait_s},
                {'rapp_approach_ld_min': rapp_approach_ld_min},
                {'rapp_goal_line_lat': rapp_goal_line_lat},
                {'rapp_approach_lat_engage': rapp_approach_lat_engage},
                # MID360 emergency 사각기둥 (x,y base_link 기준 · z livox_frame 기준)
                {'livox_emergency_front': livox_emergency_front},
                {'livox_emergency_ttc': livox_emergency_ttc},
                {'livox_emergency_back': livox_emergency_back},
                {'livox_emergency_side': livox_emergency_side},
                {'livox_nose_x': em_nose_x},   # TTC 원점 = 로봇 앞코 (base_link 전방) — yaml 정본
                {'livox_emergency_z_max': livox_emergency_z_max},
                {'livox_emergency_margin_side': livox_emergency_margin_side},
                {'robot_half_width': 0.22},   # 로봇 반폭 [m] (y) — 다리 스탠스 전폭 ~0.44m
                {'livox_max_range': 1.5},     # MID360 3D 센싱 캡 [m] — refine·좁은 틈 전용
                # 무효 리턴 컷 [m]. 드라이버가 빈 빔을 (0,0,0) 으로 내보내는데(프레임의
                # 약 12%) 그 원점이 emergency 기둥 한복판이라 안 지우면 상시 정지한다.
                {'livox_min_range': 0.05},
                # MID360 self-hit = 실측 구면 섹터 (정본: sensor_configuration.yaml
                # self_hits.livox_frame). emergency·refine 양쪽에서 이것만 자기 반사로 뺀다.
                {'livox_self_hit_sectors': livox_self_hits},
                # 하네스 사용자 = 센서 후방 섹터 전체 (거리·높이 무관). emergency 를 포함한
                # **모든 판정**에서 뺀다. 신형 하네스는 사용자가 우측에 나란히 서기도 해 ±75°
                # (2026-10-05). 기둥은 livox 뒤 0.08m(base_link x≈+0.19)에서 잘린다.
                {'livox_user_sector_deg': 150.0},  # 후방 섹터 중심각 [deg] (좌후 75° ~ 우후 75°)
                # 좁은 공간 = 폭 1m 이내 통로. 이 폭 이하면: 재계획 중단 + vx 캡 +
                # emergency 무시(중앙 복귀 vy 로 대체).
                {'livox_narrow_width': 1.0},
                {'narrow_vx_max': 0.5},       # 좁은 틈 통과 중 전진 속도 상한 [m/s]
                # 좁은 틈 중앙 복귀 vy = gain × 중앙편차 (상한 emergency_vy_max).
                # 고정 ±0.3 이던 시절 개찰구 좌우 진동·측벽 긁힘 (2026-08-07 실측).
                {'gap_repel_vy_gain': 2.0},
                # 측면 장애물 대응 vy(탈출·중앙 복귀) 밴드 [m/s]. 상한 0.3 은 보행모델
                # dead zone(|vy|<=0.3 무반응) 경계값 — 횡반응이 약하면 이걸 의심하라.
                {'livox_escape_vy': 0.3},
                {'emergency_vy_min': 0.1},
                {'emergency_vy_max': 0.3},

                # 아래의 맵 관련된 값은 clustering3d_launch.xml 의 값과 일치해야 하므로, 가능하면 건드리지 말것
                {'map_resol': 0.1},  # clustering3d_launch.xml 의 resolution과 값이 일치해야 함.
                {'map_center_x': 0},
                {'map_center_y': 0},
                {'map_size_x': 200},
                {'map_size_y': 200},
                # Tracking
                {'track_dt': 0.01},  # sec.
                {'goal_stop_distance': 0.25},  # meter
                {'gp_modify_distance': 5.0},  # GP가 벽 위에 있을 때 GP를 로봇쪽으로 당기는 최대 거리 in meter
                {'gp_modify_enable': True},   # GP 장애물 보정 on/off. 디버그 시 false 로 두면 GP를 당기지 않음                
                {'gp_next_enable': True},     # GP0 이 VIA 이고 GP1 이 있으면 GP1 까지 이어 계획 → GP0 무정지 통과 (RAPP 구간만)
            ],
            output='screen',
            emulate_tty=True,  # 터미널 입력 활성화(for debug)
        ),
    ] + sensor_tf_nodes)
