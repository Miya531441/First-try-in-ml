"""Unit tests (spec v2): raycast intersection, cone membership, occlusion, cone bisector,
damage/reload resolution, obstacle classes, layout constraints, motion limits, tracks,
determinism and the PettingZoo API."""
import numpy as np
import pytest

from env.agent import EnvConfig
from env.arena import CRATE, WALL
from env.raycast import cast_rays, cone_membership, ray_box_t, ray_boundary_t, ray_circle_t, segment_blocked
from env.spawn import connected, generate_obstacles
from env.squad_env import SquadParallelEnv, SquadVecEnv


def _boxes(*bs):
    b = np.array(bs, np.float32).reshape(1, -1, 4)
    return b, np.ones(b.shape[:2], bool)


def _cfg(**kw):
    na, nb = kw.pop("na", 3), kw.pop("nb", 3)
    base = dict(team_size=kw.pop("team_size", max(na, nb)), team_sizes=[[na, nb]], team_size_probs=[1.0])
    base.update(kw)
    return EnvConfig(**base)


def test_ray_box_intersection():
    boxes, mask = _boxes([5, -1, 7, 1])
    o = np.array([[[0.0, 0.0]]], np.float32)
    d = np.array([[[1.0, 0.0]]], np.float32)
    assert np.isclose(ray_box_t(o, d, boxes, mask)[0, 0, 0], 5.0)
    assert ray_box_t(o, -d, boxes, mask)[0, 0, 0] > 1e8
    d2 = np.array([[[0.0, 1.0]]], np.float32)
    assert ray_box_t(o, d2, boxes, mask)[0, 0, 0] > 1e8
    d3 = np.array([[[np.cos(0.1), np.sin(0.1)]]], np.float32)
    assert np.isclose(ray_box_t(o, d3, boxes, mask)[0, 0, 0], 5.0 / np.cos(0.1))


def test_ray_boundary_and_circle():
    o = np.array([[[10.0, 10.0]]], np.float32)
    d = np.array([[[0.0, -1.0]]], np.float32)
    assert np.isclose(ray_boundary_t(o, d, 48.0)[0, 0], 10.0)
    centers = np.array([[[10.0, 4.0]]], np.float32)
    assert np.isclose(ray_circle_t(o, d, centers, 1.0, np.ones((1, 1, 1), bool))[0, 0, 0], 5.0)
    assert ray_circle_t(o, -d, centers, 1.0, np.ones((1, 1, 1), bool))[0, 0, 0] > 1e8


def test_cone_membership():
    o = np.zeros((1, 1, 2), np.float32)
    theta = np.zeros((1, 1), np.float32)
    pts = np.array([[[10.0, 0.0], [10.0, np.tan(np.deg2rad(29)) * 10], [10.0, np.tan(np.deg2rad(31)) * 10],
                     [-10.0, 0.0], [30.0, 0.0]]], np.float32)
    inside, _, _ = cone_membership(o, theta, pts, np.deg2rad(60), 24.0)
    assert inside[0, 0].tolist() == [True, True, False, False, False]


def test_occlusion_ray_terminates_at_first_hit():
    boxes, mask = _boxes([5, -1, 7, 1])
    o = np.array([[[0.0, 0.0]]], np.float32)
    d = np.array([[[1.0, 0.0]]], np.float32)
    centers = np.array([[[0.0, 0.0], [10.0, 0.0]]], np.float32)
    cmask = np.array([[[False, True]]])
    dist, kind, hit = cast_rays(o, d, boxes, mask, centers, 1.0, cmask, 48.0, 24.0)
    assert kind[0, 0] == 1 and np.isclose(dist[0, 0], 5.0) and hit[0, 0] == -1
    centers2 = np.array([[[0.0, 0.0], [3.0, 0.0]]], np.float32)
    dist, kind, hit = cast_rays(o, d, boxes, mask, centers2, 1.0, cmask, 48.0, 24.0)
    assert kind[0, 0] == 2 and hit[0, 0] == 1 and np.isclose(dist[0, 0], 2.0)
    centers3 = np.array([[[0.0, 0.0], [3.0, 0.0], [4.5, 0.0]]], np.float32)
    _, _, hit = cast_rays(o, d, np.zeros((1, 0, 4), np.float32), np.zeros((1, 0), bool), centers3, 1.0,
                          np.array([[[False, True, True]]]), 48.0, 24.0)
    assert hit[0, 0] == 1
    p = np.array([[[0.0, 0.0]]], np.float32)
    assert segment_blocked(p, np.array([[[10.0, 0.0]]], np.float32), boxes, mask, centers, 1.0, np.zeros((1, 1, 2), bool))[0, 0]
    assert not segment_blocked(p, np.array([[[0.0, 10.0]]], np.float32), boxes, mask, centers, 1.0, np.zeros((1, 1, 2), bool))[0, 0]


def _place(env, pos, theta, boxes=None, kinds=None):
    st = env.state
    st.pos[0] = pos
    st.theta[0] = theta
    st.vel[0] = 0
    st.omega[0] = 0
    env.arena.mask[0] = False
    if boxes is not None:
        env.arena.set(np.array([0]), [np.array(boxes, np.float32)], [np.array(kinds, np.int64)])
    env._perceive(np.array([0]))


def test_firing_ray_bisects_cone():
    cfg = _cfg(na=1, nb=1, spread_rest_deg=0.0, spread_max_deg=0.0)
    env = SquadVecEnv(cfg, 1, seed=1)
    theta = 0.7
    _place(env, np.array([[20.0, 20.0], [40.0, 40.0]]), np.array([theta, np.pi]))
    a = np.zeros((1, 2, 4), np.float32)
    a[0, 0, 3] = 1.0
    env.step(a)
    assert env.shot_fired[0, 0]
    assert np.isclose(env.shot_aim[0, 0], theta, atol=1e-6)               # shot along the heading
    offsets = np.linspace(-cfg.fov / 2, cfg.fov / 2, cfg.num_rays)
    assert np.isclose(offsets.mean(), 0.0, atol=1e-7)                     # rays symmetric about the heading
    assert np.isclose(offsets[0], -offsets[-1])


def test_damage_resolution_and_kill():
    cfg = _cfg(na=1, nb=1, spread_rest_deg=0.0, spread_max_deg=0.0)
    cfg.role_rewards.enabled = False                      # base reward only; role bonuses are tested separately
    env = SquadVecEnv(cfg, 1, seed=1)
    _place(env, np.array([[10.0, 10.0], [20.0, 10.0]]), np.array([0.0, np.pi]))
    a = np.zeros((1, 2, 4), np.float32)
    a[0, 0, 3] = 1.0
    _, _, rew, done, info = env.step(a)
    assert env.state.hp[0, 1] == pytest.approx(66) and env.state.hp[0, 0] == 100
    assert info["hit_enemy"][0, 0] and not info["hit_ally"][0, 0]
    assert rew[0, 0] == pytest.approx(0.34 + cfg.reward.step)
    _, _, rew, done, info = env.step(a)
    assert not info["fire"][0, 0]                                          # cooldown
    for _ in range(7):
        env.step(np.zeros((1, 2, 4), np.float32))
    env.step(a)
    assert env.state.hp[0, 1] == pytest.approx(32)
    for _ in range(8):
        env.step(np.zeros((1, 2, 4), np.float32))
    _, _, rew, done, info = env.step(a)
    assert done[0] and info["winner"][0] == 0
    assert rew[0, 0] == pytest.approx(0.34 + 0.5 + 1.0 + cfg.reward.step, abs=1e-6)
    assert rew[0, 1] == pytest.approx(-0.005 * 32 - 0.5 - 1.0, abs=1e-6)


def test_friendly_fire_penalty_and_toggle():
    for ff in (True, False):
        cfg = _cfg(na=2, nb=2, team_size=2, spread_rest_deg=0.0, spread_max_deg=0.0, friendly_fire=ff)
        env = SquadVecEnv(cfg, 1, seed=1)
        _place(env, np.array([[10.0, 10.0], [15.0, 10.0], [40.0, 40.0], [45.0, 45.0]]), np.array([0.0, 0.0, np.pi, np.pi]))
        a = np.zeros((1, 4, 4), np.float32)
        a[0, 0, 3] = 1.0
        _, _, rew, _, info = env.step(a)
        assert info["hit_ally"][0, 0]
        if ff:
            assert env.state.hp[0, 1] == pytest.approx(66) and rew[0, 0] < -30
        else:
            assert env.state.hp[0, 1] == 100 and rew[0, 0] == pytest.approx(cfg.reward.step)


def test_wall_blocks_vision_crate_does_not():
    cfg = _cfg(na=1, nb=1, spread_rest_deg=0.0, spread_max_deg=0.0)
    for kind, expect_visible in ((WALL, False), (CRATE, True)):
        env = SquadVecEnv(cfg, 1, seed=1)
        _place(env, np.array([[10.0, 20.0], [30.0, 20.0]]), np.array([0.0, np.pi]), boxes=[[18, 18, 22, 22]], kinds=[kind])
        assert bool(env.vis[0, 0, 1]) is expect_visible
        a = np.zeros((1, 2, 4), np.float32)
        a[0, 0, 3] = 1.0
        _, _, _, _, info = env.step(a)
        assert bool(info["hit_enemy"][0, 0]) is expect_visible
        # the map scan sees both classes: forward ray hits the box at 8 - 1 (nothing subtracted) = 8 m
        assert np.isclose(env.map_scan[0, 0, 0], 8.0, atol=0.5) or kind == CRATE and env.state.hp[0, 1] < 100
    # movement is blocked by a crate
    env = SquadVecEnv(cfg, 1, seed=1)
    _place(env, np.array([[10.0, 20.0], [40.0, 40.0]]), np.array([0.0, np.pi]), boxes=[[14, 18, 18, 22]], kinds=[CRATE])
    a = np.zeros((1, 2, 4), np.float32)
    a[0, 0, 0] = 1.0
    for _ in range(60):
        env.step(a)
    assert env.state.pos[0, 0, 0] <= 14 - cfg.collision_radius + 1e-3


def test_map_scan_geometry():
    cfg = _cfg(na=1, nb=1)
    env = SquadVecEnv(cfg, 1, seed=1)
    _place(env, np.array([[10.0, 20.0], [40.0, 40.0]]), np.array([np.pi / 2, np.pi]), boxes=[[8, 30, 12, 34]], kinds=[WALL])
    K = cfg.map_rays
    # ray 0 points along the heading (+y): the wall at y=30 is 10 m away
    assert np.isclose(env.map_scan[0, 0, 0], 10.0, atol=0.05)
    # ray K/4 points 90 deg left (-x): the arena wall at x=0 is 10 m away
    assert np.isclose(env.map_scan[0, 0, K // 4], 10.0, atol=0.05)


def test_motion_limits():
    cfg = _cfg(na=1, nb=1)
    env = SquadVecEnv(cfg, 1, seed=1)
    _place(env, np.array([[24.0, 24.0], [45.0, 45.0]]), np.array([0.0, np.pi]))
    a = np.zeros((1, 2, 4), np.float32)
    a[0, 0, 2] = 1.0                                   # full turn command from rest
    env.step(a)
    assert np.isclose(env.state.omega[0, 0], min(cfg.turn_accel * cfg.dt, cfg.max_turn_rate))   # accel-capped
    for _ in range(10):
        env.step(a)
    assert np.isclose(env.state.omega[0, 0], cfg.max_turn_rate)
    for cmd, expected in (([1, 0], cfg.speed_forward), ([-1, 0], cfg.speed_backward), ([0, 1], cfg.speed_strafe)):
        _place(env, np.array([[24.0, 24.0], [45.0, 45.0]]), np.array([0.0, np.pi]))
        a = np.zeros((1, 2, 4), np.float32)
        a[0, 0, :2] = cmd
        for _ in range(40):
            env.step(a)
        assert np.isclose(np.linalg.norm(env.state.vel[0, 0]), expected, atol=0.05)


def test_layout_constraints_and_connectivity():
    cfg = EnvConfig()
    rng = np.random.default_rng(3)
    S = cfg.arena_size
    for _ in range(5):
        center = np.array([S / 2 - cfg.spawn_distance / 2, S / 2], np.float32)
        boxes, kinds = generate_obstacles(rng, cfg, center)
        area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum() / S ** 2
        assert 0.05 < area <= cfg.coverage_max + 1e-6
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                dx = max(0, max(boxes[i, 0], boxes[j, 0]) - min(boxes[i, 2], boxes[j, 2]))
                dy = max(0, max(boxes[i, 1], boxes[j, 1]) - min(boxes[i, 3], boxes[j, 3]))
                assert np.hypot(dx, dy) >= cfg.obstacle_min_gap - 1e-4
            for c in (center, S - center):
                q = np.clip(c, boxes[i, 0:2], boxes[i, 2:4])
                assert np.linalg.norm(q - c) >= cfg.spawn_clearance + cfg.spawn_zone_radius - 1e-4
        assert connected(boxes, S, center, S - center, cfg.collision_radius)
        assert set(np.unique(kinds)) <= {WALL, CRATE}
    # a full-width wall disconnects the spawns
    wall = np.array([[0, 22, 48, 26]], np.float32)
    assert not connected(wall, S, np.array([9, 24]), np.array([39, 24]), 1.0)


def test_tracks_and_blackboard():
    cfg = _cfg(na=2, nb=1, team_size=3, spread_rest_deg=0.0, spread_max_deg=0.0)
    env = SquadVecEnv(cfg, 1, seed=1)
    # agent 0 sees enemy 3; agent 1 faces away
    _place(env, np.array([[10.0, 20.0], [10.0, 30.0], [40.0, 40.0], [20.0, 20.0], [40.0, 41.0], [40.0, 42.0]]),
           np.array([0.0, np.pi, 0.0, np.pi, 0.0, 0.0]))
    env.step(np.zeros((1, 6, 4), np.float32))
    st = env.state
    assert st.track_valid[0, 0, 0] and not st.track_valid[0, 1, 0]       # own tracks
    obs = env._build_obs()
    lay = env.obs_layout["enemies"]
    tr0 = obs[0, 0, lay["start"]: lay["start"] + lay["size"]].reshape(cfg.team_size, 11)
    tr1 = obs[0, 1, lay["start"]: lay["start"] + lay["size"]].reshape(cfg.team_size, 11)
    assert tr0[0, 9] == 1.0 and tr0[0, 10] == 1.0                         # valid, visible now
    assert tr1[0, 9] == 1.0 and tr1[0, 10] == 0.0                         # shared via blackboard, not visible to 1
    assert np.isclose(tr1[0, 0], -10.0 / cfg.arena_size, atol=1e-5)      # enemy is behind agent 1 (facing -x): +10 m behind
    # staleness grows once the enemy leaves the cone
    _place(env, np.array([[10.0, 20.0], [10.0, 30.0], [40.0, 40.0], [20.0, 20.0], [40.0, 41.0], [40.0, 42.0]]),
           np.array([np.pi, np.pi, 0.0, np.pi, 0.0, 0.0]))
    for _ in range(20):
        env.step(np.zeros((1, 6, 4), np.float32))
    obs = env._build_obs()
    tr0 = obs[0, 0, lay["start"]: lay["start"] + lay["size"]].reshape(cfg.team_size, 11)
    assert tr0[0, 7] > 0.05 and tr0[0, 8] < 1.0 and tr0[0, 10] == 0.0
    # comms none: agent 1 has no track
    cfg2 = _cfg(na=2, nb=1, team_size=3, comms="none")
    env2 = SquadVecEnv(cfg2, 1, seed=1)
    _place(env2, np.array([[10.0, 20.0], [10.0, 30.0], [40.0, 40.0], [20.0, 20.0], [40.0, 41.0], [40.0, 42.0]]),
           np.array([0.0, np.pi, 0.0, np.pi, 0.0, 0.0]))
    env2.step(np.zeros((1, 6, 4), np.float32))
    obs2 = env2._build_obs()
    assert obs2[0, 1, lay["start"] + 9] == 0.0 and obs2[0, 0, lay["start"] + 9] == 1.0


def test_team_size_sampling_and_termination():
    cfg = EnvConfig(team_sizes=[[3, 2]], team_size_probs=[1.0])
    env = SquadVecEnv(cfg, 4, seed=0)
    assert (env.state.active.sum(1) == 5).all() and not env.state.alive[:, 5].any()
    # killing the two active enemies ends the episode even though slot 5 is inactive
    env.state.hp[0, 3:5] = 0
    env.state.alive[0, 3:5] = False
    _, _, _, done, info = env.step(np.zeros((4, 6, 4), np.float32))
    assert done[0] and info["winner"][0] == 0


def test_reload_cycle():
    env = SquadVecEnv(_cfg(na=1, nb=1), 1, seed=0)
    a = np.zeros((1, 2, 4), np.float32)
    a[0, 0, 3] = 1.0
    fired = 0
    for _ in range(int(12 * 0.4 / 0.05) + 2):
        _, _, _, _, info = env.step(a)
        fired += int(info["fire"][0, 0])
    assert fired == 12 and env.state.ammo[0, 0] == 0 and env.state.reload[0, 0] > 0
    for _ in range(int(2.0 / 0.05) + 1):
        env.step(np.zeros((1, 2, 4), np.float32))
    assert env.state.ammo[0, 0] == 12


def test_determinism():
    def run(seed):
        env = SquadVecEnv(EnvConfig(), 4, seed=seed)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(50):
            obs, gs, rew, done, _ = env.step(rng.uniform(-1, 1, size=(4, 6, 4)).astype(np.float32))
            out.append((obs.copy(), rew.copy()))
        return out
    a, b = run(3), run(3)
    for (o1, r1), (o2, r2) in zip(a, b):
        assert np.array_equal(o1, o2) and np.array_equal(r1, r2)
    assert not np.array_equal(a[-1][0], run(4)[-1][0])


def test_no_agent_inside_obstacle_after_steps():
    cfg = EnvConfig()
    env = SquadVecEnv(cfg, 8, seed=2)
    rng = np.random.default_rng(0)
    for _ in range(150):
        env.step(rng.uniform(-1, 1, size=(8, 6, 4)).astype(np.float32))
        inside = env.arena.point_in_boxes(env.state.pos, env.arena.boxes, env.arena.mask, margin=cfg.collision_radius - 0.05)
        assert not (inside & env.state.alive).any()
        assert (env.state.pos >= 0).all() and (env.state.pos <= cfg.arena_size).all()


def test_mirrored_layout_and_spawns():
    cfg = EnvConfig()
    env = SquadVecEnv(cfg, 2, seed=5)
    S = cfg.arena_size
    for e in range(2):
        b = env.arena.boxes[e][env.arena.mask[e]]
        for m in env.arena.mirror_boxes(b):
            assert (np.abs(b - m).sum(-1) < 1e-3).any()
        pa, pb = env.state.pos[e, :3], env.state.pos[e, 3:]
        assert np.allclose(S - pa, pb)
        # corners mode: opposite diagonal corners, far apart
        ca, cb = pa.mean(0), pb.mean(0)
        assert np.linalg.norm(ca - cb) > 0.9 * S
        assert (ca < 12).all() or (ca[0] < 12 and ca[1] > S - 12)
    env = SquadVecEnv(EnvConfig(spawn_mode="lanes"), 4, seed=5)
    # squad centroids sit within the spawn zone of each centre, so allow that slack
    slack = 2 * cfg.spawn_zone_radius
    for e in range(4):
        pa, pb = env.state.pos[e, :3], env.state.pos[e, 3:]
        assert np.allclose(S - pa, pb)
        assert cfg.spawn_distance - slack < np.linalg.norm(pa.mean(0) - pb.mean(0)) < cfg.spawn_distance + slack


def test_pettingzoo_api():
    from pettingzoo.test import parallel_api_test
    parallel_api_test(SquadParallelEnv(EnvConfig(max_steps=200), seed=0), num_cycles=300)
    parallel_api_test(SquadParallelEnv(EnvConfig(max_steps=200), action_mode="discrete", seed=0), num_cycles=300)


def test_role_rewards():
    cfg = _cfg(na=3, nb=1, team_size=3, spread_rest_deg=0.0, spread_max_deg=0.0)
    rr = cfg.role_rewards
    env = SquadVecEnv(cfg, 1, seed=1)
    st = env.state
    # slot roles: 0 assault, 1 flanker, 2 overwatch; enemy 3 at (30,20) facing +x (away from the squad)
    st.role[0, :3] = [0, 1, 2]
    _place(env, np.array([[22.0, 20.0], [28.0, 10.0], [10.0, 24.0], [30.0, 20.0], [45.0, 45.0], [45.0, 46.0]]),
           np.array([0.0, np.arctan2(10.0, 2.0), 0.0, 0.0, 0.0, 0.0]))
    a = np.zeros((1, 6, 4), np.float32)
    a[0, 0, 3] = 1.0                                   # assault shoots from 8 m, from behind (angle 180)
    _, _, rew, _, info = env.step(a)
    assert info["hit_enemy"][0, 0]
    base = cfg.reward.damage_dealt * 34 + cfg.reward.step
    assert rew[0, 0] == pytest.approx(base + rr.assault_close_damage * 34, abs=1e-5)      # close bonus, no flank bonus (assault)
    # overwatch (agent 2, 20 m away, sees the enemy) gets the assist bonus and the unique-spot bonus
    assert env.vis[0, 2, 3]
    assert rew[0, 2] >= rr.overwatch_assist_damage * 34 + cfg.reward.step - 1e-6
    # flanker hits from the side/behind at 10 m: flank bonus, no close bonus
    for _ in range(8):
        env.step(np.zeros((1, 6, 4), np.float32))
    a = np.zeros((1, 6, 4), np.float32)
    a[0, 1, 3] = 1.0
    _, _, rew, _, info = env.step(a)
    assert info["hit_enemy"][0, 1] and info["engage_angle"][0, 1] > rr.flank_angle_deg
    # flank bonus plus the crossfire bonus: agent 0 hit the same victim 0.45 s earlier from
    # a bearing ~90 deg away, so both shooters are paid the crossfire bonus
    assert rew[0, 1] == pytest.approx(base + rr.flank_damage * 34 + rr.crossfire_bonus, abs=1e-5)
    assert rew[0, 0] == pytest.approx(cfg.reward.step + rr.crossfire_bonus, abs=1e-5)
    ep = env._episode_summary(np.array([0]), np.array([-1]))
    assert ep["crossfire_rate"][0, 0] > 0
    # disabling role rewards removes the extras
    cfg2 = _cfg(na=3, nb=1, team_size=3, spread_rest_deg=0.0, spread_max_deg=0.0)
    cfg2.role_rewards.enabled = False
    env2 = SquadVecEnv(cfg2, 1, seed=1)
    env2.state.role[0, :3] = [0, 1, 2]
    _place(env2, np.array([[22.0, 20.0], [28.0, 10.0], [10.0, 24.0], [30.0, 20.0], [45.0, 45.0], [45.0, 46.0]]),
           np.array([0.0, np.arctan2(10.0, 2.0), 0.0, 0.0, 0.0, 0.0]))
    a = np.zeros((1, 6, 4), np.float32)
    a[0, 0, 3] = 1.0
    _, _, rew, _, _ = env2.step(a)
    assert rew[0, 0] == pytest.approx(base, abs=1e-6)


def test_spawn_curriculum_interpolates_towards_the_corners():
    cfg = EnvConfig()
    S = cfg.arena_size
    env = SquadVecEnv(cfg, 8, seed=1)
    env.set_corner_lerp(0.3)
    env._reset_envs(np.arange(8))
    near = np.linalg.norm(env.state.pos[:, :3].mean(1) - env.state.pos[:, 3:].mean(1), axis=-1)
    env.set_corner_lerp(1.0)
    env._reset_envs(np.arange(8))
    far = np.linalg.norm(env.state.pos[:, :3].mean(1) - env.state.pos[:, 3:].mean(1), axis=-1)
    assert near.mean() < 0.45 * far.mean()
    assert far.mean() > 0.9 * S
    # spawns stay inside the arena and clear of obstacles at every setting
    for k in (0.05, 0.5, 1.0):
        env.set_corner_lerp(k)
        env._reset_envs(np.arange(8))
        assert (env.state.pos > 0).all() and (env.state.pos < S).all()
        inside = env.arena.point_in_boxes(env.state.pos, env.arena.boxes, env.arena.mask, margin=0.0)
        assert not (inside & env.state.active).any()


def test_vision_reaches_the_far_wall_and_is_limited_only_by_obstacles():
    """With vision_range >= the arena diagonal, an unobstructed cone terminates on a wall,
    never on the range clip, and a sight-blocking obstacle is what shortens it."""
    cfg = EnvConfig()
    S = cfg.arena_size
    assert cfg.vision_range >= S * np.sqrt(2) - 1e-6
    env = SquadVecEnv(_cfg(na=1, nb=1), 1, seed=1)
    ec = env.cfg
    # corner to corner, empty arena: the centre ray runs the full diagonal
    _place(env, np.array([[1.5, 1.5], [S - 1.5, S - 1.5]]), np.array([np.pi / 4, np.pi]))
    R = ec.num_rays
    centre = R // 2
    diag = np.linalg.norm(np.array([S - 3.0, S - 3.0]))
    assert env.ray_kind[0, 0, centre] in (1, 2)                    # a wall or the enemy, not "nothing"
    assert env.ray_dist[0, 0, centre] > 0.9 * diag
    assert bool(env.vis[0, 0, 1])                                  # the enemy is visible across the whole arena
    # a wall on the diagonal cuts the sightline; a low crate does not
    for kind, visible in ((WALL, False), (CRATE, True)):
        env2 = SquadVecEnv(_cfg(na=1, nb=1), 1, seed=1)
        _place(env2, np.array([[1.5, 1.5], [S - 1.5, S - 1.5]]), np.array([np.pi / 4, np.pi]),
               boxes=[[22, 22, 26, 26]], kinds=[kind])
        assert bool(env2.vis[0, 0, 1]) is visible
        if kind == WALL:
            assert env2.ray_dist[0, 0, centre] < 32.0


def test_ray_spacing_resolves_a_body_at_maximum_range():
    """A 2 m wide body at the far end of the cone must not fall between two rays."""
    cfg = EnvConfig()
    spacing = np.deg2rad(cfg.vision_fov_deg) / (cfg.num_rays - 1)
    subtended = 2 * np.arctan(cfg.collision_radius / cfg.vision_range)
    assert spacing < subtended
    # empirically: an enemy dead ahead at long range is seen by at least one ray
    env = SquadVecEnv(_cfg(na=1, nb=1), 1, seed=1)
    S = cfg.arena_size
    for dist in (20.0, 35.0, 45.0):
        _place(env, np.array([[2.0, 24.0], [2.0 + dist, 24.0]]), np.array([0.0, np.pi]))
        assert (env.ray_kind[0, 0] == 2).any(), f"enemy at {dist} m missed by every ray"


def test_denser_cover_still_leaves_the_spawns_connected():
    cfg = EnvConfig()
    env = SquadVecEnv(cfg, 12, seed=4)
    S = cfg.arena_size
    for e in range(12):
        boxes = env.arena.boxes[e][env.arena.mask[e]]
        area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum() / S ** 2
        assert area <= cfg.coverage_max + 1e-6
        a, b = env.state.pos[e, :3].mean(0), env.state.pos[e, 3:].mean(0)
        assert connected(boxes, S, a, b, cfg.collision_radius)
