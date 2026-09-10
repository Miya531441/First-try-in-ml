"""Unit tests: raycast intersection, cone membership, occlusion, damage resolution,
friendly fire, determinism and the PettingZoo API."""
import numpy as np
import pytest

from env.agent import EnvConfig
from env.raycast import (cast_rays, cone_membership, ray_box_t, ray_boundary_t, ray_circle_t,
                         segment_blocked)
from env.squad_env import SquadParallelEnv, SquadVecEnv


def _boxes(*bs):
    b = np.array(bs, np.float32).reshape(1, -1, 4)
    return b, np.ones(b.shape[:2], bool)


def test_ray_box_intersection():
    boxes, mask = _boxes([5, -1, 7, 1])
    o = np.array([[[0.0, 0.0]]], np.float32)
    d = np.array([[[1.0, 0.0]]], np.float32)
    t = ray_box_t(o, d, boxes, mask)
    assert np.isclose(t[0, 0, 0], 5.0)
    # pointing away -> no hit
    t = ray_box_t(o, -d, boxes, mask)
    assert t[0, 0, 0] > 1e8
    # ray parallel to the box, missing it
    d2 = np.array([[[0.0, 1.0]]], np.float32)
    assert ray_box_t(o, d2, boxes, mask)[0, 0, 0] > 1e8
    # diagonal hit on the corner region
    d3 = np.array([[[np.cos(0.1), np.sin(0.1)]]], np.float32)
    t3 = ray_box_t(o, d3, boxes, mask)[0, 0, 0]
    assert np.isclose(t3, 5.0 / np.cos(0.1))


def test_ray_boundary_and_circle():
    o = np.array([[[10.0, 10.0]]], np.float32)
    d = np.array([[[0.0, -1.0]]], np.float32)
    assert np.isclose(ray_boundary_t(o, d, 64.0)[0, 0], 10.0)
    centers = np.array([[[10.0, 4.0]]], np.float32)
    t = ray_circle_t(o, d, centers, 0.4, np.ones((1, 1, 1), bool))
    assert np.isclose(t[0, 0, 0], 6.0 - 0.4)
    # circle behind the ray
    t = ray_circle_t(o, -d, centers, 0.4, np.ones((1, 1, 1), bool))
    assert t[0, 0, 0] > 1e8


def test_cone_membership():
    o = np.zeros((1, 1, 2), np.float32)
    theta = np.zeros((1, 1), np.float32)
    pts = np.array([[[10.0, 0.0], [10.0, np.tan(np.deg2rad(29)) * 10], [10.0, np.tan(np.deg2rad(31)) * 10],
                     [-10.0, 0.0], [35.0, 0.0]]], np.float32)
    inside, _, _ = cone_membership(o, theta, pts, np.deg2rad(60), 30.0)
    assert inside[0, 0].tolist() == [True, True, False, False, False]


def test_occlusion_by_wall_and_body():
    boxes, mask = _boxes([5, -1, 7, 1])
    o = np.array([[[0.0, 0.0]]], np.float32)
    d = np.array([[[1.0, 0.0]]], np.float32)
    centers = np.array([[[0.0, 0.0], [10.0, 0.0]]], np.float32)   # self + agent behind the wall
    cmask = np.array([[[False, True]]])
    dist, kind, hit = cast_rays(o, d, boxes, mask, centers, 0.4, cmask, 64.0, 30.0)
    assert kind[0, 0] == 1 and np.isclose(dist[0, 0], 5.0) and hit[0, 0] == -1
    # agent in front of the wall is hit first
    centers2 = np.array([[[0.0, 0.0], [3.0, 0.0]]], np.float32)
    dist, kind, hit = cast_rays(o, d, boxes, mask, centers2, 0.4, cmask, 64.0, 30.0)
    assert kind[0, 0] == 2 and hit[0, 0] == 1 and np.isclose(dist[0, 0], 2.6)
    # body occludes another body
    centers3 = np.array([[[0.0, 0.0], [3.0, 0.0], [4.0, 0.0]]], np.float32)
    cmask3 = np.array([[[False, True, True]]])
    _, _, hit = cast_rays(o, d, np.zeros((1, 0, 4), np.float32), np.zeros((1, 0), bool), centers3, 0.4, cmask3, 64.0, 30.0)
    assert hit[0, 0] == 1
    # LOS segment blocked by the wall, clear otherwise
    p = np.array([[[0.0, 0.0]]], np.float32)
    q = np.array([[[10.0, 0.0]]], np.float32)
    assert segment_blocked(p, q, boxes, mask, centers, 0.4, np.zeros((1, 1, 2), bool))[0, 0]
    q2 = np.array([[[0.0, 10.0]]], np.float32)
    assert not segment_blocked(p, q2, boxes, mask, centers, 0.4, np.zeros((1, 1, 2), bool))[0, 0]


def _place(env, pos, theta):
    st = env.state
    st.pos[0] = pos
    st.theta[0] = theta
    st.vel[0] = 0
    env.arena.mask[0] = False
    env._perceive(np.array([0]))


def test_damage_resolution_and_kill():
    cfg = EnvConfig(team_size=1, spread_rest_deg=0.0, spread_max_deg=0.0)
    env = SquadVecEnv(cfg, 1, seed=1)
    _place(env, np.array([[10.0, 10.0], [20.0, 10.0]]), np.array([0.0, np.pi]))
    a = np.zeros((1, 2, 4), np.float32)
    a[0, 0, 3] = 1.0                      # agent 0 fires at agent 1
    _, _, rew, done, info = env.step(a)
    assert env.state.hp[0, 1] == pytest.approx(100 - 34)
    assert env.state.hp[0, 0] == 100
    assert info["hit_enemy"][0, 0] and not info["hit_ally"][0, 0]
    assert rew[0, 0] == pytest.approx(0.01 * 34 + cfg.reward.step + cfg.reward.shaping_los)
    # cooldown suppresses the next shot
    _, _, rew, done, info = env.step(a)
    assert not info["fire"][0, 0]
    # wait out the cooldown then land two more hits -> kill, terminal reward
    for _ in range(7):
        env.step(np.zeros((1, 2, 4), np.float32))
    _, _, rew, done, info = env.step(a)
    assert env.state.hp[0, 1] == pytest.approx(100 - 68)
    for _ in range(8):
        env.step(np.zeros((1, 2, 4), np.float32))
    _, _, rew, done, info = env.step(a)
    assert done[0] and info["winner"][0] == 0
    # no shaping on the final step: the only enemy is dead, so nothing is in the cone
    assert rew[0, 0] == pytest.approx(0.34 + 0.5 + 1.0 + cfg.reward.step, abs=1e-6)
    # damage taken is clipped to the remaining 32 hp
    assert rew[0, 1] == pytest.approx(-0.005 * 32 - 0.5 - 1.0, abs=1e-6)


def test_friendly_fire_penalty_and_toggle():
    for ff in (True, False):
        cfg = EnvConfig(team_size=2, spread_rest_deg=0.0, spread_max_deg=0.0, friendly_fire=ff)
        env = SquadVecEnv(cfg, 1, seed=1)
        _place(env, np.array([[10.0, 10.0], [15.0, 10.0], [40.0, 40.0], [45.0, 45.0]]),
               np.array([0.0, 0.0, np.pi, np.pi]))
        a = np.zeros((1, 4, 4), np.float32)
        a[0, 0, 3] = 1.0
        _, _, rew, _, info = env.step(a)
        assert info["hit_ally"][0, 0]
        if ff:
            assert env.state.hp[0, 1] == pytest.approx(66)
            assert rew[0, 0] < -30
        else:
            assert env.state.hp[0, 1] == 100
            assert rew[0, 0] == pytest.approx(cfg.reward.step)


def test_reload_cycle():
    cfg = EnvConfig(team_size=1)
    env = SquadVecEnv(cfg, 1, seed=0)
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


def test_observation_vision_channel():
    cfg = EnvConfig(team_size=1, include_prev_action=False)
    env = SquadVecEnv(cfg, 1, seed=1)
    _place(env, np.array([[10.0, 10.0], [20.0, 10.0]]), np.array([0.0, np.pi]))
    obs = env._build_obs()
    vis = obs[0, 0, : cfg.num_rays * 5].reshape(cfg.num_rays, 5)
    centre = cfg.num_rays // 2
    # the middle rays hit the enemy (one-hot index 4) at ~9.6 m
    assert vis[centre - 1:centre + 1, 4].max() == 1.0
    ray = vis[centre - 1:centre + 1][vis[centre - 1:centre + 1, 4] == 1.0][0]
    assert np.isclose(ray[0], 9.6 / 30.0, atol=0.01)
    assert env.obs_dim == obs.shape[-1]


def test_determinism():
    def run(seed):
        cfg = EnvConfig()
        env = SquadVecEnv(cfg, 4, seed=seed)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(50):
            a = rng.uniform(-1, 1, size=(4, 6, 4)).astype(np.float32)
            obs, gs, rew, done, _ = env.step(a)
            out.append((obs.copy(), rew.copy()))
        return out
    a, b = run(3), run(3)
    for (o1, r1), (o2, r2) in zip(a, b):
        assert np.array_equal(o1, o2) and np.array_equal(r1, r2)
    c = run(4)
    assert not np.array_equal(a[-1][0], c[-1][0])


def test_no_agent_inside_obstacle_after_steps():
    cfg = EnvConfig()
    env = SquadVecEnv(cfg, 8, seed=2)
    rng = np.random.default_rng(0)
    for _ in range(200):
        env.step(rng.uniform(-1, 1, size=(8, 6, 4)).astype(np.float32))
        pos = env.state.pos
        inside = env.arena.point_in_boxes(pos, env.arena.boxes, env.arena.mask, margin=cfg.collision_radius - 0.05)
        assert not (inside & env.state.alive).any()
        assert (pos >= 0).all() and (pos <= cfg.arena_size).all()


def test_mirrored_layout_and_spawns():
    cfg = EnvConfig()
    env = SquadVecEnv(cfg, 2, seed=5)
    S = cfg.arena_size
    for e in range(2):
        b = env.arena.boxes[e][env.arena.mask[e]]
        mirrored = env.arena.mirror_boxes(b)
        for m in mirrored:
            assert (np.abs(b - m).sum(-1) < 1e-3).any()
        pa, pb = env.state.pos[e, :3], env.state.pos[e, 3:]
        assert np.allclose(S - pa, pb)
        assert 38 < np.linalg.norm(pa.mean(0) - pb.mean(0)) < 44


def test_pettingzoo_api():
    from pettingzoo.test import parallel_api_test
    env = SquadParallelEnv(EnvConfig(max_steps=200), seed=0)
    parallel_api_test(env, num_cycles=300)
    env_d = SquadParallelEnv(EnvConfig(max_steps=200), action_mode="discrete", seed=0)
    parallel_api_test(env_d, num_cycles=300)
