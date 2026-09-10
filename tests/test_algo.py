"""Algorithm-level tests: GAE, chunked sequence evaluation matches the acting pass,
buffer/minibatch layout, league Elo bookkeeping, discriminator bonus."""
import numpy as np
import torch

from algo.buffer import RolloutBuffer
from algo.discriminator import RoleDiscriminator
from algo.league import LATEST, League
from algo.augment import mirror_action, mirror_aux, mirror_obs
from algo.networks import ActorCritic
from env.agent import EnvConfig
from env.squad_env import SquadVecEnv


def _env():
    return SquadVecEnv(EnvConfig(), 2, seed=0)


def _ac(env, **kw):
    return ActorCritic(env.obs_layout, env.obs_dim, env.gs_dim, env.T, **kw)


def test_gae_matches_reference():
    T, E, N = 8, 2, 2
    buf = RolloutBuffer(T, E, N, 3, 2, 4, chunk_len=4, gamma=0.9, lam=0.8)
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(T, E, N)).astype(np.float32)
    values = rng.normal(size=(T, E, N)).astype(np.float32)
    dones = np.zeros((T, E), bool)
    dones[3, 0] = True
    for t in range(T):
        buf.add(torch.zeros(E, N, 3), torch.zeros(E, N, 2), torch.zeros(E, N, 4), torch.zeros(E, N),
                torch.as_tensor(values[t]), torch.zeros(E), torch.ones(E, N), torch.ones(E, N), torch.zeros(E, N, 4))
        buf.add_outcome(torch.as_tensor(rewards[t]), dones[t])
    last = rng.normal(size=(E, N)).astype(np.float32)
    buf.compute_gae(torch.as_tensor(last))
    # reference
    adv = np.zeros((T, E, N))
    for e in range(E):
        for n in range(N):
            g = 0.0
            for t in reversed(range(T)):
                nv = last[e, n] if t == T - 1 else values[t + 1, e, n]
                nt = 0.0 if dones[t, e] else 1.0
                delta = rewards[t, e, n] + 0.9 * nv * nt - values[t, e, n]
                g = delta + 0.9 * 0.8 * nt * g
                adv[t, e, n] = g
    assert np.allclose(buf.advantages.numpy(), adv, atol=1e-5)


def test_sequence_evaluate_matches_act_with_resets():
    torch.manual_seed(0)
    env = _env()
    obs_dim, gs_dim, T = env.obs_dim, env.gs_dim, env.T
    ac = _ac(env, hidden=32, d_model=32)
    L, B = 6, 4
    obs = torch.randn(L, B, obs_dim)
    gs = torch.randn(L, B, gs_dim)
    first = torch.zeros(L, B)
    first[0] = 1
    first[3, 1] = 1                     # episode reset mid-sequence for one stream
    slot = torch.arange(B) % T
    h = ac.initial_hidden(B)
    acts, logps, vals = [], [], []
    for t in range(L):
        a, lp, v, h, _ = ac.act(obs[t], gs[t], h, slot, first[t])
        acts.append(a)
        logps.append(lp)
        vals.append(v)
    acts, logps, vals = torch.stack(acts), torch.stack(logps), torch.stack(vals)
    lp2, ent, v2, hs, aux = ac.evaluate(obs, obs, gs, ac.initial_hidden(B), first, acts, slot)
    assert torch.allclose(lp2, logps, atol=1e-5)
    assert torch.allclose(v2, vals, atol=1e-5)
    assert torch.allclose(hs[3, 1], torch.zeros(32))          # reset applied before step 3
    assert aux.shape == (L, B, T, 2)


def test_minibatch_layout_keeps_teams_together():
    T, E, N = 8, 3, 6
    buf = RolloutBuffer(T, E, N, 2, 2, 4, chunk_len=4, gamma=0.9, lam=0.8, plan_tokens=4)
    for t in range(T):
        obs = torch.zeros(E, N, 2)
        obs[..., 0] = torch.arange(N)[None]
        obs[..., 1] = t
        buf.add(obs, torch.zeros(E, N, 2), torch.zeros(E, N, 4), torch.zeros(E, N), torch.zeros(E, N),
                torch.zeros(E), torch.ones(E, N), torch.ones(E, N), torch.zeros(E, N, 4))
        buf.add_outcome(torch.zeros(E, N), np.zeros(E, bool))
    buf.compute_gae(torch.zeros(E, N))
    for b in buf.iterate(2, np.random.default_rng(0)):
        P = b["num_pairs"]
        o = b["obs"].view(4, P, N, 2)
        assert torch.equal(o[0, :, :, 0], torch.arange(N).float()[None].repeat(P, 1))
        assert torch.equal(b["slot"], (torch.arange(N) % 3).repeat(P))
        assert b["plan_action"].shape == (4, P, 2)
        assert (o[1:, :, :, 1] - o[:-1, :, :, 1] == 1).all()      # consecutive timesteps


def test_league_sampling_and_elo(tmp_path):
    lg = League(str(tmp_path), p_latest=0.7, scripted=["charger"])
    rng = np.random.default_rng(0)
    names = [lg.sample_opponent(rng) for _ in range(2000)]
    frac_latest = np.mean([n == LATEST for n in names])
    assert 0.62 < frac_latest < 0.78
    ac = _ac(_env(), hidden=8, d_model=8, n_heads=2)
    snap = lg.add_snapshot(ac, 1)
    for _ in range(20):
        lg.record(snap, 1.0)
    assert lg.elo[LATEST] > 1000 > lg.elo[snap]
    assert lg.members[snap]["winrate"] < 0.5
    lg.save()
    lg2 = League(str(tmp_path))
    lg2.load()
    assert lg2.elo[LATEST] == lg.elo[LATEST]


def test_discriminator_bonus_is_zero_centred_then_learns():
    d = RoleDiscriminator(3, beta=0.05, stat_dim=4)
    stats = np.random.default_rng(0).normal(size=(4, 4)).astype(np.float32)
    roles = np.array([0, 1, 2, 0])
    b0 = d.bonus(stats, roles)
    assert np.all(np.abs(b0) < 0.05)              # near-uniform q at init
    rng = np.random.default_rng(1)
    # role k has a distinctive first statistic
    for _ in range(600):
        r = rng.integers(0, 3)
        d.add(np.array([[r * 1.0, 0, 0, 0]]) + rng.normal(scale=0.05, size=(1, 4)), np.array([r]))
    out = d.train_steps(steps=300, rng=rng)
    assert out["disc/accuracy"] > 0.8
    assert d.bonus(np.array([[2.0, 0, 0, 0]], np.float32), np.array([2]))[0] > 0.02


def test_mirror_augmentation_is_an_involution_and_consistent():
    env = _env()
    rng = np.random.default_rng(0)
    for _ in range(5):
        env.step(rng.uniform(-1, 1, size=(2, 6, 4)).astype(np.float32))
    obs = torch.as_tensor(env._build_obs()).view(-1, env.obs_dim)
    m = mirror_obs(obs, env.obs_layout, env.obs_dim)
    assert torch.allclose(mirror_obs(m, env.obs_layout, env.obs_dim), obs)
    assert not torch.allclose(m, obs)
    lay = env.obs_layout["vision"]
    R, C = lay["shape"]
    v = obs[:, lay["start"]: lay["start"] + lay["size"]].reshape(-1, R, C)
    mv = m[:, lay["start"]: lay["start"] + lay["size"]].reshape(-1, R, C)
    assert torch.equal(mv, v.flip(1))
    a = torch.tensor([[0.3, 0.5, -0.7, 1.0]])
    assert torch.equal(mirror_action(mirror_action(a, "hybrid"), "hybrid"), a)
    d = torch.tensor([[2.0, 0.0, 6.0, 1.0]])
    assert torch.equal(mirror_action(d, "discrete"), torch.tensor([[2.0, 2.0, 0.0, 1.0]]))
    aux = torch.as_tensor(env.aux_targets())
    assert torch.allclose(mirror_aux(mirror_aux(aux)), aux)
    # a mirrored world gives the mirrored policy input: stacked frames are mirrored frame-wise
    stacked = torch.cat([obs, obs], -1)
    ms = mirror_obs(stacked, env.obs_layout, env.obs_dim)
    assert torch.allclose(ms[:, : env.obs_dim], m) and torch.allclose(ms[:, env.obs_dim:], m)
