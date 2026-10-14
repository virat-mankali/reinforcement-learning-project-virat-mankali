"""Regression tests for the revised inventory model's physical and monetary ledger.

Run from this folder: .venv/bin/python -m pytest -q test_inventory.py
The tests intentionally require invalid model inputs to fail explicitly.
"""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from inventory_experiment import InventoryEnv, evaluate, heuristic, scenarios


def run_episode(env, actions, *, demands=None, seed=101):
    options = None if demands is None else {"demands": demands}
    first, _ = env.reset(seed=seed, options=options)
    trajectory = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append((obs.copy(), reward, terminated, truncated, info.copy()))
    return first, trajectory


def test_gymnasium_environment_checker():
    check_env(InventoryEnv(), skip_render_check=True)


def test_seeded_reset_reproduces_observations_rewards_and_demands():
    env = InventoryEnv(horizon=30)
    actions = [0, 2, 4, 0, 12] * 6
    first, expected = run_episode(env, actions, seed=2718)
    run_episode(env, actions, seed=17)
    restarted, actual = run_episode(env, actions, seed=2718)
    np.testing.assert_array_equal(first, restarted)
    for lhs, rhs in zip(expected, actual):
        np.testing.assert_array_equal(lhs[0], rhs[0])
        assert lhs[1:] == rhs[1:]
    _, different = run_episode(env, actions, seed=2719)
    assert [x[4]["demand"] for x in expected] != [x[4]["demand"] for x in different]


@pytest.mark.parametrize("lead_time", [1, 2, 3, 5])
def test_observation_bounds_dtype_pipeline_and_time(lead_time):
    env = InventoryEnv(lead_time=lead_time, horizon=40, capacity=123)
    obs, info = env.reset(seed=99)
    assert isinstance(info, dict)
    assert obs.dtype == np.float32
    assert obs.shape == (lead_time + 2,)
    assert env.observation_space.contains(obs)
    rng = np.random.default_rng(818)
    for day in range(env.horizon):
        obs, reward, terminated, truncated, info = env.step(int(rng.integers(env.action_space.n)))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert not truncated
        assert terminated == (day == env.horizon - 1)
        assert obs[0] == pytest.approx(env.on_hand / env.capacity)
        np.testing.assert_allclose(obs[1:-1], np.asarray(env.pipeline) / env.capacity)
        assert obs[-1] == pytest.approx((env.horizon - day - 1) / env.horizon)
    assert obs[-1] == 0


@pytest.mark.parametrize("lead_time", [1, 2, 3, 5])
def test_order_arrives_exactly_lead_time_days_after_decision(lead_time):
    horizon = lead_time + 2
    env = InventoryEnv(horizon=horizon, lead_time=lead_time, initial_inventory=0, holding_cost=0)
    _, trajectory = run_episode(env, [3] + [0] * (horizon - 1), demands=[0] * horizon)
    expected_arrivals = [0] * horizon
    expected_arrivals[lead_time] = 15
    assert [t[4]["arrivals"] for t in trajectory] == expected_arrivals
    assert env.on_hand == 15
    assert sum(env.pipeline) == 0
    assert sum(t[4]["business_profit"] for t in trajectory) == 0


def test_arrivals_are_available_before_same_day_demand():
    env = InventoryEnv(horizon=3, lead_time=2, initial_inventory=0)
    _, trajectory = run_episode(env, [2, 0, 0], demands=[0, 0, 12])
    last = trajectory[-1][4]
    assert last["arrivals"] == 10
    assert last["sold"] == 10
    assert last["lost_units"] == 2
    assert last["on_hand"] == 0


def test_capacity_reserves_all_pipeline_and_never_discards_arrivals():
    env = InventoryEnv(horizon=5, lead_time=2, initial_inventory=100, capacity=125, max_order=50)
    _, trajectory = run_episode(env, [10] * 5, demands=[0] * 5)
    infos = [t[4] for t in trajectory]
    assert [i["accepted_order"] for i in infos] == [25, 0, 0, 0, 0]
    assert [i["arrivals"] for i in infos] == [0, 0, 25, 0, 0]
    assert [i["rejected_order"] for i in infos] == [25, 50, 50, 50, 50]
    assert env.on_hand == 125
    assert sum(env.pipeline) == 0


def test_capacity_is_checked_before_arrivals_and_demand():
    env = InventoryEnv(horizon=2, initial_inventory=120, capacity=125)
    _, trajectory = run_episode(env, [4, 4], demands=[100, 0])
    assert trajectory[0][4]["accepted_order"] == 5
    assert trajectory[1][4]["accepted_order"] == 20


@pytest.mark.parametrize("seed", range(12))
def test_mass_balance_fifo_and_capacity_across_random_trajectories(seed):
    horizon, lead_time, initial = 50, 3, 40
    env = InventoryEnv(horizon=horizon, lead_time=lead_time, initial_inventory=initial, capacity=123)
    rng = np.random.default_rng(seed)
    demands = rng.poisson(35, size=horizon)
    obs, _ = env.reset(seed=seed, options={"demands": demands})
    total_accepted = total_sold = 0
    history = []
    for day in range(horizon):
        old_position = env.on_hand + sum(env.pipeline)
        action = int(rng.integers(env.action_space.n))
        obs, _, terminated, _, info = env.step(action)
        history.append(info)
        total_accepted += info["accepted_order"]
        total_sold += info["sold"]
        assert info["accepted_order"] == min(info["requested_order"], env.capacity - old_position)
        assert info["requested_order"] == info["accepted_order"] + info["rejected_order"]
        assert info["arrivals"] == (history[day - lead_time]["accepted_order"] if day >= lead_time else 0)
        assert initial + total_accepted == total_sold + env.on_hand + sum(env.pipeline)
        assert 0 <= env.on_hand + sum(env.pipeline) <= env.capacity
        assert info["demand"] == info["sold"] + info["lost_units"]
        assert info["stockout_day"] == int(info["lost_units"] > 0)
        assert info["in_transit"] == sum(env.pipeline)
        assert env.observation_space.contains(obs)
        assert terminated == (day == horizon - 1)


@pytest.mark.parametrize("action", [-1, 25, 999, 1.25, "2", None, [], np.array([1])])
def test_invalid_actions_rejected_without_mutating_state(action):
    env = InventoryEnv()
    obs, _ = env.reset(seed=42)
    old_state = (env.day, env.on_hand, tuple(env.pipeline))
    with pytest.raises(ValueError, match="[Ii]nvalid.*action"):
        env.step(action)
    assert (env.day, env.on_hand, tuple(env.pipeline)) == old_state
    np.testing.assert_array_equal(env._obs(), obs)


def test_cannot_step_past_terminal_state_and_reset_clears_pipeline():
    env = InventoryEnv(horizon=1)
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(4)
    assert terminated and not truncated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    obs, _ = env.reset(seed=1)
    assert env.day == 0
    assert env.on_hand == env.initial_inventory
    assert env.pipeline == [0] * env.lead_time
    assert obs[-1] == 1


def test_opening_holding_and_terminal_accounting_hand_calculation():
    env = InventoryEnv(horizon=3, initial_inventory=10, lead_time=2)
    _, trajectory = run_episode(env, [1, 0, 0], demands=[4, 12, 0])
    infos = [t[4] for t in trajectory]
    assert [i["opening_inventory_cost"] for i in infos] == [600, 0, 0]
    assert [i["purchase_cost"] for i in infos] == [300, 0, 0]
    assert [i["holding_cost"] for i in infos] == [12, 0, 10]
    assert [i["stockout_cost"] for i in infos] == [0, 300, 0]
    assert [i["terminal_salvage"] for i in infos] == [0, 0, 300]
    assert [i["business_profit"] for i in infos] == [-512, 300, 290]
    assert sum(i["business_profit"] for i in infos) == 78
    assert sum(t[1] for t in trajectory) == pytest.approx(0.078)


def test_terminal_credit_includes_all_undelivered_pipeline():
    env = InventoryEnv(horizon=2, lead_time=3, initial_inventory=10)
    _, trajectory = run_episode(env, [2, 3], demands=[4, 3])
    infos = [t[4] for t in trajectory]
    assert [i["arrivals"] for i in infos] == [0, 0]
    assert env.on_hand == 3
    assert sum(env.pipeline) == 25
    assert infos[-1]["terminal_salvage"] == 60 * (3 + 25)
    assert sum(i["business_profit"] for i in infos) == 262


@pytest.mark.parametrize("seed", range(8))
def test_episode_profit_reconciles_to_sold_margin_less_operating_costs(seed):
    env = InventoryEnv(horizon=30, lead_time=3, initial_inventory=40)
    rng = np.random.default_rng(seed)
    actions = rng.integers(env.action_space.n, size=env.horizon).tolist()
    _, trajectory = run_episode(env, actions, seed=seed)
    infos = [t[4] for t in trajectory]
    total_profit = sum(i["business_profit"] for i in infos)
    expected = sum((env.price - env.unit_cost) * i["sold"] - i["holding_cost"] - i["stockout_cost"] for i in infos)
    assert total_profit == pytest.approx(expected)
    for info in infos:
        assert info["business_profit"] == pytest.approx(info["revenue"] - info["purchase_cost"] - info["opening_inventory_cost"] - info["holding_cost"] - info["stockout_cost"] + info["terminal_salvage"])


@pytest.mark.parametrize("multiplier", [0.5, 1.0, 2.0, 10.0])
def test_reward_shaping_preserves_original_business_profit_and_transitions(multiplier):
    demands = [5, 80, 10, 100, 0, 55]
    actions = [2, 5, 0, 8, 4, 1]
    base = InventoryEnv(horizon=6, penalty_multiplier=1)
    shaped = InventoryEnv(horizon=6, penalty_multiplier=multiplier)
    _, expected = run_episode(base, actions, demands=demands)
    _, actual = run_episode(shaped, actions, demands=demands)
    for lhs, rhs in zip(expected, actual):
        np.testing.assert_array_equal(lhs[0], rhs[0])
        assert lhs[2:4] == rhs[2:4]
        li, ri = lhs[4], rhs[4]
        for key in li:
            if key != "shaped_profit":
                assert li[key] == ri[key]
        assert ri["shaped_profit"] == pytest.approx(ri["business_profit"] - (multiplier - 1) * ri["stockout_cost"])
        assert rhs[1] * 1000 == pytest.approx(ri["shaped_profit"])
    assert sum(t[4]["business_profit"] for t in expected) == sum(t[4]["business_profit"] for t in actual)


def test_scenario_banks_are_repeatable_and_seeds_change_realizations():
    a = scenarios(303, n=8)
    b = scenarios(303, n=8)
    c = scenarios(304, n=8)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (8, 30)
    assert np.issubdtype(a.dtype, np.integer)
    assert (a >= 0).all()
    assert not np.array_equal(a, c)
    assert len({tuple(row) for row in a}) == len(a)


def test_evaluation_uses_identical_demand_bank_for_every_policy():
    bank = scenarios(5101, n=6)
    original = bank.copy()
    zero_rows, zero_days = evaluate(lambda obs, env: 0, bank, "zero")
    stock_rows, stock_days = evaluate(heuristic(60, 95), bank, "heuristic")
    np.testing.assert_array_equal(bank, original)
    np.testing.assert_array_equal(zero_rows["demand"], bank.sum(axis=1))
    np.testing.assert_array_equal(stock_rows["demand"], bank.sum(axis=1))
    np.testing.assert_array_equal(zero_days["demand"], bank[0])
    np.testing.assert_array_equal(stock_days["demand"], bank[0])
    for rows in (zero_rows, stock_rows):
        np.testing.assert_allclose(rows["average_daily_profit"], rows["business_profit"] / 30)
        np.testing.assert_allclose(rows["fill_rate"], rows["sold"] / rows["demand"])
        np.testing.assert_allclose(rows["stockout_day_rate"], rows["stockout_day"] / 30)
        np.testing.assert_allclose(rows["business_profit"], rows["revenue"] - rows["purchase_cost"] - rows["opening_inventory_cost"] - rows["holding_cost"] - rows["stockout_cost"] + rows["terminal_salvage"])


def test_zero_demand_fill_rate_is_one():
    rows, _ = evaluate(lambda obs, env: 0, np.zeros((2, 30), dtype=int), "zero")
    assert (rows["fill_rate"] == 1).all()
    assert (rows["stockout_day_rate"] == 0).all()


def test_three_day_stress_has_correct_calendar_window_and_intensity():
    # Fixed seed and generous tolerances make this a stable distribution regression.
    # Normal and stress are separate banks; matching is across policies WITHIN
    # each condition, not a claim of identical non-shock demand across conditions.
    normal = scenarios(6101, n=4000, stress=False)
    stress = scenarios(6101, n=4000, stress=True)
    np.testing.assert_array_equal(stress, scenarios(6101, n=4000, stress=True))
    expected = np.full(30, 20.0)
    expected[14:17] = 60.0  # Human-readable days 15, 16 and 17 only.
    np.testing.assert_allclose(normal.mean(axis=0), 20.0, atol=0.6, rtol=0)
    np.testing.assert_allclose(stress.mean(axis=0), expected, atol=0.8, rtol=0)


@pytest.mark.parametrize("demands", [
    [0, 1], [0, 1, 2, 3], [0, -1, 2], [0, -0.25, 2],
    [0, 1.5, 2], [0, float("nan"), 2], [0, float("inf"), 2],
    [[0], [1], [2]], ["bad", 1, 2],
])
def test_invalid_demand_scenarios_are_rejected(demands):
    env = InventoryEnv(horizon=3)
    with pytest.raises(ValueError):
        env.reset(options={"demands": demands})


@pytest.mark.parametrize("kwargs", [
    {"horizon": 0}, {"horizon": -1}, {"horizon": 2.5},
    {"lead_time": 0}, {"lead_time": -1}, {"lead_time": 1.5},
    {"capacity": 0}, {"capacity": -1}, {"capacity": 1.5},
    {"initial_inventory": -1}, {"initial_inventory": 251}, {"initial_inventory": 1.5},
    {"max_order": -1}, {"max_order": 2.5},
    {"demand_mean": -1}, {"demand_mean": float("nan")},
    {"penalty_multiplier": -1}, {"penalty_multiplier": float("nan")},
    {"price": -1}, {"price": float("inf")}, {"unit_cost": -1},
    {"holding_cost": -1}, {"stockout_penalty": -1},
])
def test_invalid_environment_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        InventoryEnv(**kwargs)
