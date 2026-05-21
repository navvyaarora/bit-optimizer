"""
Programmatic Bid Optimization with Reinforcement Learning
==========================================================
Trains a Q-learning agent to optimize CPM bids across audience segments
in a simulated programmatic auction environment. Reward function is aligned
to ROAS targets (value-per-click / bid cost).

Outputs:
  - bid_optimization_results.csv   → per-episode performance (Power BI ready)
  - bid_optimization.png           → learning curves + policy heatmap + CPA comparison
  - policy_table.csv               → learned Q-table bid policy per state
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── 1. AUCTION ENVIRONMENT ──────────────────────────────────────────────────────
AUDIENCE_SEGMENTS = ["retargeting", "lookalike", "broad", "keyword"]
PLACEMENTS        = ["social_feed", "display", "video"]
BID_LEVELS        = [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0]  # CPM USD

# Base win rates and conversion rates per (segment, placement)
WIN_RATE = {
    ("retargeting", "social_feed"): 0.75, ("retargeting", "display"): 0.65, ("retargeting", "video"): 0.60,
    ("lookalike",   "social_feed"): 0.60, ("lookalike",   "display"): 0.50, ("lookalike",   "video"): 0.45,
    ("broad",       "social_feed"): 0.45, ("broad",       "display"): 0.35, ("broad",       "video"): 0.30,
    ("keyword",     "social_feed"): 0.55, ("keyword",     "display"): 0.60, ("keyword",     "video"): 0.40,
}

CONV_VALUE = {   # expected revenue per converted impression (USD)
    ("retargeting", "social_feed"): 12.0, ("retargeting", "display"): 9.0, ("retargeting", "video"): 10.5,
    ("lookalike",   "social_feed"): 7.5,  ("lookalike",   "display"): 6.0, ("lookalike",   "video"): 7.0,
    ("broad",       "social_feed"): 4.0,  ("broad",       "display"): 3.5, ("broad",       "video"): 3.0,
    ("keyword",     "social_feed"): 8.5,  ("keyword",     "display"): 9.5, ("keyword",     "video"): 6.0,
}

CONV_RATE = {    # click-to-conversion rate
    ("retargeting", "social_feed"): 0.12, ("retargeting", "display"): 0.09, ("retargeting", "video"): 0.10,
    ("lookalike",   "social_feed"): 0.07, ("lookalike",   "display"): 0.06, ("lookalike",   "video"): 0.065,
    ("broad",       "social_feed"): 0.03, ("broad",       "display"): 0.025, ("broad",       "video"): 0.02,
    ("keyword",     "social_feed"): 0.09, ("keyword",     "display"): 0.11, ("keyword",     "video"): 0.07,
}

def simulate_auction(segment, placement, bid):
    """Simulate one auction impression. Returns (won, reward)."""
    key = (segment, placement)
    base_win = WIN_RATE[key]
    # Higher bid → higher win probability (diminishing returns)
    win_prob = min(base_win + (bid / 30), 0.97)
    won = np.random.rand() < win_prob
    if not won:
        return False, 0.0

    conv_rate  = CONV_RATE[key] * (1 + np.random.normal(0, 0.1))
    conv_value = CONV_VALUE[key] * (1 + np.random.normal(0, 0.05))
    revenue    = conv_rate * conv_value
    cost       = bid / 1000  # CPM → cost per impression
    reward     = revenue - cost  # ROAS-aligned reward

    return True, max(reward, -cost)

# ── 2. Q-LEARNING AGENT ─────────────────────────────────────────────────────────
# State: (segment_idx, placement_idx)  → 4 × 3 = 12 states
# Action: bid level index              → 8 actions
n_segments   = len(AUDIENCE_SEGMENTS)
n_placements = len(PLACEMENTS)
n_actions    = len(BID_LEVELS)

Q = np.zeros((n_segments, n_placements, n_actions))

ALPHA        = 0.15   # learning rate
GAMMA        = 0.95   # discount factor
EPSILON      = 1.0    # exploration rate (decays)
EPSILON_MIN  = 0.05
EPSILON_DECAY= 0.9995
N_EPISODES   = 10000
IMPS_PER_EP  = 50     # impressions per episode

print("Training Q-learning bid optimization agent...")
print(f"  States: {n_segments * n_placements} | Actions: {n_actions} | Episodes: {N_EPISODES:,}")

episode_rewards  = []
episode_wins     = []
epsilon_history  = []

for ep in range(N_EPISODES):
    seg_idx   = np.random.randint(n_segments)
    place_idx = np.random.randint(n_placements)

    ep_reward = 0
    ep_wins   = 0

    for _ in range(IMPS_PER_EP):
        # ε-greedy action selection
        if np.random.rand() < EPSILON:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[seg_idx, place_idx])

        bid = BID_LEVELS[action]
        seg  = AUDIENCE_SEGMENTS[seg_idx]
        plc  = PLACEMENTS[place_idx]

        won, reward = simulate_auction(seg, plc, bid)
        ep_reward  += reward
        ep_wins    += int(won)

        # Next state (random transition — episodic)
        next_seg   = np.random.randint(n_segments)
        next_place = np.random.randint(n_placements)

        # Q-update
        best_next  = np.max(Q[next_seg, next_place])
        Q[seg_idx, place_idx, action] += ALPHA * (
            reward + GAMMA * best_next - Q[seg_idx, place_idx, action]
        )

        seg_idx, place_idx = next_seg, next_place

    episode_rewards.append(ep_reward)
    episode_wins.append(ep_wins)
    epsilon_history.append(EPSILON)
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

print(f"  Training complete. Final ε = {EPSILON:.4f}")

# ── 3. POLICY EXTRACTION ────────────────────────────────────────────────────────
policy_rows = []
for si, seg in enumerate(AUDIENCE_SEGMENTS):
    for pi, plc in enumerate(PLACEMENTS):
        best_action = np.argmax(Q[si, pi])
        best_bid    = BID_LEVELS[best_action]
        best_q      = Q[si, pi, best_action]
        policy_rows.append({
            "segment": seg, "placement": plc,
            "optimal_bid_cpm": best_bid, "q_value": round(best_q, 4)
        })

policy_df = pd.DataFrame(policy_rows)
policy_df.to_csv("policy_table.csv", index=False)
print("Policy table saved → policy_table.csv")

# ── 4. BASELINE COMPARISON ──────────────────────────────────────────────────────
# Baseline: always bid median ($3.5 CPM)
N_TEST = 2000
rl_rewards, base_rewards = [], []
rl_wins,    base_wins    = [], []

for _ in range(N_TEST):
    si  = np.random.randint(n_segments)
    pi  = np.random.randint(n_placements)
    seg = AUDIENCE_SEGMENTS[si]
    plc = PLACEMENTS[pi]

    # RL agent action (greedy)
    rl_action = np.argmax(Q[si, pi])
    rl_bid    = BID_LEVELS[rl_action]
    won_rl, r_rl = simulate_auction(seg, plc, rl_bid)

    # Baseline: fixed $3.5 CPM
    won_b, r_b = simulate_auction(seg, plc, 3.5)

    rl_rewards.append(r_rl); base_rewards.append(r_b)
    rl_wins.append(int(won_rl)); base_wins.append(int(won_b))

rl_total   = sum(rl_rewards)
base_total = sum(base_rewards)
improvement = (rl_total - base_total) / abs(base_total) * 100

print(f"\n  RL Agent total reward:   ${rl_total:.2f}")
print(f"  Baseline total reward:   ${base_total:.2f}")
print(f"  Improvement over baseline: {improvement:.1f}%")

# ── 5. PLOTS ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 5a. Learning curve (smoothed)
window = 300
smoothed = pd.Series(episode_rewards).rolling(window).mean()
axes[0].plot(smoothed, color="#1B5FA8", linewidth=1.5)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_title("Agent Learning Curve (Rolling Avg)", fontweight="bold")
axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Total Reward per Episode")

# 5b. Optimal bid heatmap
pivot = policy_df.pivot(index="segment", columns="placement", values="optimal_bid_cpm")
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues", ax=axes[1], cbar_kws={"label": "Optimal CPM Bid ($)"})
axes[1].set_title("Learned Bid Policy Heatmap (CPM $)", fontweight="bold")

# 5c. RL vs Baseline reward comparison
cats   = ["RL Agent", "Baseline ($3.5 CPM)"]
totals = [rl_total, base_total]
colors = ["#1B5FA8", "#AABDD6"]
bars = axes[2].bar(cats, totals, color=colors, width=0.4)
axes[2].set_title(f"Total Reward: RL vs Baseline\n(+{improvement:.1f}% improvement)", fontweight="bold")
axes[2].set_ylabel("Cumulative Reward ($)")
for bar, val in zip(bars, totals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"${val:.1f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("bid_optimization.png", dpi=150)
plt.close()
print("Plots saved → bid_optimization.png")

# ── 6. EXPORT EPISODE LOG ───────────────────────────────────────────────────────
ep_log = pd.DataFrame({
    "episode": range(1, N_EPISODES + 1),
    "total_reward": episode_rewards,
    "impressions_won": episode_wins,
    "epsilon": epsilon_history,
    "smoothed_reward": pd.Series(episode_rewards).rolling(100, min_periods=1).mean().round(4)
})
ep_log.to_csv("bid_optimization_results.csv", index=False)
print("Episode log saved → bid_optimization_results.csv")

print("\n" + "="*50)
print("OPTIMAL BID POLICY LEARNED")
print("="*50)
print(policy_df.to_string(index=False))
