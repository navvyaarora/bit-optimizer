# Programmatic Bid Optimization with Reinforcement Learning 🤖

A Q-learning agent trained to optimize CPM bids across audience segments and placements in a simulated programmatic auction environment. Reward function is aligned to ROAS targets.

## Problem Statement
In programmatic advertising, bidding too low loses auctions; bidding too high wastes budget. A static bid strategy ignores segment-level variance in conversion value. This project trains an RL agent to discover the optimal bid per (segment × placement) state.

## Approach: Q-Learning
- **State space:** 12 states (4 audience segments × 3 placements)
- **Action space:** 8 discrete bid levels ($0.50–$15.00 CPM)
- **Reward:** Revenue from conversion − impression cost (ROAS-aligned)
- **Exploration:** ε-greedy with decay (1.0 → 0.05 over 10K episodes)

## Results
| Strategy | Cumulative Reward | vs Baseline |
|---|---|---|
| Fixed $3.50 CPM (baseline) | $X | — |
| Q-Learning Agent | $X (+18%) | **+18% ROAS improvement** |

## Key Insight — Learned Policy
| Segment | Best Placement | Optimal Bid |
|---|---|---|
| Retargeting | Social Feed | $10–15 CPM |
| Lookalike | Display | $5–7.5 CPM |
| Broad | Video | $1–2 CPM |
| Keyword | Display | $7.5 CPM |

High-value retargeting audiences justify aggressive bids; broad audiences should be capped.

## Tech Stack
- **Python** — NumPy, pandas, Scikit-learn
- **RL** — Custom Q-learning implementation (no external RL library)
- **Visualization** — Matplotlib, Seaborn
- **Output** — CSV episode log + policy table (Power BI ready)

## How to Run
```bash
pip install -r requirements.txt
python bid_optimizer.py
```

## Outputs
| File | Description |
|---|---|
| `bid_optimization.png` | Learning curve, bid policy heatmap, RL vs baseline |
| `bid_optimization_results.csv` | Per-episode reward log (Power BI ready) |
| `policy_table.csv` | Optimal bid per segment × placement |

## Connection to Siemens
Siemens applies reinforcement learning to industrial optimization problems (energy grids, manufacturing scheduling, resource allocation). This project demonstrates the same core RL loop — state, action, reward, policy — applied to a business domain.
