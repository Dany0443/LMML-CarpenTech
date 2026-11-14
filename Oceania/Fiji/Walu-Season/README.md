# Walu Season 

## Overview

The goal was to develop a strategy that catches at least **5,300 kg of fish per day**. Each day, the agent has 1,000 casts across 10 fishing spots with unknown reward distributions. The spots are shuffled daily, so the agent must learn from scratch every day.

## Approach

We used the **Upper Confidence Bound (UCB)** algorithm to balance exploration and exploitation.

Initially, we tested a trimmed mean version of UCB, which ignored unusually large catches. However, this performed poorly on reefs where rare walu runs were the main difference between fishing spots.

We switched to standard UCB, which considers every catch, including rare large rewards.

The algorithm works by first trying each fishing spot at least once. It then tracks the average catch and how often each spot has been selected. Based on this information, it prioritizes spots with higher average rewards while occasionally exploring less-tested ones.

## Testing

We compared both approaches across different reef configurations.

| Test | Standard UCB | Trimmed Mean UCB |
|---|---:|---:|
| Practice reef | 5,691 kg/day | 5,816 kg/day |
| Local evaluation | 7,191 kg/day | 6,730 kg/day |
| Runs-only reef | 5,465 kg/day | 4,680 kg/day |

Standard UCB was more consistent, especially when rare catches determined which spots were most profitable.

## Final Results

The official evaluation ran for 100 days and returned:

- **Average catch:** 6,241.5 kg/day
- **Required average:** 5,300 kg/day
- **Worst / best day:** 2,472.6 / 8,497.8 kg
- **Runtime:** 4.5 seconds
- **Status:** PASS

## Conclusion

The key insight was that rare, unusually large catches were valuable signals rather than just noise. Standard UCB preserved this information and performed more reliably across different reefs.

The solution exceeded the required threshold by approximately **17.8%**.

**Flag: `SIGMOID_BAGGING`**