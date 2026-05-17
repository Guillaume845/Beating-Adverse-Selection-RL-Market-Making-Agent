# RL Market Maker (Glosten-Milgrom)

Built this side project to mess around with Reinforcement Learning applied to market microstructure. 

I wanted to see if a vanilla PPO agent could "rediscover" the Glosten-Milgrom (1985) model without me hardcoding any of the Bayesian math or pricing logic. Spoiler: it actually does.

### The Idea
The agent acts as a market maker quoting bid-ask spreads. The market environment I built (using `Gymnasium`) throws two types of traders at it:
- **Noise traders**: They buy/sell randomly. Free money for the spread.
- **Informed traders**: Toxic flow. They know the true hidden price of the asset. 

If the agent keeps a tight spread all the time, it gets completely run over by the informed traders (adverse selection). Over thousands of episodes, the agent figures out that getting hit sequentially on the same side means the flow is toxic. It learns to dynamically widen its spread to stop the bleeding, and it aggressively mean-reverts its inventory to avoid getting caught holding the bag.

### Getting started

Install the basics:
```bash
pip install gymnasium stable-baselines3
