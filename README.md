"# alpha zero tutorial test" 

Notes
Goal here with alpha go is Self-Play --> Generate Training Date --> Training --> Optimize Model for Self Play --> Self-Play

Model

f(state) = (policy, value)
Each action, will tell us how well the action is

Monte Carlo Tree Search
- Find most promising action
- Start with root and build into the future

1. Selection | Walk down until Leaf Node
2. Expansion | Create New Node
3. Simulation | Play Randomly
4. Backpropogation - if situation win, update all nodes traversed with wins and number of times nodes traversed


Node - win (w), node (how many times took this action) - win = 3, took action = 4
Find most promising route
Calc winning ratio

This balances exploration and exploitation

It performs random sampling in the form of simulations and stores the statistics of actions to make more educated choices in each subsequent iteration.

Monte Carlo tree search only searches a few layers deep into the tree and prioritizes which parts of the tree to explore.

UCB Formula

Wi/Ni + Constant * (sqrt(log (Total Number of simulations done for the parent node) / Number of simulations done for the current child node))


General AI ( How to define win situation?) Chess, GO is relatively "simple"