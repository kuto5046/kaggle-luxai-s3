# kaggle luxai season3
[NeurIPS 2024 - Lux AI Season 3](https://www.kaggle.com/competitions/lux-ai-season-3/overview)

## setup & run
```bash
uv sync
```
### imitation learning
You have to prepare the episode data in local.
```bash
uv run exp/best/data_processor.py
uv run exp/best/train.py
```

### reinforcement learning
I tried RL but not good.
```bash
uv run exp/rl_best/rl.py
```

---
## Solution
Our solution is following.
https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568789

### Summary

- Imitation learning based on match results of top teams
- Selected high-performing top teams submissions and carry out two-stage fine tuning.
- Adopted a UNet architecture to predict map-based policy and SAP probabilities
- Estimated unknown environmental parameters and hidden states
- Assigned actions to each unit using minimum cost flow

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2603247%2F596b965ef06160a106c3af7edaa06547%2FLuxAIS3%20Solution%20(2).svg?generation=1742258560784280&alt=media)
  
---
### Dataset

- **Stage 1**
    - Frog Parade (submission_id=42704976)
    - 7,935 episodes
    - Trained only on winning games
- **Stage 2**
    - Frog Parade (submission_id=43276830)
    - 1,550 episodes
    - Trained on both winning and losing games
- Mixing data from different policies (multiple submissions) did not significantly improve performance.
  
---

### Features

- Created 14-channel map features and 16-channel scalar features.
- Normalized these features to the range of -1 to 1, except for some exceptions.
- Standardized normalization criteria (e.g., `OWN_UNIT_ENERGY` and `UNIT_MOVE_COST` both represent energy values and were normalized using a common scale).
- Applied mirroring so that the initial position of the home team is always (0,0).

#### Map Features (14 channels)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2603247%2Fa3319de3171d3ba98071629d79b59f63%2Fimage.png?generation=1742257505470205&alt=media)

| Variable Name | Description |
| --- | --- |
| `TILE_TYPE` | The type of the current tile (`UNKNOWN = -1, EMPTY = 0, NEBULA = 1, ASTEROID = 2`) |
| `NEXT_TILE_TYPE` | The type of the tile in the next step |
| `ENERGY` | (Estimated) The amount of energy on the tile |
| `NEBULA_ENERGY_REDUCTION` | (Estimated) Energy reduction caused by nebula tiles |
| `SENSOR_MASK` | Area visible to sensors |
| `RELICS` | Locations of relic nodes |
| `POINTS` | (Estimated) Nodes where points can be gained near relic nodes |
| `ENTROPY` | Entropy of estimated `POINTS` |
| `OWN_UNIT_COUNT` | Number of own units per cell |
| `OWN_UNIT_ENERGY` | Energy of own units per cell |
| `OPP_UNIT_COUNT` | (Estimated) Number of opponent units per cell |
| `OPP_UNIT_ENERGY` | Energy of opponent units per cell |
| `VISIT_COUNT` | Number of times each tile has been visited in the current match |
| `SAP_AVAILABLE_AREA` | Area where sap can be used |

#### Scalar Features (16 channels)

| Variable Name | Description |
| --- | --- |
| `MATCH_STEPS` | Number of steps elapsed in the current match |
| `MATCH_COUNT` | Current match number |
| `TEAM_POINTS` | Points earned by the team (own team points - opponent team points) |
| `TEAM_WINS` | Number of matches won by the team (own wins - opponent wins) |
| `UNIT_MOVE_COST` | Energy cost for unit movement |
| `UNIT_SAP_COST` | Energy cost for sap action |
| `UNIT_SAP_RANGE` | Range within which sap can be used |
| `UNIT_SENSOR_RANGE` | Sensor range of units |
| `NEBULA_TILE_VISION_REDUCTION_MEAN` | Mean vision reduction by nebula tiles |
| `NEBULA_TILE_VISION_REDUCTION_SIGMA` | Standard deviation of vision reduction by nebula tiles |
| `ENERGY_NODE_DRIFT_SPEED_MEAN` | Mean movement speed of energy nodes |
| `ENERGY_NODE_DRIFT_SPEED_SIGMA` | Standard deviation of movement speed of energy nodes |
| `NEBULA_TILE_ENERGY_REDUCTION_MEAN` | Mean energy reduction by nebula tiles |
| `NEBULA_TILE_ENERGY_REDUCTION_SIGMA` | Standard deviation of energy reduction by nebula tiles |
| `UNIT_SAP_DROPOFF_FACTOR_MEAN` | Mean energy drop-off factor for sap action |
| `UNIT_SAP_DROPOFF_FACTOR_SIGMA` | Standard deviation of energy drop-off factor for sap action |

### Estimation of Hidden States

Unknown environmental parameters and hidden states, which vary randomly per match, were estimated using the following methods:

- **`NEXT_TILE_TYPE`**
    - Estimated tile drift speed from observed tile movements and predicted the next tile state accordingly.
- **`ENERGY`**
    - Estimate the overall energy map by narrowing down the energy nodes from the partially observed energy map by a Exhaustive search
- **`POINTS`**
    - Estimate point probabilities by Bayesian inference.
    - Initialized relic-adjacent tiles with 0.5 and others with 0.0.
- **`OPP_UNIT_COUNT`**
    - Track all possible actions of previously observed enemy units to estimate their existence probabilities.
- **`NEBULA_TILE_VISION_REDUCTION`**
    - Estimated based on observed unit vision changes when stepping on nebula tiles.
- **`ENERGY_NODE_DRIFT_SPEED`**
    - Prepared candidate drift speeds and refined estimates using Bayesian inference.
- **`NEBULA_TILE_ENERGY_REDUCTION`**
    - Estimated from observed unit energy reductions when stepping on nebula tiles.
- **`UNIT_SAP_DROPOFF_FACTOR`**
    - Estimated based on observed energy values of adjacent enemy units.

---
### Model Architecture
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2603247%2Ff50dfe138452d75cab6a32ffc9edfa98%2Fimage.png?generation=1742264183178712&alt=media)
- Based on the 6th place LuxAI Season 1 solution using a UNet architecture ([link](https://www.kaggle.com/competitions/lux-ai-2021/discussion/293776)).
- **Inputs**
    - Map features `(14, n_stack, 24,24)`
    - Scalar features `(16, n_stack)`
    - `n_stack` is a parameter for how many steps in the past to input the state, we use n_stack=4. This means that the states of the past 4steps are input to the model as features.
- **Outputs**
    - Policy logits `(6,24,24)`
        - `CENTER = 0, UP = 1, RIGHT = 2, DOWN = 3, LEFT = 4, SAP = 5`
    - SAP policy logits `(1,24,24)`
        - Probability of sap in that cell.
- **Model layers**
    - UNet (encoder/decoder)
    - SAP Policy Layer
        - Predicted SAP policy using a residual convolutional layer.
        - Stabilized training by adding batch normalization in the final layer.
    - Policy Layer
        - Combined SAP policy with UNet decoder output and predicted policy using a residual convolutional layer.

---
### Training

- loss
    - policy → DiceLoss
    - sap policy → Custom Focal Tversky Loss (weight=0.1)
        - Sap targets are on the (24,24) map (1 if the target team is at the sap location, 0 otherwise)
        - The target was masked so that only `SAP_AVAILABLE_AREA` was the training target because the target was sparse, and Focal Tversky Loss was employed to counter sparse targets.
        - Set loss weight is 0.1 not to inhibit policy learning.
- training setting
    - epoch=30
    - lr=1e-3
    - batch_size=1024
- augmentation
    - Applied mirroring along the x and y axes.

---
### Action Selection

- Predicted two map-based policies using a neural network:
    - **Policy (6, 24, 24)**
        - Actions: `CENTER = 0, UP = 1, RIGHT = 2, DOWN = 3, LEFT = 4, SAP = 5`
        - Applied softmax on valid actions to convert logits into probabilities.
        - Sampled actions probabilistically like following:
        
        ```python
        action = np.random.choice(range(6), p=policy)
        ```
        
    - **SAP Policy (1, 24, 24)**
        - Predicted SAP probability per cell.
- **Action assignment by minimum cost flow**
    - Units in the same cell tend to take the same action, so actions were assigned using minimum cost flow based on unit policies.
    1. **Assignment of SAP Actions**
        1. Enumerate the units (SAP units) for which a SAP action is sampled at a given step.
        2. For each SAP unit, list the SAP position candidates based on the sap map and the next-best move action after the sap action with its probability. (to avoid unnecessary sap when the sap map value is very small around a unit).
        3. Define cost of target sap point(x,y) as `cost = 1 - sap_policy[y,x] * policy[5]` if it is a sap action, `cost = 1 - policy[action_idx]`otherwise
        4. Assign actions to SAP units by minimizing the total cost using minimum cost flow.
    2. **Assignment of Move Actions**
        1. Enumerate the units (MOVE units) that are not SAP units at a given step.
        2. For each move unit, list all possible actions(without sap) along with their probability values.
        3. Define the cost of each action as `cost = 1 - unit_policy[action_idx]`  and add a penalty cost for moving to an already occupied cell.
        4. Assign actions to MOVE units by minimizing the total cost using minimum cost flow.

- **TTA (Test Time Augmentation)**
    - Applied mirroring along the x and y axes.

---
### Performance Evaluation

#### Visualizer (by streamlit)

- Checked features estimation results.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2603247%2F26f2536cc2bf3fa307521b6a3c7ca79e%2Fimage4.png?generation=1742257445637244&alt=media)

- Checked unit policies from match results.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2603247%2Fc1ec36dddb5de597d5578f7a666e3f2f%2Fimage2.png?generation=1742257461980005&alt=media)]

#### Custom Tournament Evaluation

- Modified `luxai_runner` to display win rate and p-value in a tournament setting.
- Conducted a sufficient number of local matches (e.g., 100 episodes) and adopted model changes only if the win rate significantly improved over the best existing model.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2603247%2Fb76b4431a94e0457b58a3a727e1c2199%2Fimage3.png?generation=1742257374557581&alt=media)
