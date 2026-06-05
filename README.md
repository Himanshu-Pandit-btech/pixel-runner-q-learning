# Pixel Runner AI

Pixel Runner AI is a small Pygame experiment where 50 parallel Q-learning agents learn to play a runner game. Each generation runs all agents, selects the best performer as the champion, and starts the next generation with 49 offspring.

## Features

- 50 agents training in parallel
- Shared Q-table learning
- Champion and offspring generation loop
- Frame-based scoring for stable training
- HUD with generation, champion, score, epsilon, dodges, and Q-table heatmap
- Autosaves learned Q-values during training

## Project Files

The main files are:

- `main.py` - game loop, rendering flow, generation handling
- `agent.py` - Q-table, agent physics, rewards, evolution behavior
- `game_utils.py` - constants, renderer, HUD

Asset folders:

- `graphics/`
- `audio/`
- `font/`

## Setup

Install Python 3.10+ and then install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Controls:

- `S` - save Q-table manually
- `Q` or `Esc` - save and quit

## Training Notes

The game creates a learned Q-table file while it runs. This file is intentionally ignored by Git because it is generated training data:

```text
q_table*.npy
```

If you want to restart learning from scratch, delete the generated Q-table file and run the game again.

## GitHub Upload

Recommended files to commit:

- Python source files
- `graphics/`, `audio/`, and `font/` assets
- `README.md`
- `requirements.txt`
- `.gitignore`
- `LICENSE`

Do not commit:

- `__pycache__/`
- `.pyc` files
- generated `q_table*.npy` files

