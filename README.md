# GTA VI week-one trophy forecast

[![make.fun competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Leaderboard rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/athrony/gta6-trophy-rarity-forecast-01-asong/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![Competition reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/athrony/gta6-trophy-rarity-forecast-01-asong/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

This offline Python 3.13 forecaster estimates what percentage of tracked PS5 players will earn each GTA VI trophy. It models new players joining separately from existing players unlocking trophies, then combines those estimates with observed rate trends to produce five quantiles per trophy.

## Run it

Place the `*.in` case files in `input/`. From the repository root, run

```sh
MAKEFUN_INPUT_DIR=./input MAKEFUN_OUTPUT_DIR=./output python3 main.py
```

Each case produces a matching `.out` JSON file in `output/`.

---

<sub>Explore more AI coding bounties on [make.fun](https://make.fun/?ref=bs4h7b2w&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01).</sub>
