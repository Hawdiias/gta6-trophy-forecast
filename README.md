[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hawdiias/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hawdiias/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

A forecasting program for the make.fun bounty "Forecast how many GTA VI
players earn each trophy in week one". Given a case file holding every read
the house made of GTA VI's Exophase trophy page before a checkpoint, it
writes, for each of the game's 55 PlayStation 5 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of that trophy's earn rate at the case's target —
the percent of Exophase-tracked players who will have earned it.

## Method

An earn rate is a ratio, earners / players tracked, and its two sides move
for different reasons: the tracked pool keeps growing and dilutes every
rate, while the players already tracked keep progressing and lift earners,
story trophies first. The program therefore forecasts the two forces
separately — the pool from its own growth curve, each trophy's earners in
log space — and divides at the end. In parallel it fits the rate itself in
logit space against ln(1 + hours since launch), with the slope damped and
shrunk toward the pooled slope of the board's well-observed trophies, so a
trophy still sitting near zero borrows the board's progression wave instead
of being forecast flat. The median blends the two forecasts and is clamped
in how far it may travel from the last observed rate — in v2 the clamp was
tightened to +1.5 / −1.2 logit after synthetic holdout validation showed
the original 3.2 / 2.6 caps let the extrapolation overshoot the realised
rates, costing about a third of the total pinball loss; the same validation
shortened the recency half-life of the fits from 14h to 8h.

Page reads are step functions — the same page is re-read many times between
Exophase refreshes — so every fit runs over the case's distinct page states,
weighted by recency in clock time, never over the raw reads. The spread
starts from a random-walk width and widens for trophies whose fit is poor,
whose recent states jump around, or whose earner count is so small that
counting noise alone moves the rate; a case with almost no history gets a
wide spread, because one unseen refresh can move every rate at once.

The program is a pure function of its case file: no clock, no network, no
randomness — two runs write byte-identical forecasts.

## Running

The harness runs it as `python3 main.py` (Python 3.13, standard library
only). It reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`)
and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else
`/output`).

## License

MIT — see LICENSE.
