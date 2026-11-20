"""Offline, refresh-aware trophy forecasts using a deterministic model ensemble."""

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
DAY = 86400.0
LOW_LOGIT = math.log(0.0001 / 0.9999)
HIGH_LOGIT = -LOW_LOGIT


def bounded(value, lower, upper):
    return min(upper, max(lower, value))


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def to_logit(rate):
    probability = bounded(rate / 100.0, 0.0001, 0.9999)
    return math.log(probability / (1.0 - probability))


def to_percent(value):
    return 100.0 / (1.0 + math.exp(-bounded(value, -35.0, 35.0)))


@dataclass
class Read:
    time: float
    seen_until: float
    players: float
    earned: float | None
    rates: tuple


@dataclass(frozen=True)
class Profile:
    group: str
    ceiling: float
    days: float
    shape: float
    uncertainty: float


def trophy_profile(trophy):
    description = trophy.get("description", "").lower()
    tier = trophy.get("tier", "bronze").lower()
    secret = bool(trophy.get("secret", False))
    if tier == "platinum":
        return Profile("completion", 0.035, 19.0, 3.1, 0.95)
    if "100% completion" in description:
        return Profile("completion", 0.10, 13.0, 2.8, 0.85)
    chapter = re.search(r"complete chapter\s+(\d+)", description)
    if chapter:
        chapter_number = int(chapter.group(1))
        return Profile("story", max(0.50, 0.94 - 0.065 * chapter_number),
                       0.65 * 1.75 ** min(chapter_number - 1, 8), 1.65, 0.50)
    if "complete the final mission" in description and "mission for" not in description:
        return Profile("story", 0.61, 5.5, 2.0, 0.65)
    if secret and description.startswith("complete ") and "mission for" not in description:
        return Profile("story", 0.70, 3.4, 1.8, 0.65)
    if "mission for" in description or "mission strands" in description:
        return Profile("side", 0.35, 5.2, 2.0, 0.75)
    demanding = ("all ", "every ", "250 ", "100 locations", "50 challenges",
                 "20 different", "collectible", "gold medal")
    if any(phrase in description for phrase in demanding):
        return Profile("grind", 0.16 if tier == "silver" else 0.24, 7.0, 2.2, 0.80)
    if tier in ("gold", "silver"):
        return Profile("grind", 0.23, 5.5, 2.0, 0.75)
    return Profile("side", 0.22 if secret else 0.34,
                   3.7 if secret else 2.7, 1.65, 0.65 if secret else 0.55)


def read_history(case, checkpoint, launch):
    width = len(case["trophies"])
    states = []
    last_players = 1.0
    for row in case.get("history", []):
        instant = timestamp(row.get("at") or row["slot"])
        if instant > checkpoint:
            continue
        players = row.get("players_tracked")
        if number(players) and players > 0:
            last_players = float(players)
        rates = [None] * width
        for index, rate in enumerate(row.get("rates", [])[:width]):
            if number(rate):
                rates[index] = bounded(float(rate), 0.0, 100.0)
        earned = row.get("trophies_earned")
        earned = float(earned) if number(earned) and earned >= 0 else None
        age = (instant - launch) / DAY
        rates = tuple(rates)
        if states and (last_players, earned, rates) == (states[-1].players, states[-1].earned, states[-1].rates):
            states[-1].seen_until = age
        else:
            states.append(Read(age, age, last_players, earned, rates))
    if not states:
        states.append(Read((checkpoint - launch) / DAY, (checkpoint - launch) / DAY,
                           1.0, None, (None,) * width))
    if len(states) > 48:
        indices = {round(index * (len(states) - 1) / 47) for index in range(48)}
        for trophy_index in range(width):
            latest_known = next((position for position in range(len(states) - 1, -1, -1)
                                 if states[position].rates[trophy_index] is not None), len(states) - 1)
            indices.add(latest_known)
        states = [state for index, state in enumerate(states) if index in indices]
    return states


def arrival_model(states, now, horizon):
    observations = []
    for earlier, later in zip(states, states[1:]):
        distance = math.log((max(later.time, 0.0) + 0.4) / (max(earlier.time, 0.0) + 0.4))
        if distance > 0.015 and later.players >= earlier.players:
            exponent = math.log(later.players / earlier.players) / distance
            weight = distance * math.exp(-max(0.0, now - later.time) / 1.5)
            observations.append((bounded(exponent, 0.0, 3.5), weight))
    support = sum(weight for _, weight in observations)
    exponent = (0.20 * 0.9 + sum(value * weight for value, weight in observations)) / (0.20 + support)
    exponent = bounded(exponent, 0.0, 2.8)
    intervals = [later.time - earlier.time for earlier, later in zip(states, states[1:])
                 if later.time > earlier.time]
    refresh_days = bounded(median(intervals[-12:]) if intervals else 0.25, 1.0 / 48, 1.0)
    stale_days = max(0.0, now - states[-1].time)
    refresh_probability = -math.expm1(-min(50.0,
        ((stale_days + horizon) / refresh_days) ** 1.35 - (stale_days / refresh_days) ** 1.35))
    effective_horizon = max(0.0, now + horizon - states[-1].time - min(refresh_days / 2, horizon / 4))
    final_age = states[-1].time + effective_horizon
    age_ratio = (max(final_age, 0.0) + 0.4) / (max(states[-1].time, 0.0) + 0.4)
    growth = max(1.0, age_ratio ** exponent)
    uncertainty = (0.23 + 0.45 / math.sqrt(1 + len(observations))) * math.sqrt(min(effective_horizon, 8.0))
    if observations:
        exponent_deviation = median(abs(value - exponent) for value, _ in observations)
        uncertainty = math.hypot(uncertainty, 0.35 * exponent_deviation * math.log(age_ratio))
    else:
        pending_batch = 0.9 * math.exp(-max(now, 0.0) / 1.5)
        uncertainty = math.hypot(uncertainty, pending_batch)
    return growth, uncertainty, final_age, refresh_probability


def historical_cohorts(states):
    cohorts = [(min(states[0].time, 0.0), states[0].players)]
    previous_players = states[0].players
    for earlier, later in zip(states, states[1:]):
        if later.players < previous_players:
            retained = later.players / previous_players
            cohorts = [(joined, players * retained) for joined, players in cohorts]
        increase = max(0.0, later.players - previous_players)
        if increase:
            cohorts.append(((earlier.seen_until + later.time) / 2, increase))
        previous_players = later.players
    return cohorts


def completion_mass(cohorts, age, profile, speed):
    total = 0.0
    scale = profile.days * speed
    for joined, players in cohorts:
        elapsed = max(0.0, age - joined)
        completion = -math.expm1(-min(40.0, (elapsed / scale) ** profile.shape))
        total += players * profile.ceiling * completion
    return total


def future_cohorts(cohorts, last, final_age, growth):
    result = list(cohorts)
    duration = max(0.0, final_age - last.time)
    steps = min(32, max(1, math.ceil(duration * 8)))
    age_base = max(last.time, 0.0) + 0.4
    log_span = math.log((max(final_age, 0.0) + 0.4) / age_base)
    previous = last.players
    for step in range(1, steps + 1):
        age = last.time + duration * step / steps
        fraction = math.log((max(age, 0.0) + 0.4) / age_base) / log_span if log_span > 1e-9 else step / steps
        players = last.players * growth ** bounded(fraction, 0.0, 1.0)
        result.append((age - duration / (2 * steps), max(0.0, players - previous)))
        previous = players
    return result


def growth_evidence(states, index, masses):
    exposure = 0.0
    earned = 0.0
    correction = 0.0
    for position in range(1, len(states)):
        earlier, later = states[position - 1], states[position]
        old_rate, new_rate = earlier.rates[index], later.rates[index]
        if old_rate is None or new_rate is None:
            continue
        old_count = earlier.players * old_rate / 100.0
        new_count = later.players * new_rate / 100.0
        rounding = 1.0 + 0.00005 * (earlier.players + later.players)
        change = new_count - old_count
        if change < -max(3 * rounding, old_count * 0.12):
            correction += 1.0
            continue
        weight = math.exp(-max(0.0, states[-1].time - later.time) / 1.25)
        weight /= 1.0 + rounding * rounding / max(1.0, abs(change))
        exposure += max(0.0, masses[position] - masses[position - 1]) * weight
        earned += change * weight
    return exposure, max(0.0, earned), correction


def trend_estimate(states, index):
    observations = [(state, state.rates[index]) for state in states if state.rates[index] is not None]
    if len(observations) < 2:
        return 0.0, 0.0, 0.0
    latest, last_rate = observations[-1]
    slopes = []
    changes = []
    for state, rate in observations[:-1]:
        elapsed = latest.time - state.time
        distance = math.log((max(latest.time, 0.0) + 0.5) / (max(state.time, 0.0) + 0.5))
        if elapsed < 1.0 / 24 or distance < 0.025:
            continue
        change = to_logit(last_rate) - to_logit(rate)
        slopes.append(bounded(change / distance, -6.0, 6.0))
        changes.append(abs(change))
    span = max(0.0, latest.time - observations[0][0].time)
    confidence = (len(slopes) / (len(slopes) + 3.0)) * span / (span + 0.2)
    slope = median(slopes) * confidence if slopes else 0.0
    scatter = median(abs(value - median(slopes)) for value in slopes) if slopes else 0.0
    movement = median(changes) if changes else 0.0
    return slope, min(2.0, scatter), movement


def mixture_quantiles(components):
    total_weight = sum(weight for weight, _, _ in components)
    result = []
    for level in LEVELS:
        lower, upper = LOW_LOGIT, HIGH_LOGIT
        for _ in range(38):
            midpoint = (lower + upper) / 2
            cumulative = 0.0
            for weight, center, spread in components:
                if spread <= 1e-10:
                    probability = float(midpoint >= center)
                else:
                    probability = 0.5 * math.erfc((center - midpoint) / (math.sqrt(2.0) * spread))
                cumulative += weight * probability
            if cumulative < level * total_weight:
                lower = midpoint
            else:
                upper = midpoint
        value = round(to_percent((lower + upper) / 2), 6)
        result.append(max(result[-1] if result else 0.0, bounded(value, 0.0, 100.0)))
    return result


def forecast(case):
    checkpoint = timestamp(case["at"])
    launch = timestamp(case.get("game", {}).get("release_date") or case["at"])
    horizon = max(0.0, float(case.get("horizon_minutes", 0.0)) / 1440.0)
    if case.get("target"):
        horizon = max(0.0, (timestamp(case["target"]) - checkpoint) / DAY)
    now = (checkpoint - launch) / DAY
    states = read_history(case, checkpoint, launch)
    last = states[-1]
    profiles = [trophy_profile(trophy) for trophy in case["trophies"]]
    growth, growth_uncertainty, final_age, refresh_probability = arrival_model(states, now, horizon)
    cohorts = historical_cohorts(states)
    projected_cohorts = future_cohorts(cohorts, last, final_age, growth)
    models = []
    for speed in (0.65, 1.0, 1.7):
        evidence = []
        groups = {}
        for index, profile in enumerate(profiles):
            masses = [completion_mass(cohorts, state.time, profile, speed) for state in states]
            exposure, earned, correction = growth_evidence(states, index, masses)
            future_mass = completion_mass(projected_cohorts, final_age, profile, speed)
            anchor_index = next((position for position in range(len(states) - 1, -1, -1)
                                 if states[position].rates[index] is not None), len(states) - 1)
            evidence.append((exposure, earned, correction, max(0.0, future_mass - masses[anchor_index])))
            known_rates = [state.rates[index] for state in states if state.rates[index] is not None]
            if known_rates and min(known_rates) >= 99.99:
                continue
            pool = groups.setdefault(profile.group, [0.0, 0.0])
            influence = min(1.0, 200.0 / max(1.0, exposure))
            pool[0] += exposure * influence
            pool[1] += earned * influence
        models.append((evidence, groups))
    output = []
    for index, trophy in enumerate(case["trophies"]):
        profile = profiles[index]
        anchor = next((state for state in reversed(states) if state.rates[index] is not None), last)
        rate = anchor.rates[index]
        if rate is None:
            peers = [last.rates[position] for position, other in enumerate(profiles)
                     if other.group == profile.group and last.rates[position] is not None]
            rate = median(peers) if peers else 0.01
        current = to_logit(rate)
        if horizon <= 0:
            output.append({"id": trophy["id"], "q": [rate] * 5})
            continue
        slope, scatter, movement = trend_estimate(states, index)
        age_distance = math.log((max(final_age, 0.0) + 0.5) / (max(anchor.time, 0.0) + 0.5))
        drift = bounded(slope * age_distance, -3.5, 3.5)
        expected_count = anchor.players * max(rate, 0.005) / 100.0
        rarity_uncertainty = 0.45 / math.sqrt(1.0 + expected_count)
        duration_scale = math.sqrt(min(10.0, max(horizon, final_age - last.time)))
        base_spread = (0.32 + 0.25 * profile.uncertainty + rarity_uncertainty) * duration_scale
        if rate >= 99.99 and all(state.rates[index] in (None, 100, 99.99) for state in states):
            output.append({"id": trophy["id"], "q": [99.99, 99.999, 100.0, 100.0, 100.0]})
            continue
        information = (len(states) - 1) / (len(states) + 2.0)
        information *= min(1.0, max(0.0, last.time - states[0].time) / 0.5)
        curve_weight = 0.25 + 0.35 * information
        trend_weight = 0.25 + 0.10 * min(1.0, movement)
        persistence_weight = 1.0 - curve_weight - trend_weight
        components = [(1.0 - refresh_probability, current, 0.0)]
        components.append((refresh_probability * persistence_weight, current, max(0.08, base_spread)))
        components.append((refresh_probability * trend_weight, current + drift,
                           max(0.08, base_spread * 0.8 + 0.2 * scatter * age_distance)))
        for model_index, (evidence, groups) in enumerate(models):
            exposure, earned, correction, future_gain = evidence[index]
            pooled_exposure, pooled_earned = groups[profile.group]
            pooled_amplitude = bounded((5.0 + pooled_earned) / (5.0 + pooled_exposure), 0.003, 5.0)
            amplitude = bounded((2.0 * pooled_amplitude + earned) / (2.0 + exposure), 0.001, 8.0)
            new_count = max(0.0, anchor.players * rate / 100.0 + amplitude * future_gain)
            modeled_rate = bounded(100.0 * new_count / (last.players * growth), 0.0, 100.0)
            center = to_logit(modeled_rate)
            amplitude_uncertainty = 0.65 / math.sqrt(1.0 + earned / 3.0)
            spread = math.sqrt(base_spread ** 2 + growth_uncertainty ** 2
                               + (amplitude_uncertainty * min(2.0, duration_scale)) ** 2
                               + 0.15 * correction)
            model_weight = (0.25, 0.50, 0.25)[model_index]
            weight = refresh_probability * curve_weight * model_weight
            components.append((weight * 0.9, center, max(0.10, spread)))
            components.append((weight * 0.1, center, max(0.20, spread * 2.0)))
        output.append({"id": trophy["id"], "q": mixture_quantiles(components)})
    return {"trophies": output}


def run():
    input_directory = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_directory = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_directory.mkdir(parents=True, exist_ok=True)
    for source in sorted(input_directory.glob("*.in")):
        case = json.loads(source.read_text(encoding="utf-8"))
        prediction = forecast(case)
        encoded = (json.dumps(prediction, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > 65536:
            raise ValueError("Forecast exceeds the output size limit")
        destination = output_directory / f"{source.stem}.out"
        temporary = output_directory / f".{source.stem}.out.tmp"
        temporary.write_bytes(encoded)
        temporary.replace(destination)


if __name__ == "__main__":
    run()
