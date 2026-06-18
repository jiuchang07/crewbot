"""
The Crew: The Quest for Planet Nine — rules engine + fixed evaluation metric.

This file is the crewbot analog of autoresearch's `prepare.py`: it holds the
fixed game rules, the observation encoding, and the ground-truth evaluation
metric (`evaluate_winrate`). It has NO torch dependency and is deliberately
kept FIXED — the agent does not edit this file. `train.py` imports from here.

Design notes
------------
- Fixed at 3 players (per project scope).
- 40-card deck: 4 colors x ranks 1..9 (36 cards) + 4 trump "rockets" 1..4.
- A *mission* is a general constraint spec (task cards + assignment + optional
  completion ordering). This single representation covers the campaign's 50
  missions as a difficulty ladder, and a single model conditioned on the
  mission encoding generalizes across all of them and all hand deals.
- The metric is `win_rate` (fraction of missions completed). Higher is better.

Card indexing (canonical, 0..39):
    color c in {0,1,2,3}, rank r in {1..9}  -> index = c*9 + (r-1)   (0..35)
    trump rank r in {1..4}                  -> index = 36 + (r-1)     (36..39)
"""

import os
import math
import numpy as np
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

N_PLAYERS = 3
N_COLORS = 4
RANKS_PER_COLOR = 9
N_TRUMP = 4
N_CARDS = N_COLORS * RANKS_PER_COLOR + N_TRUMP   # 40
TRUMP_COLOR = N_COLORS                            # pseudo-color id (4) for trump
CARDS_PER_PLAYER_MAX = (N_CARDS + N_PLAYERS - 1) // N_PLAYERS  # 14
# Official 3-player rule: 40 cards dealt 14/13/13; the player with the extra
# card never plays one card, so there are exactly 13 full 3-card tricks and one
# leftover (unplayed) card. A task card stranded as the leftover is never
# captured -> that mission fails, which is correct game behavior.
TOTAL_TRICKS = N_CARDS // N_PLAYERS               # 13

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "crewbot")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# Difficulty ladder: the 50 campaign missions, approximated as an increasing
# number of task cards plus (for harder missions) a required completion order.
# Task cards themselves are drawn randomly each game, so this generalizes to
# "any hand combination" automatically. (num_tasks, ordered)
def _mission_table():
    table = {}
    for m in range(1, 51):
        # tasks grow ~linearly with mission number, capped at 10
        num_tasks = min(1 + (m - 1) // 5, 10)
        if num_tasks < 1:
            num_tasks = 1
        # the back half of the campaign adds ordered-completion constraints
        ordered = m >= 26
        table[m] = (num_tasks, ordered)
    return table

MISSIONS = _mission_table()
ALL_MISSIONS = list(range(1, 51))

# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

def card_color(c):
    return TRUMP_COLOR if c >= 36 else c // RANKS_PER_COLOR

def card_rank(c):
    return (c - 36 + 1) if c >= 36 else (c % RANKS_PER_COLOR + 1)

def is_trump(c):
    return c >= 36

def card_name(c):
    if is_trump(c):
        return f"R{card_rank(c)}"
    return f"{'PBGY'[card_color(c)]}{card_rank(c)}"

NON_TRUMP_CARDS = [c for c in range(N_CARDS) if not is_trump(c)]

# ---------------------------------------------------------------------------
# Mission specification
# ---------------------------------------------------------------------------

@dataclass
class Mission:
    mission_id: int                 # 1..50
    assign: dict                    # card_index -> player who must capture it
    order: list = field(default_factory=list)  # cards in required completion order ([] = unordered)

    @property
    def tasks(self):
        return list(self.assign.keys())

def sample_mission(rng, mission_id):
    """Draw a concrete mission instance (random task cards + assignment)."""
    num_tasks, ordered = MISSIONS[mission_id]
    num_tasks = min(num_tasks, len(NON_TRUMP_CARDS))
    task_cards = list(rng.choice(NON_TRUMP_CARDS, size=num_tasks, replace=False))
    assign = {int(c): int(rng.integers(N_PLAYERS)) for c in task_cards}
    order = []
    if ordered and num_tasks >= 2:
        order = list(task_cards)
        rng.shuffle(order)
        order = [int(c) for c in order]
    return Mission(mission_id=mission_id, assign=assign, order=order)

# ---------------------------------------------------------------------------
# Trick resolution
# ---------------------------------------------------------------------------

def card_beats(c, w, led_color):
    """Does card c beat the current winning card w, given the led color?"""
    if w is None:
        return True
    ct, wt = is_trump(c), is_trump(w)
    if ct and not wt:
        return True
    if ct and wt:
        return card_rank(c) > card_rank(w)
    if (not ct) and wt:
        return False
    # both non-trump: c can only win if it follows the led color
    if card_color(c) != led_color:
        return False
    if card_color(w) != led_color:
        return True
    return card_rank(c) > card_rank(w)

def trick_winner(trick, led_color):
    """trick: list of (player, card). Returns winning player index."""
    best_p, best_c = trick[0]
    for p, c in trick[1:]:
        if card_beats(c, best_c, led_color):
            best_p, best_c = p, c
    return best_p

# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    hands: list                     # list[set[int]] per player
    mission: Mission
    assigned: np.ndarray            # [40] player assigned, or -1
    order_pos: np.ndarray           # [40] 1-based order position, or 0 if unordered/not-task
    done_tasks: set                 # completed task cards
    captured_by: np.ndarray         # [40] player who captured card, or -1
    on_table: list                  # list[(player, card)] current trick
    led_color: int                  # -1 if no card led yet
    leader: int                     # player who leads current trick
    turn: int                       # current player to act
    comm: list                      # per player: (card, type, valid); type 0=only,1=high,2=low
    tricks_played: int
    done: bool = False
    success: bool = False
    failed: bool = False

# ---------------------------------------------------------------------------
# Communication (simplified): each player reveals one legal non-trump card.
# Rule (real game): you may reveal a card that is your only / highest / lowest
# of its color. We auto-pick one such card per player via a fixed heuristic so
# the cooperative information is present in the observation.
# ---------------------------------------------------------------------------

def _auto_communication(hand, rng):
    by_color = {col: sorted([c for c in hand if card_color(c) == col], key=card_rank)
                for col in range(N_COLORS)}
    candidates = []  # (card, type) where type 0=only,1=high,2=low
    for col, cards in by_color.items():
        if not cards:
            continue
        if len(cards) == 1:
            candidates.append((cards[0], 0))
        else:
            candidates.append((cards[-1], 1))
            candidates.append((cards[0], 2))
    if not candidates:
        return (-1, -1, 0)  # only trumps in hand -> nothing to communicate
    # prefer revealing an "only" card (highest information), else random
    onlys = [c for c in candidates if c[1] == 0]
    pick = onlys[rng.integers(len(onlys))] if onlys else candidates[rng.integers(len(candidates))]
    return (int(pick[0]), int(pick[1]), 1)

# ---------------------------------------------------------------------------
# Game lifecycle
# ---------------------------------------------------------------------------

def new_game(rng, mission_id, use_comm=True):
    deck = list(range(N_CARDS))
    rng.shuffle(deck)
    # 3 players: 14/13/13, the extra card going to a random player
    sizes = [13, 13, 13]
    sizes[rng.integers(N_PLAYERS)] += 1
    hands, idx = [], 0
    for sz in sizes:
        hands.append(set(deck[idx:idx + sz]))
        idx += sz

    mission = sample_mission(rng, mission_id)
    assigned = np.full(N_CARDS, -1, dtype=np.int64)
    for c, p in mission.assign.items():
        assigned[c] = p
    order_pos = np.zeros(N_CARDS, dtype=np.int64)
    for i, c in enumerate(mission.order):
        order_pos[c] = i + 1

    comm = [_auto_communication(hands[p], rng) if use_comm else (-1, -1, 0)
            for p in range(N_PLAYERS)]

    # Commander = holder of the highest trump (rocket 4) leads the first trick.
    leader = next(p for p in range(N_PLAYERS) if 39 in hands[p])

    return GameState(
        hands=hands, mission=mission, assigned=assigned, order_pos=order_pos,
        done_tasks=set(), captured_by=np.full(N_CARDS, -1, dtype=np.int64),
        on_table=[], led_color=-1, leader=leader, turn=leader, comm=comm,
        tricks_played=0,
    )

def legal_actions(s):
    hand = s.hands[s.turn]
    if s.led_color == -1:
        return sorted(hand)
    follow = [c for c in hand if card_color(c) == s.led_color]
    return sorted(follow) if follow else sorted(hand)

def step(s, action):
    """Play `action` for the current player. Mutates and returns s."""
    assert not s.done
    assert action in s.hands[s.turn], f"illegal action {card_name(action)}"
    p = s.turn
    s.hands[p].discard(action)
    if not s.on_table:
        s.led_color = card_color(action)
    s.on_table.append((p, action))

    if len(s.on_table) < N_PLAYERS:
        s.turn = (s.turn + 1) % N_PLAYERS
        return s

    # Resolve completed trick
    winner = trick_winner(s.on_table, s.led_color)
    for _, c in s.on_table:
        s.captured_by[c] = winner
    # Evaluate task constraints for any task cards in this trick
    for _, c in s.on_table:
        if s.assigned[c] != -1:
            if winner != s.assigned[c]:
                s.failed = True
            else:
                op = s.order_pos[c]
                if op > 0:
                    # all earlier-order tasks must already be completed
                    earlier = [d for d in range(N_CARDS) if 0 < s.order_pos[d] < op]
                    if not all(d in s.done_tasks for d in earlier):
                        s.failed = True
                if not s.failed:
                    s.done_tasks.add(int(c))

    s.on_table = []
    s.led_color = -1
    s.leader = winner
    s.turn = winner
    s.tricks_played += 1

    n_tasks = len(s.mission.assign)
    if s.failed:
        s.done, s.success = True, False
    elif len(s.done_tasks) == n_tasks:
        s.done, s.success = True, True
    elif s.tricks_played >= TOTAL_TRICKS:
        s.done, s.success = True, (len(s.done_tasks) == n_tasks)
    return s

# ---------------------------------------------------------------------------
# Observation encoding (from the acting player's perspective)
# ---------------------------------------------------------------------------

def _onehot(idx, n):
    v = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v

def observe(s, player=None):
    """Encode the state from `player`'s perspective (default: current turn)."""
    p = s.turn if player is None else player
    blocks = []
    hand = s.hands[p]

    # 1. own hand
    b = np.zeros(N_CARDS, dtype=np.float32)
    for c in hand:
        b[c] = 1.0
    blocks.append(b)

    # 2. cards captured in completed tricks (out of play)
    blocks.append((s.captured_by >= 0).astype(np.float32))

    # 3. current trick on table + 4. current winning card
    table = np.zeros(N_CARDS, dtype=np.float32)
    win = np.zeros(N_CARDS, dtype=np.float32)
    if s.on_table:
        for _, c in s.on_table:
            table[c] = 1.0
        wc = s.on_table[0][1]
        for _, c in s.on_table[1:]:
            if card_beats(c, wc, s.led_color):
                wc = c
        win[wc] = 1.0
    blocks.append(table)
    blocks.append(win)

    # 5. led color (5 = 4 colors + trump)
    blocks.append(_onehot(s.led_color if s.led_color != -1 else N_COLORS + 1, N_COLORS + 1)
                  if s.led_color != -1 else np.zeros(N_COLORS + 1, dtype=np.float32))

    # 6-8. tasks relative to me / others / done
    mine = np.zeros(N_CARDS, dtype=np.float32)
    other = np.zeros(N_CARDS, dtype=np.float32)
    done = np.zeros(N_CARDS, dtype=np.float32)
    for c in range(N_CARDS):
        a = s.assigned[c]
        if a == -1:
            continue
        if c in s.done_tasks:
            done[c] = 1.0
        elif a == p:
            mine[c] = 1.0
        else:
            other[c] = 1.0
    blocks.extend([mine, other, done])

    # 9. order position (normalized) and 10. "ready" (all predecessors done)
    n_order = int(s.order_pos.max())
    order_norm = np.where(s.order_pos > 0, s.order_pos / max(n_order, 1), 0.0).astype(np.float32)
    ready = np.zeros(N_CARDS, dtype=np.float32)
    for c in range(N_CARDS):
        op = s.order_pos[c]
        if op > 0 and c not in s.done_tasks:
            earlier = [d for d in range(N_CARDS) if 0 < s.order_pos[d] < op]
            if all(d in s.done_tasks for d in earlier):
                ready[c] = 1.0
        elif s.assigned[c] != -1 and op == 0 and c not in s.done_tasks:
            ready[c] = 1.0
    blocks.extend([order_norm, ready])

    # 11. communication hints, relative seating (offset 0 = self)
    for off in range(N_PLAYERS):
        q = (p + off) % N_PLAYERS
        card, htype, valid = s.comm[q]
        blocks.append(_onehot(card, N_CARDS))
        blocks.append(_onehot(htype, 3))
        blocks.append(np.array([float(valid)], dtype=np.float32))

    # 12. relative hand sizes
    blocks.append(np.array([len(s.hands[(p + off) % N_PLAYERS]) / CARDS_PER_PLAYER_MAX
                            for off in range(N_PLAYERS)], dtype=np.float32))

    # 13. scalars
    blocks.append(np.array([
        s.tricks_played / TOTAL_TRICKS,
        1.0 if s.turn == s.leader else 0.0,
        len(s.on_table) / N_PLAYERS,
    ], dtype=np.float32))

    return np.concatenate(blocks)

def legal_mask(s):
    m = np.zeros(N_CARDS, dtype=np.float32)
    for c in legal_actions(s):
        m[c] = 1.0
    return m

# Compute OBS_DIM once from a sample game.
OBS_DIM = observe(new_game(np.random.default_rng(0), 1)).shape[0]
ACT_DIM = N_CARDS

# ---------------------------------------------------------------------------
# Heuristic policy (for bootstrapping self-play data; no neural net needed)
# ---------------------------------------------------------------------------

def _open_task(s, c):
    return s.assigned[c] != -1 and c not in s.done_tasks

def heuristic_action(s, rng, epsilon=0.1):
    """Cooperative heuristic: deliver task cards to their assignee, otherwise
    duck. Far from optimal, but gives self-play data real signal."""
    legal = legal_actions(s)
    if rng.random() < epsilon:
        return int(legal[rng.integers(len(legal))])

    # current winning player/card on the table (if any)
    win_p, wc = (None, None)
    if s.on_table:
        win_p, wc = s.on_table[0]
        for p, c in s.on_table[1:]:
            if card_beats(c, wc, s.led_color):
                win_p, wc = p, c

    if s.led_color == -1:
        # Leading. If I hold one of my own task cards that is strong enough to
        # likely win (highest unseen of its color, or a high trump), lead it.
        my_tasks = [c for c in legal if _open_task(s, c) and s.assigned[c] == s.turn]
        strong = [c for c in my_tasks if _is_boss(s, c)]
        if strong:
            return int(max(strong, key=card_rank))
        # otherwise lead a low, non-task, non-trump card to keep flexibility
        return int(min(legal, key=lambda c: (_open_task(s, c), is_trump(c), card_rank(c))))

    # Following a trick.
    want_win = any(_open_task(s, c) and s.assigned[c] == s.turn for _, c in s.on_table)
    if want_win:
        winners = [c for c in legal if card_beats(c, wc, s.led_color)]
        if winners:
            return int(min(winners, key=card_rank))   # win my task cheaply
        return int(min(legal, key=card_rank))          # can't win; lose cheaply

    # Not my task in the trick. If the current winner is the assignee of one of
    # my legal task cards, feed them that card (as long as it won't steal the trick).
    feed = [c for c in legal if _open_task(s, c) and s.assigned[c] == win_p
            and not card_beats(c, wc, s.led_color)]
    if feed:
        return int(max(feed, key=card_rank))

    # Otherwise duck: dump the highest card that does NOT take the trick and is
    # not an open task card (avoid mis-delivering someone's task).
    safe = [c for c in legal if not card_beats(c, wc, s.led_color)]
    safe_nontask = [c for c in safe if not _open_task(s, c)] or safe
    if safe_nontask:
        return int(max(safe_nontask, key=card_rank))
    # forced to win: do it as cheaply as possible
    return int(min(legal, key=card_rank))

def _is_boss(s, c):
    """True if c is currently the strongest live card of its group (color/trump):
    no higher card of the same color (or any higher trump) remains unplayed."""
    seen = s.captured_by >= 0
    if is_trump(c):
        return not any((not seen[d]) and is_trump(d) and card_rank(d) > card_rank(c)
                       for d in range(36, N_CARDS))
    col = card_color(c)
    return not any((not seen[d]) and card_color(d) == col and card_rank(d) > card_rank(c)
                   for d in range(N_CARDS))

# ---------------------------------------------------------------------------
# Evaluation metric (FIXED — this is the ground-truth win_rate)
# ---------------------------------------------------------------------------

def play_one(action_fn, rng, mission_id, use_comm=True, max_tricks=60):
    """Play a full mission. action_fn(state)->card. Returns (success, n_steps)."""
    s = new_game(rng, mission_id, use_comm=use_comm)
    steps = 0
    while not s.done and steps < max_tricks * N_PLAYERS:
        a = action_fn(s)
        legal = legal_actions(s)
        if a not in legal:  # safety: fall back to a legal move
            a = legal[0]
        step(s, a)
        steps += 1
    return s.success, steps

def evaluate_winrate(action_fn, missions=None, games_per_mission=200, seed=12345, use_comm=True):
    """Average mission success rate, the metric to maximize. Higher is better.

    Returns (overall_win_rate, {mission_id: win_rate}).
    """
    missions = missions if missions is not None else ALL_MISSIONS
    rng = np.random.default_rng(seed)
    per = {}
    wins_total = games_total = 0
    for m in missions:
        wins = 0
        for _ in range(games_per_mission):
            ok, _ = play_one(action_fn, rng, m, use_comm=use_comm)
            wins += int(ok)
        per[m] = wins / games_per_mission
        wins_total += wins
        games_total += games_per_mission
    return wins_total / games_total, per


if __name__ == "__main__":
    # Quick self-test: heuristic vs random across the ladder.
    rng = np.random.default_rng(0)
    print(f"OBS_DIM={OBS_DIM} ACT_DIM={ACT_DIM}")
    heur = lambda s: heuristic_action(s, rng, epsilon=0.0)
    rand = lambda s: int(legal_actions(s)[rng.integers(len(legal_actions(s)))])
    for name, fn in [("random", rand), ("heuristic", heur)]:
        wr, per = evaluate_winrate(fn, missions=[1, 5, 10, 20, 30, 40, 50],
                                   games_per_mission=300)
        print(f"{name:10s} overall={wr:.3f}  " +
              "  ".join(f"m{m}={per[m]:.2f}" for m in sorted(per)))
