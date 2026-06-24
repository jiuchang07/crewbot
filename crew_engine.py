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

# Action space: 0..39 play card c; 40..79 communicate card c; 80 = pass.
# Communication happens only in a one-round setup phase at the start of the
# game (before any trick): each player, in turn order from the leader, either
# communicates one eligible card or passes. No mid-trick / per-trick comm.
COMM_OFFSET = N_CARDS                              # comm action id = COMM_OFFSET + card
PASS_ACTION = 2 * N_CARDS                          # decline to communicate (80)
# A game = N_PLAYERS comm-phase decisions + TOTAL_TRICKS*N_PLAYERS plays. +1 cushion.
MAX_PLIES = N_PLAYERS + TOTAL_TRICKS * N_PLAYERS + 1

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "crewbot")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# Real campaign missions (from The Crew mission log). Each entry:
#   tasks  : number of task cards
#   tiles  : priority-tile labels applied to the first len(tiles) revealed tasks,
#            in reveal order. Numbered '1'..'5' (absolute position) and relative
#            '>','>>','>>>','>>>>','W' (W = Omega, must be completed last).
#   mode   : task distribution mode — 'standard' (pick from pool), 'group'
#            (commander reveals one-by-one + yes/no + assign), 'solo' (one player
#            does all). Build 1 trains all as 'standard'; mode kept for later.
#   note   : deferred riders (comm variants / special objectives) not yet modeled.
# Special-objective-only missions (no tasks) are out of scope for now: in_scope=False.
_M = {
    1:  dict(tasks=1,  tiles=[]),
    2:  dict(tasks=2,  tiles=[]),
    3:  dict(tasks=2,  tiles=['1', '2']),
    4:  dict(tasks=3,  tiles=[]),
    6:  dict(tasks=3,  tiles=['>', '>>'], note='comm interference'),
    7:  dict(tasks=3,  tiles=['W']),
    8:  dict(tasks=3,  tiles=['1', '2', '3']),
    10: dict(tasks=4,  tiles=[]),
    11: dict(tasks=4,  tiles=['1'], note='cmdr mutes a crew'),
    12: dict(tasks=4,  tiles=['W'], note='post-trick1 random card pass'),
    14: dict(tasks=4,  tiles=['>', '>>', '>>>'], note='comm interference'),
    15: dict(tasks=4,  tiles=['1', '2', '3', '4']),
    17: dict(tasks=2,  tiles=[], note='no 9 wins a trick'),
    18: dict(tasks=5,  tiles=[], note='comm disrupted till trick 2'),
    19: dict(tasks=5,  tiles=['1'], note='comm disrupted till trick 3'),
    20: dict(tasks=2,  tiles=[], mode='solo'),
    21: dict(tasks=5,  tiles=['1', '2'], note='comm interference'),
    22: dict(tasks=5,  tiles=['>', '>>', '>>>', '>>>>']),
    23: dict(tasks=5,  tiles=['1', '2', '3', '4', '5'], note='pre-pick tile swap'),
    24: dict(tasks=6,  tiles=[], mode='group'),
    25: dict(tasks=6,  tiles=['>', '>>'], note='comm interference'),
    28: dict(tasks=6,  tiles=['1', 'W'], note='comm disrupted till trick 3'),
    30: dict(tasks=6,  tiles=['>', '>>', '>>>'], note='comm disrupted till trick 2'),
    31: dict(tasks=6,  tiles=['1', '2', '3']),
    32: dict(tasks=7,  tiles=[], mode='group'),
    35: dict(tasks=7,  tiles=['>', '>>', '>>>']),
    36: dict(tasks=7,  tiles=['1', '2'], mode='group'),
    37: dict(tasks=4,  tiles=[], mode='solo'),
    38: dict(tasks=8,  tiles=[], note='comm disrupted till trick 3'),
    39: dict(tasks=8,  tiles=['>', '>>', '>>>'], note='comm interference'),
    40: dict(tasks=8,  tiles=['1', '2', '3'], note='pre-pick tile move'),
    42: dict(tasks=9,  tiles=[]),
    43: dict(tasks=9,  tiles=[], mode='group'),
    45: dict(tasks=9,  tiles=['>', '>>', '>>>']),
    47: dict(tasks=10, tiles=[]),
    48: dict(tasks=3,  tiles=['W'], note='Omega on final trick'),
    49: dict(tasks=10, tiles=['>', '>>', '>>>']),
}
# Special-objective-only missions (deferred): 5,9,13,16,26,29,33,34,41,44,46,50
SPECIAL_ONLY = [5, 9, 13, 16, 26, 29, 33, 34, 41, 44, 46, 50]

def _build_missions():
    table = {}
    for m in range(1, 51):
        spec = _M.get(m)
        if spec is None:
            table[m] = dict(tasks=0, tiles=[], mode='standard', in_scope=False, note='special')
        else:
            table[m] = dict(tasks=spec['tasks'], tiles=spec.get('tiles', []),
                            mode=spec.get('mode', 'standard'), in_scope=True,
                            note=spec.get('note', ''))
    return table

MISSIONS = _build_missions()
ALL_MISSIONS = list(range(1, 51))
TASK_MISSIONS = [m for m in ALL_MISSIONS if MISSIONS[m]['in_scope']]  # trainable set

_REL_RANK = {'>': 1, '>>': 2, '>>>': 3, '>>>>': 4}

def tiles_to_predecessors(tiles, num_tasks):
    """Given tile labels on the first len(tiles) task slots (reveal order), return
    preds[slot] = set of slots that must be completed before it. Encodes the real
    rules: numbered tiles are absolute positions (and untiled tasks come after all
    numbered); relative tiles form a chain; Omega ('W') must be completed last."""
    labels = list(tiles) + [None] * (num_tasks - len(tiles))
    numbered = {i: int(l) for i, l in enumerate(labels) if l and l.isdigit()}
    relative = {i: _REL_RANK[l] for i, l in enumerate(labels) if l in _REL_RANK}
    omega = [i for i, l in enumerate(labels) if l == 'W']
    has_numbered = len(numbered) > 0
    preds = [set() for _ in range(num_tasks)]
    for i in range(num_tasks):
        if i in numbered:
            preds[i] = {j for j, n in numbered.items() if n < numbered[i]}
        elif i in relative:
            preds[i] = {j for j, r in relative.items() if r < relative[i]}
        elif i in omega:
            pass                                   # filled below
        elif has_numbered:                          # untiled, numbered mission
            preds[i] = set(numbered.keys())         # after all numbered tasks
        # else untiled in a relative/plain mission: no predecessors (anytime)
    for i in omega:                                 # Omega: after every other task
        preds[i] = set(range(num_tasks)) - {i}
    return preds

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
    card_preds: dict = field(default_factory=dict)  # card -> set of cards that must precede it
    tiles: dict = field(default_factory=dict)        # card -> tile label (display/info)

    @property
    def tasks(self):
        return list(self.assign.keys())

def _slot_to_card_constraints(task_cards, tiles_spec):
    """Map slot-level tiles to per-card predecessors + tile labels. task_cards are
    in reveal order; tiles_spec applies to the first len(tiles_spec) of them."""
    n = len(task_cards)
    slot_preds = tiles_to_predecessors(tiles_spec, n)
    card_preds = {int(task_cards[i]): {int(task_cards[j]) for j in slot_preds[i]}
                  for i in range(n)}
    tiles = {int(task_cards[i]): tiles_spec[i] for i in range(min(len(tiles_spec), n))}
    return card_preds, tiles

def sample_mission(rng, mission_id):
    """Draw a concrete mission instance (random task cards + assignment)."""
    spec = MISSIONS[mission_id]
    num_tasks = min(spec['tasks'], len(NON_TRUMP_CARDS))
    task_cards = [int(c) for c in rng.choice(NON_TRUMP_CARDS, size=num_tasks, replace=False)]
    assign = {c: int(rng.integers(N_PLAYERS)) for c in task_cards}
    card_preds, tiles = _slot_to_card_constraints(task_cards, spec['tiles'])
    return Mission(mission_id=mission_id, assign=assign, card_preds=card_preds, tiles=tiles)

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
# Perfect-information cooperative solver: given all hands, is there a joint line
# of play that completes every task (respecting assignment + ordering)? Used to
# build a *solvable* mission distribution so win_rate measures skill on winnable
# deals rather than being dragged down by impossible random assignments.
# Backtracking DFS controlling all players, with a node cap (cap reached => treat
# as unsolvable, which conservatively biases toward clearly-winnable missions).
# ---------------------------------------------------------------------------

def is_solvable(hands, assigned, card_preds, node_cap=4000):
    tasks = [c for c in range(N_CARDS) if assigned[c] != -1]
    n_tasks = len(tasks)
    if n_tasks == 0:
        return True
    hands = [set(h) for h in hands]
    leader0 = next(p for p in range(N_PLAYERS) if 39 in hands[p])
    done = set()
    nodes = [0]

    def rec(on_table, led, turn, tricks_played):
        if len(done) == n_tasks:
            return True
        if tricks_played >= TOTAL_TRICKS:
            return False
        nodes[0] += 1
        if nodes[0] > node_cap:
            return False
        hand = hands[turn]
        if not on_table:
            legal = sorted(hand)
        else:
            follow = [c for c in hand if card_color(c) == led]
            legal = sorted(follow) if follow else sorted(hand)
        for a in legal:
            hand.discard(a)
            new_led = card_color(a) if not on_table else led
            on_table.append((turn, a))
            if len(on_table) < N_PLAYERS:
                if rec(on_table, new_led, (turn + 1) % N_PLAYERS, tricks_played):
                    hand.add(a); on_table.pop(); return True
            else:
                winner = trick_winner(on_table, new_led)
                ok, added = True, []
                for _, c in on_table:                  # play order, matches step()
                    if assigned[c] != -1:
                        if winner != assigned[c]:
                            ok = False; break
                        if not all(d in done for d in card_preds.get(c, ())):
                            ok = False; break
                        done.add(c); added.append(c)
                if ok and rec([], -1, winner, tricks_played + 1):
                    for c in added: done.discard(c)
                    hand.add(a); on_table.pop(); return True
                for c in added: done.discard(c)
            hand.add(a); on_table.pop()
        return False

    return rec([], -1, leader0, 0)

# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    hands: list                     # list[set[int]] per player
    mission: Mission
    assigned: np.ndarray            # [40] player assigned, or -1
    card_preds: dict                # card -> set of task cards that must precede it
    done_tasks: set                 # completed task cards
    captured_by: np.ndarray         # [40] player who captured card, or -1
    on_table: list                  # list[(player, card)] current trick
    led_color: int                  # -1 if no card led yet
    leader: int                     # player who leads current trick
    turn: int                       # current player to act
    comm: list                      # per player: (card, type, valid); type 0=only,1=high,2=low
    comm_phase: bool                # True during the start-of-game communication round
    comm_count: int                 # players who have made their comm-phase decision
    tricks_played: int
    done: bool = False
    success: bool = False
    failed: bool = False

# ---------------------------------------------------------------------------
# Communication (real rule): each player may reveal one card that is their only
# / highest / lowest of its color (no trumps), during a setup round before play.
# The policy chooses WHICH card (or to pass); the signal type below is derived
# from the hand. type: 0=only, 1=highest, 2=lowest
# ---------------------------------------------------------------------------

def communicable(hand):
    """Return {card: type} for every card the player may legally reveal now."""
    out = {}
    for col in range(N_COLORS):
        cards = sorted((c for c in hand if card_color(c) == col), key=card_rank)
        if not cards:
            continue
        if len(cards) == 1:
            out[cards[0]] = 0          # only card of its color
        else:
            out[cards[-1]] = 1         # highest of its color
            out[cards[0]] = 2          # lowest of its color
    return out

# ---------------------------------------------------------------------------
# Game lifecycle
# ---------------------------------------------------------------------------

def _deal(rng):
    deck = list(range(N_CARDS))
    rng.shuffle(deck)
    sizes = [13, 13, 13]            # 14/13/13, extra card to a random player
    sizes[rng.integers(N_PLAYERS)] += 1
    hands, idx = [], 0
    for sz in sizes:
        hands.append(set(deck[idx:idx + sz]))
        idx += sz
    return hands

def _mission_arrays(mission):
    assigned = np.full(N_CARDS, -1, dtype=np.int64)
    for c, p in mission.assign.items():
        assigned[c] = p
    card_preds = {int(c): set(int(d) for d in preds)
                  for c, preds in mission.card_preds.items()}
    return assigned, card_preds

def _playout_captures(rng, hands):
    """Random-legal cooperative play-out. Returns captures as (trick_idx, winner,
    card) in completion order (within a trick: in play order)."""
    hands = [set(h) for h in hands]
    turn = next(p for p in range(N_PLAYERS) if 39 in hands[p])
    on_table, led, captures, trick_idx = [], -1, [], 0
    for _ in range(TOTAL_TRICKS * N_PLAYERS):       # 39 plays
        hand = hands[turn]
        if not on_table:
            legal = sorted(hand)
        else:
            follow = [c for c in hand if card_color(c) == led]
            legal = sorted(follow) if follow else sorted(hand)
        a = int(legal[rng.integers(len(legal))])
        hand.discard(a)
        if not on_table:
            led = card_color(a)
        on_table.append((turn, a))
        if len(on_table) < N_PLAYERS:
            turn = (turn + 1) % N_PLAYERS
        else:
            w = trick_winner(on_table, led)
            for _, c in on_table:
                captures.append((trick_idx, w, c))
            on_table, led, turn, trick_idx = [], -1, w, trick_idx + 1
    return captures

def construct_solvable_mission(rng, hands, mission_id):
    """Build a guaranteed-winnable mission: play a cooperative line, then assign
    chosen captured (non-trump) cards to whoever captured them. Tiles are applied
    in the line's completion order, so the recorded line satisfies every ordering
    constraint -> the mission is solvable by construction."""
    spec = MISSIONS[mission_id]
    caps = _playout_captures(rng, hands)
    cand = [(ti, w, c) for (ti, w, c) in caps if not is_trump(c)]
    num_tasks = min(spec['tasks'], len(cand))
    chosen = sorted(rng.choice(len(cand), size=num_tasks, replace=False))
    sel = [cand[i] for i in chosen]                 # preserves (trick, play-order)
    # reveal order = completion order, so tile[i] lands on the i-th-completed task
    task_cards = [int(c) for (ti, w, c) in sel]
    assign = {int(c): int(w) for (ti, w, c) in sel}
    card_preds, tiles = _slot_to_card_constraints(task_cards, spec['tiles'])
    return Mission(mission_id=mission_id, assign=assign, card_preds=card_preds, tiles=tiles)

def new_game(rng, mission_id, use_comm=True, solvable_only=False, max_redeal=40):
    hands = _deal(rng)
    if solvable_only:
        # Construct a winnable mission from a cooperative play-out (cheap, no
        # search) so win_rate measures skill, not luck of a (often impossible)
        # random task assignment.
        mission = construct_solvable_mission(rng, hands, mission_id)
    else:
        mission = sample_mission(rng, mission_id)
    assigned, card_preds = _mission_arrays(mission)

    # Communication starts empty; filled during the setup round (if enabled).
    comm = [(-1, -1, 0) for _ in range(N_PLAYERS)]

    # Commander = holder of the highest trump (rocket 4) leads the first trick
    # and decides first in the communication round.
    leader = next(p for p in range(N_PLAYERS) if 39 in hands[p])

    return GameState(
        hands=hands, mission=mission, assigned=assigned, card_preds=card_preds,
        done_tasks=set(), captured_by=np.full(N_CARDS, -1, dtype=np.int64),
        on_table=[], led_color=-1, leader=leader, turn=leader, comm=comm,
        comm_phase=bool(use_comm), comm_count=0, tricks_played=0,
    )

def legal_actions(s):
    # Setup communication round: pass, or reveal one eligible card.
    if s.comm_phase:
        return [PASS_ACTION] + [COMM_OFFSET + c for c in sorted(communicable(s.hands[s.turn]))]
    # Trick play.
    hand = s.hands[s.turn]
    if s.led_color == -1:
        return sorted(hand)
    follow = [c for c in hand if card_color(c) == s.led_color]
    return sorted(follow) if follow else sorted(hand)

def step(s, action):
    """Apply `action`: during the comm phase a pass (80) or communicate (40..79);
    otherwise a play (0..39)."""
    assert not s.done
    p = s.turn

    # --- Communication round: each player decides once, in order from leader.
    if s.comm_phase:
        if action != PASS_ACTION:
            c = action - COMM_OFFSET
            types = communicable(s.hands[p])
            assert c in types, f"illegal communication {card_name(c)}"
            s.comm[p] = (int(c), int(types[c]), 1)
        s.comm_count += 1
        if s.comm_count >= N_PLAYERS:
            s.comm_phase = False
            s.turn = s.leader
        else:
            s.turn = (s.leader + s.comm_count) % N_PLAYERS
        return s

    # --- Play action.
    assert action in s.hands[s.turn], f"illegal action {card_name(action)}"
    s.hands[p].discard(action)
    if s.comm[p][0] == action:
        s.comm[p] = (-1, -1, 0)   # signal consumed once its card is played (#3)
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
                # every predecessor task (priority tiles) must already be done
                if not all(d in s.done_tasks for d in s.card_preds.get(c, ())):
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

    # 9. blocked-ness (fraction of a task's predecessors not yet done) and
    # 10. "ready" (open task with all predecessors complete).
    blocked = np.zeros(N_CARDS, dtype=np.float32)
    ready = np.zeros(N_CARDS, dtype=np.float32)
    for c in range(N_CARDS):
        if s.assigned[c] == -1 or c in s.done_tasks:
            continue
        preds = s.card_preds.get(c, ())
        unmet = [d for d in preds if d not in s.done_tasks]
        if preds:
            blocked[c] = len(unmet) / len(preds)
        if not unmet:
            ready[c] = 1.0
    blocks.extend([blocked, ready])

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

    # 13. scalars (incl. communication-phase flag so the policy knows the phase)
    blocks.append(np.array([
        s.tricks_played / TOTAL_TRICKS,
        1.0 if s.turn == s.leader else 0.0,
        len(s.on_table) / N_PLAYERS,
        1.0 if s.comm_phase else 0.0,
    ], dtype=np.float32))

    return np.concatenate(blocks)

def legal_mask(s):
    m = np.zeros(ACT_DIM, dtype=np.float32)
    for a in legal_actions(s):
        m[a] = 1.0
    return m

# Action space: 40 play + 40 communicate + 1 pass.
ACT_DIM = 2 * N_CARDS + 1

# Compute OBS_DIM once from a sample game.
OBS_DIM = observe(new_game(np.random.default_rng(0), 1)).shape[0]

# ---------------------------------------------------------------------------
# Heuristic policy (for bootstrapping self-play data; no neural net needed)
# ---------------------------------------------------------------------------

def _open_task(s, c):
    return s.assigned[c] != -1 and c not in s.done_tasks

def heuristic_action(s, rng, epsilon=0.1):
    """Cooperative heuristic: deliver task cards to their assignee, otherwise
    duck. Far from optimal, but gives self-play data real signal. Never
    communicates: passes in the comm phase and plays cards otherwise."""
    if s.comm_phase:
        return PASS_ACTION
    legal = [a for a in legal_actions(s) if a < COMM_OFFSET]
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

def play_one(action_fn, rng, mission_id, use_comm=True, max_tricks=60, solvable_only=False):
    """Play a full mission. action_fn(state)->card. Returns (success, n_steps)."""
    s = new_game(rng, mission_id, use_comm=use_comm, solvable_only=solvable_only)
    steps = 0
    while not s.done and steps < max_tricks * N_PLAYERS:
        a = action_fn(s)
        legal = legal_actions(s)
        if a not in legal:  # safety: fall back to a legal move
            a = legal[0]
        step(s, a)
        steps += 1
    return s.success, steps

def evaluate_winrate(action_fn, missions=None, games_per_mission=200, seed=12345,
                     use_comm=True, solvable_only=False):
    """Average mission success rate, the metric to maximize. Higher is better.
    With solvable_only, missions are rejection-sampled to be winnable, so the
    metric measures skill rather than luck of the deal.

    Returns (overall_win_rate, {mission_id: win_rate}).
    """
    missions = missions if missions is not None else TASK_MISSIONS
    rng = np.random.default_rng(seed)
    per = {}
    wins_total = games_total = 0
    for m in missions:
        wins = 0
        for _ in range(games_per_mission):
            ok, _ = play_one(action_fn, rng, m, use_comm=use_comm, solvable_only=solvable_only)
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
    print(f"task missions: {len(TASK_MISSIONS)}  (special-only deferred: {len(SPECIAL_ONLY)})")
    sample = [1, 3, 7, 8, 15, 22, 31, 42, 49]   # spans plain/numbered/relative/Omega tiles
    for name, fn in [("random", rand), ("heuristic", heur)]:
        for so in (False, True):
            wr, per = evaluate_winrate(fn, missions=sample, games_per_mission=200,
                                       solvable_only=so)
            tag = "solvable" if so else "random  "
            print(f"{name:10s} {tag} overall={wr:.3f}  " +
                  "  ".join(f"m{m}={per[m]:.2f}" for m in sorted(per)))
