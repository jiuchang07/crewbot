"""
play.py — play The Crew *with* the trained bot and learn the optimal moves.

You take one seat; the bot (the PPO policy from ppo.py / train.py) plays the
other two. On every one of YOUR decisions the tool shows, for each legal move:

  - the bot's instinct  : the policy's probability of choosing that move, and
  - win %               : an honest estimate of the team's win probability if
                          you make that move and everyone (you included) plays
                          the bot's policy from there on — measured by Monte-Carlo
                          rollouts of the policy in this exact deal.

The move with the highest win % is starred as the bot's recommendation. After
you move it coaches you when you diverge, and at the end it recaps where your
choices cost (or gained) win probability. Communication (the start-of-game
signalling round) is taught the same way — signalling is a move like any other.

This file only READS the engine and a trained checkpoint; it never modifies
crew_engine.py and needs no training to run (a checkpoint must already exist,
e.g. ~/.cache/crewbot/ppo_model.pt from `python ppo.py`). Without a checkpoint
it falls back to the cooperative heuristic so you can still play.

Usage:
    python play.py                         # auto seat (commander), mission picker
    python play.py --mission 12 --seat 0   # specific mission / seat
    python play.py --rollouts 0            # instant play, no win% estimate
    python play.py --no-color              # plain text
    python play.py --ckpt path/to.pt       # a specific checkpoint
"""

import os
import sys
import copy
import argparse
import numpy as np

# Windows consoles default to cp1252 and choke on box-drawing / emoji glyphs.
# Force UTF-8 (errors="replace" guarantees we never crash on an odd character).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                       # pragma: no cover
    pass

import crew_engine as E

# Checkpoint search order: repo dir (git-tracked copy), then the local cache.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Torch is only needed for the neural bot; degrade gracefully to the heuristic.
try:
    import torch
    from train import PolicyValueNet
    _HAVE_TORCH = True
except Exception:                                       # pragma: no cover
    _HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Pretty-printing (cards, suits, the board)
# ---------------------------------------------------------------------------

_USE_COLOR = True
# Suit order in card_name() is 'PBGY' (pink, blue, green, yellow); trump = rocket.
_SUIT_ANSI = {0: "95", 1: "94", 2: "92", 3: "93", E.TRUMP_COLOR: "1;91"}
_SUIT_FULL = {0: "Pink", 1: "Blue", 2: "Green", 3: "Yellow", E.TRUMP_COLOR: "Rocket"}


def _c(text, ansi):
    if not _USE_COLOR or ansi is None:
        return text
    return f"\033[{ansi}m{text}\033[0m"


def card_str(c):
    """Coloured card label, e.g. a green 7 or a red rocket."""
    return _c(E.card_name(c), _SUIT_ANSI[E.card_color(c)])


def action_label(s, a):
    """Human-readable label for any action id in the current state."""
    if a >= E.CLAIM_OFFSET:
        c = a - E.CLAIM_OFFSET
        tile = s.mission.tiles.get(c)
        return f"Claim task {card_str(c)}" + (f"  [tile {tile}]" if tile else "")
    if a == E.PASS_ACTION:
        return "Pass — reveal nothing"
    if a >= E.COMM_OFFSET:
        c = a - E.COMM_OFFSET
        htype = E.communicable(s.hands[s.turn]).get(c, 0)
        kind = {0: "my only", 1: "my highest", 2: "my lowest"}[htype]
        col = _SUIT_FULL[E.card_color(c)]
        return f"Signal {card_str(c)}  ({kind} {col})"
    return f"Play {card_str(c := a)}"


def hand_str(hand):
    """Hand grouped by suit, sorted, trumps last."""
    parts = []
    for col in list(range(E.N_COLORS)) + [E.TRUMP_COLOR]:
        cards = sorted((c for c in hand if E.card_color(c) == col), key=E.card_rank)
        if cards:
            parts.append(" ".join(card_str(c) for c in cards))
    return "   ".join(parts) if parts else "(empty)"


def seat_label(s, seat, human_seat):
    you = " (you)" if seat == human_seat else ""
    lead = "  ◄lead" if seat == s.leader else ""
    return f"P{seat}{you}{lead}"


# ---------------------------------------------------------------------------
# The bot: load the policy, batched inference, Monte-Carlo win% rollouts
# ---------------------------------------------------------------------------

class Bot:
    """Wraps the trained policy (or the heuristic) and exposes the two things
    the coach needs: a move distribution, and a win% estimate per candidate."""

    def __init__(self, ckpt=None, device="cpu"):
        self.model = None
        self.device = device
        self.rng = np.random.default_rng()
        if _HAVE_TORCH and ckpt and os.path.exists(ckpt):
            model = PolicyValueNet(bounded_value=False).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            self.model = model
            self.name = f"PPO policy  ({os.path.basename(ckpt)})"
        else:
            self.name = "cooperative heuristic (no checkpoint found)"

    # --- low-level batched policy over a list of states (each at its own turn)
    def _probs_batch(self, states, temp=1.0):
        obs = np.stack([E.observe(s) for s in states]).astype(np.float32)
        mask = np.stack([E.legal_mask(s) for s in states])
        with torch.no_grad():
            logits, _ = self.model(torch.from_numpy(obs).to(self.device))
            logits = logits / max(temp, 1e-6)
            logits = logits.masked_fill(torch.from_numpy(mask).to(self.device) == 0,
                                        float("-inf"))
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def move_distribution(self, s):
        """{action: probability} the bot would choose now (its instinct)."""
        legal = E.legal_actions(s)
        if self.model is None:
            return {a: 1.0 / len(legal) for a in legal}          # heuristic: uniform display
        p = self._probs_batch([s])[0]
        return {a: float(p[a]) for a in legal}

    def greedy_action(self, s):
        legal = E.legal_actions(s)
        if self.model is None:
            return E.heuristic_action(s, self.rng, epsilon=0.0)
        p = self._probs_batch([s])[0]
        return max(legal, key=lambda a: p[a])

    def _sample_batch_actions(self, states, temp):
        if self.model is None:
            return [E.heuristic_action(s, self.rng, epsilon=0.25) for s in states]
        probs = self._probs_batch(states, temp=temp)
        out = []
        for i, s in enumerate(states):
            legal = E.legal_actions(s)
            pl = np.array([probs[i][a] for a in legal], dtype=np.float64)
            pl = pl / pl.sum() if pl.sum() > 0 else None
            out.append(int(self.rng.choice(legal, p=pl)))
        return out

    def rollout_winprob(self, base, candidates, n=60, temp=0.6, max_plies=None):
        """Win probability of each candidate move from `base`, estimated by
        playing the deal out `n` times per candidate under the bot's policy.

        Returns {action: win_fraction}. This is the 'how good is this move'
        signal: it assumes the actual (hidden) cards and best-effort bot play,
        so it teaches the optimal continuation in *this* deal."""
        max_plies = max_plies or (E.MAX_PLIES + 2)
        pool, owner = [], []
        for ci, a in enumerate(candidates):
            for _ in range(n):
                s2 = copy.deepcopy(base)
                E.step(s2, a)
                pool.append(s2)
                owner.append(ci)
        # play every rollout out in lockstep, batching policy inference
        for _ in range(max_plies):
            live = [i for i, s in enumerate(pool) if not s.done]
            if not live:
                break
            acts = self._sample_batch_actions([pool[i] for i in live], temp)
            for i, a in zip(live, acts):
                if a not in E.legal_actions(pool[i]):
                    a = E.legal_actions(pool[i])[0]
                E.step(pool[i], a)
        wins = np.zeros(len(candidates))
        for i, s in enumerate(pool):
            wins[owner[i]] += int(s.success)
        return {candidates[ci]: wins[ci] / n for ci in range(len(candidates))}


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

def render_board(s, human_seat):
    print("\n" + "═" * 70)
    m = s.mission
    phase = "TASK DISTRIBUTION (draft — claim your tasks)" if s.dist_phase else \
            "COMMUNICATION (signalling round)" if s.comm_phase else \
            f"Trick {s.tricks_played + 1}/{E.TOTAL_TRICKS}"
    print(f"Mission {m.mission_id}    {phase}")

    # tasks (claimed during the distribution draft; -1 = still in the pool)
    print(_c("Tasks:", "1"))
    for c in sorted(s.task_cards, key=lambda c: (m.tiles.get(c, "~"), c)):
        who = int(s.assigned[c])
        if c in s.done_tasks:
            tag = "✔ done"
        elif who == -1:
            tag = _c("(unclaimed)", "90")
        elif who == human_seat:
            tag = _c("YOURS", "1;96")
        else:
            tag = f"P{who} must win"
        tile = m.tiles.get(c)
        order = f"  [tile {tile}]" if tile else ""
        ready = ""
        if c not in s.done_tasks and who != -1:
            unmet = [d for d in s.card_preds.get(c, ()) if d not in s.done_tasks]
            ready = _c("  (blocked)", "90") if unmet else ""
        print(f"   {card_str(c)}  → {tag}{order}{ready}")

    # communication signals revealed so far
    sig = []
    for seat in range(E.N_PLAYERS):
        card, htype, valid = s.comm[seat]
        if valid:
            kind = {0: "only", 1: "high", 2: "low"}[htype]
            sig.append(f"P{seat}{'(you)' if seat == human_seat else ''}: "
                       f"{card_str(card)}({kind})")
    if sig:
        print(_c("Signals: ", "1") + "   ".join(sig))

    # current trick
    if s.on_table:
        wc = s.on_table[0][1]
        for _, c in s.on_table[1:]:
            if E.card_beats(c, wc, s.led_color):
                wc = c
        led = _SUIT_FULL[s.led_color]
        cells = []
        for seat, c in s.on_table:
            mark = _c("★", "1;92") if c == wc else " "
            cells.append(f"P{seat}:{card_str(c)}{mark}")
        print(_c(f"Trick (led {led}): ", "1") + "   ".join(cells))
    elif not s.comm_phase and not s.dist_phase:
        print(_c("Trick: ", "1") + "(empty — leader to play)")

    print(_c("Your hand: ", "1;96") + hand_str(s.hands[human_seat]))
    print("═" * 70)


# ---------------------------------------------------------------------------
# A single human decision, with coaching
# ---------------------------------------------------------------------------

def analyze(s, bot, rollouts, show_progress=True):
    """Rank the legal moves: by rollout win% when available, else policy instinct.
    Returns (ordered_actions, best_action, instinct_dist, winprob_dict)."""
    legal = E.legal_actions(s)
    dist = bot.move_distribution(s)
    win = {}
    if rollouts > 0 and bot.model is not None:
        if show_progress:
            sys.stdout.write("  …estimating win% by rollout\r")
            sys.stdout.flush()
        win = bot.rollout_winprob(s, legal, n=rollouts)
        if show_progress:
            sys.stdout.write(" " * 44 + "\r")
    rank_key = (lambda a: (win.get(a, -1), dist.get(a, 0))) if win else (lambda a: dist.get(a, 0))
    ordered = sorted(legal, key=rank_key, reverse=True)
    return ordered, ordered[0], dist, win


def print_menu(s, ordered, best, dist, win):
    """Render the ranked move menu; return {menu_index: action}."""
    print(_c("\nYour move — bot analysis:", "1;97"))
    print(f"   {'#':>2}  {'move':<34} {'bot instinct':>12} {'win %':>8}")
    idx_map = {}
    for i, a in enumerate(ordered, 1):
        idx_map[i] = a
        star = _c(" ★ recommended", "1;92") if a == best else ""
        wtxt = f"{100*win[a]:6.0f}%" if a in win else "   —  "
        print(f"   {i:>2}  {action_label(s, a):<34} {100*dist.get(a,0):10.0f}%  {wtxt}{star}")
    return idx_map


def print_menu_blind(s):
    """Blind-mode menu: show legal moves in natural order, no bot data.
    Returns {menu_index: action} and the ordered list of actions."""
    legal = E.legal_actions(s)
    print(_c("\nYour move — choose without the bot's guidance:", "1;97"))
    idx_map = {}
    for i, a in enumerate(legal, 1):
        idx_map[i] = a
        print(f"   {i:>2}  {action_label(s, a)}")
    return idx_map, legal


def reveal_analysis(s, chosen, ordered, best, dist, win):
    """After blind choice, print the full bot analysis table plus coaching feedback."""
    print(_c("\n--- Bot analysis revealed ---", "1;97"))
    print(f"   {'#':>2}  {'move':<34} {'bot instinct':>12} {'win %':>8}")
    for i, a in enumerate(ordered, 1):
        star = _c(" ★ recommended", "1;92") if a == best else ""
        you  = _c(" ◄ your pick", "1;96") if a == chosen else ""
        wtxt = f"{100*win[a]:6.0f}%" if a in win else "   —  "
        print(f"   {i:>2}  {action_label(s, a):<34} {100*dist.get(a,0):10.0f}%  {wtxt}{star}{you}")
    print(_c("----------------------------", "1;97"))


def _step_and_log(s, a, human_seat):
    """Apply any action in-place; return human-readable log lines (incl. a
    resolved-trick note when the trick completes)."""
    lines = []
    pre = s.tricks_played
    if s.dist_phase:
        lines.append(f"P{s.turn} claims task {card_str(a - E.CLAIM_OFFSET)}.")
    elif s.comm_phase:
        lines.append("P%d passes (no signal)." % s.turn if a == E.PASS_ACTION
                     else f"P{s.turn} signals {card_str(a - E.COMM_OFFSET)}.")
    else:
        lines.append(f"P{s.turn} plays {card_str(a)}.")
    was_play = not s.comm_phase and not s.dist_phase
    E.step(s, a)
    if was_play and s.tricks_played > pre:
        lines.append(f"→ Trick won by P{s.leader}{' (you)' if s.leader == human_seat else ''}. "
                     f"Tasks {len(s.done_tasks)}/{len(s.task_cards)}")
    return lines


def advance_bots(s, human_seat, bot):
    """Clone `s` and let the bot play every NON-your seat until it is your turn
    again (or the game ends). Returns (new_state, log_lines)."""
    s = copy.deepcopy(s)
    lines = []
    while not s.done and s.turn != human_seat:
        lines += _step_and_log(s, bot.greedy_action(s), human_seat)
    return s, lines


def explore(root, bot, human_seat, rollouts):
    """Fork explorer (sandbox). From `root` (your turn), try a move and watch the
    line play out (bot fills the other seats), then `back` to try another — branch
    freely and compare outcomes. Nothing touches the real game. Returns the FIRST
    move of the line you settle on if you `commit`, else None."""
    path = [copy.deepcopy(root)]    # path[-1] = current node
    moves = []                      # moves[i]: your action taken from path[i] -> path[i+1]
    logs = [[]]                     # logs[i]: what happened arriving at path[i]
    tried = {}                      # first-move label -> "WIN"/"loss" of a watched line

    def first_label():
        return action_label(path[0], moves[0]) if moves else None

    print(_c("\n🔱 FORK EXPLORER — try a move, watch it play out, then 'back' and "
             "try another. 'commit' to play your line for real, 'done' to leave.", "1;95"))
    while True:
        cur, depth = path[-1], len(moves)
        lbl = first_label()
        print(_c(f"\n[fork] depth {depth}" +
                 (f" · this line opens with: {lbl}" if lbl else " · at your real decision"), "95"))
        for ln in logs[-1]:
            print(_c("   " + ln, "90"))

        idx_map = {}
        if cur.done:
            ok = cur.success
            print(_c(f"   ⇒ this line {'WINS ✅' if ok else 'LOSES ❌'} "
                     f"({len(cur.done_tasks)}/{len(cur.task_cards)} tasks)",
                     "1;92" if ok else "1;91"))
            cmds = "'back' up · 'root' restart · 'commit' · 'done'"
        elif cur.turn == human_seat:
            render_board(cur, human_seat)
            ordered, best, dist, win = analyze(cur, bot, rollouts)
            idx_map = print_menu(cur, ordered, best, dist, win)
            cmds = "# try a move · 'auto' bot finishes the line · 'back' · 'root' · 'commit' · 'done'"
        else:                                          # safety; advance_bots avoids this
            cmds = "'back' · 'done'"

        if tried:
            print(_c("   lines tried:", "1"))
            for k, v in tried.items():
                print(_c(f"     {k:<30} {v}", "92" if v == "WIN" else "90"))

        raw = input(_c(f"\n  [fork] {cmds}: ", "95")).strip().lower()

        if raw in ("done", "leave", "exit", ""):
            return None
        if raw in ("back", "u", "up"):
            if len(path) > 1:
                path.pop(); moves.pop(); logs.pop()
            else:
                print(_c("  already at the fork root.", "90"))
            continue
        if raw in ("root", "reset"):
            r0, l0 = path[0], logs[0]
            path[:], moves[:], logs[:] = [r0], [], [l0]
            continue
        if raw in ("commit", "play"):
            if moves:
                return moves[0]
            print(_c("  pick a move first — commit plays the FIRST move of your line.", "90"))
            continue
        if raw in ("auto", "finish") and not cur.done and cur.turn == human_seat:
            first = bot.greedy_action(cur)
            s2, lns = copy.deepcopy(cur), []
            while not s2.done:
                lns += _step_and_log(s2, bot.greedy_action(s2), human_seat)
            path.append(s2); moves.append(first); logs.append(lns)
            tried[first_label()] = "WIN" if s2.success else "loss"
            continue
        if raw.isdigit() and not cur.done and cur.turn == human_seat and int(raw) in idx_map:
            a = idx_map[int(raw)]
            s2 = copy.deepcopy(cur)
            lns = _step_and_log(s2, a, human_seat)
            s2, more = advance_bots(s2, human_seat, bot)
            path.append(s2); moves.append(a); logs.append(lns + more)
            if s2.done:
                tried[first_label()] = "WIN" if s2.success else "loss"
            continue
        print(_c("  ? unrecognized command.", "91"))


def human_decision(s, bot, human_seat, rollouts, divergences, blind=False):
    if blind:
        # --- blind mode: player picks first, analysis revealed after ---
        idx_map, _ = print_menu_blind(s)
        while True:
            raw = input(_c("\n  Choose # ('q' quit): ", "96")).strip().lower()
            if raw in ("q", "quit"):
                print("Bye!")
                sys.exit(0)
            if raw.isdigit() and int(raw) in idx_map:
                chosen = idx_map[int(raw)]
                break
            print(_c("  ? enter one of the listed numbers.", "91"))
        # now compute analysis and reveal
        ordered, best, dist, win = analyze(s, bot, rollouts)
        reveal_analysis(s, chosen, ordered, best, dist, win)
    else:
        # --- normal mode: analysis shown upfront ---
        ordered, best, dist, win = analyze(s, bot, rollouts)
        idx_map = print_menu(s, ordered, best, dist, win)
        while True:
            raw = input(_c("\n  Choose # (or 'h' hint, 'd' deep analysis, 'f' fork & explore, "
                           "'q' quit): ", "96")).strip().lower()
            if raw in ("q", "quit"):
                print("Bye!")
                sys.exit(0)
            if raw in ("h", "hint", ""):
                why = f"Bot recommends {action_label(s, best)}"
                if best in win:
                    why += f" — best win chance ({100*win[best]:.0f}%)."
                print(_c("  💡 " + why, "92"))
                continue
            if raw in ("f", "fork", "explore"):
                committed = explore(s, bot, human_seat, rollouts)
                if committed is not None:
                    chosen = committed
                    print(_c(f"  ↳ committing your explored move: {action_label(s, chosen)}", "95"))
                    break
                idx_map = print_menu(s, ordered, best, dist, win)   # redraw on return
                continue
            if raw in ("d", "deep"):
                if bot.model is None:
                    print(_c("  (deeper analysis needs a trained checkpoint)", "90"))
                    continue
                print(_c("  …deep rollout (200/move)", "90"))
                ordered, best, dist, win = analyze(s, bot, 200, show_progress=False)
                idx_map = print_menu(s, ordered, best, dist, win)
                continue
            if raw.isdigit() and int(raw) in idx_map:
                chosen = idx_map[int(raw)]
                break
            print(_c("  ? enter one of the listed numbers.", "91"))

    # coaching feedback (shared for both modes)
    if chosen == best:
        print(_c(f"  ✓ Agrees with the bot: {action_label(s, chosen)}", "92"))
    else:
        msg = f"  ⓘ You chose {action_label(s, chosen)}; bot preferred {action_label(s, best)}"
        if chosen in win and best in win:
            gap = 100 * (win[best] - win[chosen])
            msg += f"  (≈{gap:.0f}% win-prob difference)"
            divergences.append((s.mission.mission_id, s.tricks_played,
                                action_label(s, chosen), action_label(s, best), gap))
        print(_c(msg, "93" if (chosen in win and win[best] - win.get(chosen, 0) > 0.05) else "90"))
    return chosen


# ---------------------------------------------------------------------------
# End-of-game reveal: starting hands + a trick-by-trick play summary
# ---------------------------------------------------------------------------

def print_reveal(start_hands, final_state, human_seat, play_log):
    """After the mission resolves, show everyone's original hands and replay the
    whole deal trick by trick (winner starred, task captures flagged)."""
    s = final_state
    commander = next(p for p in range(E.N_PLAYERS) if 39 in start_hands[p])

    print(_c("\nStarting hands (revealed):", "1;97"))
    for seat in range(E.N_PLAYERS):
        you = _c(" (you)", "1;96") if seat == human_seat else ""
        cmd = _c("  ◄ commander", "90") if seat == commander else ""
        print(f"  P{seat}{you}{cmd}")
        print("     " + hand_str(start_hands[seat]))

    # signals made during the communication round
    sig = []
    for seat in range(E.N_PLAYERS):
        card, htype, valid = s.comm[seat]
        if valid:
            kind = {0: "only", 1: "high", 2: "low"}[htype]
            sig.append(f"P{seat}{'(you)' if seat == human_seat else ''}: "
                       f"{card_str(card)}({kind})")
    print(_c("\nSignals: ", "1;97") + ("   ".join(sig) if sig else "(nobody signalled)"))

    print(_c("\nPlay-by-play:", "1;97"))
    for ti, trick in enumerate(play_log, 1):
        cells = []
        for pl, c in trick["plays"]:
            star = _c("★", "1;92") if pl == trick["winner"] else ""
            cells.append(f"P{pl} {card_str(c)}{star}")
        won = trick["winner"]
        line = (f"  Trick {ti:>2}: " + "   ".join(cells) +
                f"   → won by P{won}{' (you)' if won == human_seat else ''}")
        if trick["new_tasks"]:
            line += _c("   ✔ task " + " ".join(card_str(c) for c in trick["new_tasks"]),
                       "92")
        print(line)


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def play_game(mission_id, human_seat, bot, rollouts, seed=None, blind=False):
    rng = np.random.default_rng(seed)
    s = E.new_game(rng, mission_id, use_comm=True)
    if human_seat is None:
        human_seat = s.leader                       # default: you are the commander
    print(_c(f"\nYou are P{human_seat}. "
             f"P{s.leader} holds the highest rocket and leads / signals first.", "1"))
    print(f"Bot: {bot.name}")

    # Hands get mutated as cards are played, so snapshot the deal up front to
    # reveal at the end alongside a trick-by-trick replay.
    start_hands = copy.deepcopy(s.hands)
    play_log = []                       # one entry per completed trick
    cur_trick = []                      # (player, card) plays of the in-progress trick
    prev_done = set()                   # task cards already completed before this trick

    divergences = []
    last_trick_count = 0
    while not s.done:
        actor, in_comm = s.turn, s.comm_phase
        if s.turn == human_seat:
            render_board(s, human_seat)
            a = human_decision(s, bot, human_seat, rollouts, divergences, blind=blind)
            E.step(s, a)
        else:
            a = bot.greedy_action(s)
            if s.comm_phase:
                if a == E.PASS_ACTION:
                    print(_c(f"  P{s.turn} passes (no signal).", "90"))
                else:
                    print(_c(f"  P{s.turn} signals {card_str(a - E.COMM_OFFSET)}.", "90"))
            else:
                print(_c(f"  P{s.turn} plays {card_str(a)}.", "90"))
            E.step(s, a)

        if not in_comm:
            cur_trick.append((actor, a))

        # announce a resolved trick
        if not s.comm_phase and s.tricks_played > last_trick_count:
            last_trick_count = s.tricks_played
            new_tasks = sorted(s.done_tasks - prev_done)
            prev_done = set(s.done_tasks)
            play_log.append({"plays": cur_trick, "winner": s.leader,
                             "new_tasks": new_tasks})
            cur_trick = []
            print(_c(f"  → Trick won by P{s.leader}"
                     f"{' (you)' if s.leader == human_seat else ''}. "
                     f"Tasks done: {len(s.done_tasks)}/{len(s.task_cards)}", "1;90"))

    # outcome
    print("\n" + "█" * 70)
    if s.success:
        print(_c("  🎉 MISSION COMPLETE — all tasks captured by the right players!", "1;92"))
    else:
        print(_c(f"  ✗ Mission failed. Tasks completed: "
                 f"{len(s.done_tasks)}/{len(s.task_cards)}.", "1;91"))
    print("█" * 70)

    print_reveal(start_hands, s, human_seat, play_log)

    if divergences:
        print(_c("\nCoaching recap — where you differed from the bot:", "1;97"))
        for mid, trick, chose, rec, gap in divergences:
            g = round(gap)
            sign = (_c(f"-{g}%", "91") if g > 0 else
                    _c(f"+{-g}%", "92") if g < 0 else _c("~0%", "90"))
            print(f"  trick {trick + 1}: you {chose}  vs  bot {rec}   ({sign} win-prob)")
    else:
        print(_c("\nYou matched the bot's top pick on every move. 🧠", "92"))


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(description="Play The Crew with the trained bot and learn optimal moves.")
    ap.add_argument("--mission", type=int, default=None, help="mission 1..50 (prompted if omitted)")
    ap.add_argument("--seat", type=int, default=None, choices=[0, 1, 2],
                    help="your seat (default: the commander)")
    ap.add_argument("--rollouts", type=int, default=60,
                    help="rollouts/move for the win%% estimate (0 = off, faster)")
    _repo_ckpt  = os.path.join(_SCRIPT_DIR, "ppo_real_model.pt")
    _cache_ckpt = os.path.join(E.CACHE_DIR,  "ppo_real_model.pt")
    _default_ckpt = _repo_ckpt if os.path.exists(_repo_ckpt) else _cache_ckpt
    ap.add_argument("--ckpt", type=str, default=_default_ckpt)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--blind", action="store_true",
                    help="hide bot instinct/win%% until after you move; revealed as feedback")
    args = ap.parse_args()
    _USE_COLOR = not args.no_color

    device = "cuda" if (_HAVE_TORCH and torch.cuda.is_available()) else "cpu"
    bot = Bot(ckpt=args.ckpt, device=device)

    print(_c("\n  The Crew: The Quest for Planet Nine — play & learn with the bot", "1;96"))
    if args.blind:
        print(_c("  [BLIND MODE] Bot analysis is hidden — make your move, then it's revealed.", "1;95"))
    if bot.model is None:
        print(_c("  (No checkpoint — using the heuristic. Train one with `python ppo.py`.)", "93"))

    mission = args.mission
    if mission is None:
        raw = input("  Mission to play (1-50, harder = more tasks) [1]: ").strip()
        mission = int(raw) if raw.isdigit() and 1 <= int(raw) <= 50 else 1

    while True:
        play_game(mission, args.seat, bot, args.rollouts, seed=args.seed, blind=args.blind)
        again = input(_c("\n  Play again? same mission [Enter], new # 1-50, or 'q': ", "96")).strip().lower()
        if again in ("q", "quit", "n", "no"):
            break
        if again.isdigit() and 1 <= int(again) <= 50:
            mission = int(again)
    print("Thanks for playing!")


if __name__ == "__main__":
    main()
