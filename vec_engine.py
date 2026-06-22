"""
Vectorized (batched) Crew environment for GPU self-play.

This steps B independent games in lockstep as tensors so that policy inference
AND environment transitions are batched on the GPU — the only way a GPU helps
self-play for a 40-card game. It reproduces crew_engine.py EXACTLY (verified by
test_vec_consistency.py): same legal moves (play + communicate), trick
resolution, task/order logic, rewards, and observation encoding (block for
block, OBS_DIM/ACT_DIM read from crew_engine).

State is a bag of tensors of shape [B, ...]. All games run for a fixed
E.MAX_PLIES plies (plays + communications); finished games are masked out
(no-ops) until the wave ends.
"""

import numpy as np
import torch

import crew_engine as E

N_P = E.N_PLAYERS
N_C = E.N_CARDS
TOTAL_TRICKS = E.TOTAL_TRICKS

# Per-card constant lookups (built once, moved to device on demand)
_COLOR = torch.tensor([E.card_color(c) for c in range(N_C)], dtype=torch.long)
_RANK = torch.tensor([E.card_rank(c) for c in range(N_C)], dtype=torch.long)
_TRUMP = torch.tensor([E.is_trump(c) for c in range(N_C)], dtype=torch.bool)


class VecCrew:
    def __init__(self, B, device, color, rank, trump):
        self.B = B
        self.device = device
        self.COLOR, self.RANK, self.TRUMP = color, rank, trump
        self.arangeB = torch.arange(B, device=device)

    # ------------------------------------------------------------------ resets
    @classmethod
    def from_scalar(cls, states, device="cpu"):
        """Build a batch from a list of crew_engine.GameState.

        Assembles everything in NumPy first, then transfers each field to the
        device in ONE shot. (Per-element writes into CUDA tensors here were the
        dominant cost of self-play resets — ~80k tiny kernel launches per batch.)
        """
        B = len(states)
        owner = np.full((B, N_C), -1, np.int64)
        assigned = np.full((B, N_C), -1, np.int64)
        order_pos = np.zeros((B, N_C), np.int64)
        done_tasks = np.zeros((B, N_C), bool)
        captured = np.full((B, N_C), -1, np.int64)
        table = np.full((B, N_P), -1, np.int64)
        comm_card = np.full((B, N_P), -1, np.int64)
        comm_type = np.full((B, N_P), -1, np.int64)
        comm_valid = np.zeros((B, N_P), bool)
        comm_phase = np.zeros(B, bool); comm_count = np.zeros(B, np.int64)
        led = np.full(B, -1, np.int64)
        leader = np.zeros(B, np.int64); turn = np.zeros(B, np.int64)
        plays = np.zeros(B, np.int64); tricks = np.zeros(B, np.int64)
        done = np.zeros(B, bool); success = np.zeros(B, bool); failed = np.zeros(B, bool)
        for b, s in enumerate(states):
            for p in range(N_P):
                for c in s.hands[p]:
                    owner[b, c] = p
            assigned[b] = s.assigned
            order_pos[b] = s.order_pos
            for c in s.done_tasks:
                done_tasks[b, c] = True
            captured[b] = s.captured_by
            for j, (pl, c) in enumerate(s.on_table):
                table[b, j] = c
            plays[b] = len(s.on_table)
            for p in range(N_P):
                card, htype, valid = s.comm[p]
                comm_card[b, p] = card; comm_type[b, p] = htype; comm_valid[b, p] = bool(valid)
            comm_phase[b] = bool(s.comm_phase); comm_count[b] = s.comm_count
            led[b] = s.led_color; leader[b] = s.leader; turn[b] = s.turn
            tricks[b] = s.tricks_played
            done[b] = s.done; success[b] = s.success; failed[b] = s.failed

        color, rank, trump = (_COLOR.to(device), _RANK.to(device), _TRUMP.to(device))
        self = cls(B, device, color, rank, trump)
        t = lambda a: torch.from_numpy(a).to(device)
        self.owner = t(owner); self.assigned = t(assigned); self.order_pos = t(order_pos)
        self.done_tasks = t(done_tasks); self.captured_by = t(captured); self.table = t(table)
        self.comm_card = t(comm_card); self.comm_type = t(comm_type); self.comm_valid = t(comm_valid)
        self.comm_phase = t(comm_phase); self.comm_count = t(comm_count)
        self.led = t(led); self.leader = t(leader); self.turn = t(turn)
        self.plays = t(plays); self.tricks = t(tricks)
        self.done = t(done); self.success = t(success); self.failed = t(failed)
        return self

    @classmethod
    def new_games(cls, B, mission_ids, rng, device="cpu", solvable_only=False):
        """Fresh batch. mission_ids: array-like length B. rng: np.random.Generator."""
        states = [E.new_game(rng, int(mission_ids[b]), solvable_only=solvable_only)
                  for b in range(B)]
        return cls.from_scalar(states, device=device)

    # ------------------------------------------------------------------- moves
    def _comm_table(self):
        """Per current player: (comm_legal [B,N_C] bool, comm_type [B,N_C] long).
        Mirrors crew_engine.communicable: only/highest/lowest non-trump card."""
        B, dev = self.B, self.device
        hand = self.owner == self.turn[:, None]                    # [B,N_C]
        comm_legal = torch.zeros((B, N_C), dtype=torch.bool, device=dev)
        comm_type = torch.zeros((B, N_C), dtype=torch.long, device=dev)
        rpc = E.RANKS_PER_COLOR
        for col in range(E.N_COLORS):
            lo, hi = col * rpc, col * rpc + rpc
            in_col = hand[:, lo:hi]                                # [B,rpc]
            ranks = self.RANK[lo:hi]                               # [rpc] = 1..9
            cnt = in_col.sum(dim=1, keepdim=True)                  # [B,1]
            big = torch.where(in_col, ranks[None, :], torch.zeros_like(ranks)[None, :])
            small = torch.where(in_col, ranks[None, :], torch.full_like(ranks, 99)[None, :])
            maxr = big.max(dim=1, keepdim=True).values
            minr = small.min(dim=1, keepdim=True).values
            only = in_col & (cnt == 1)
            ishigh = in_col & (cnt >= 2) & (ranks[None, :] == maxr)
            islow = in_col & (cnt >= 2) & (ranks[None, :] == minr)
            comm_legal[:, lo:hi] = only | ishigh | islow
            comm_type[:, lo:hi] = ishigh.long() + 2 * islow.long()   # only -> 0
        self._ctype_cache = comm_type   # reused by step() within the same ply
        return comm_legal, comm_type

    def legal_mask(self):
        hand = self.owner == self.turn[:, None]                    # [B,N_C]
        leading = self.led == -1                                   # [B]
        same = self.COLOR[None, :] == self.led[:, None]            # [B,N_C]
        follow = hand & same & (~leading[:, None])
        has_follow = follow.any(dim=1)                             # [B]
        use_follow = (~leading) & has_follow
        play = torch.where(use_follow[:, None], follow, hand)      # [B,N_C]
        comm_legal, _ = self._comm_table()
        in_comm = self.comm_phase[:, None]                         # [B,1]
        full = torch.zeros((self.B, E.ACT_DIM), dtype=torch.bool, device=self.device)
        full[:, :N_C] = play & (~in_comm)                          # plays only in play phase
        full[:, N_C:2 * N_C] = comm_legal & in_comm                # comm only in comm phase
        full[:, E.PASS_ACTION] = self.comm_phase                   # pass legal in comm phase
        # safety for finished games whose current player may be empty-handed
        empty = ~full.any(dim=1)
        if empty.any():
            full[empty, 0] = True
        return full

    def _strength(self, cards):
        """Trick strength of cards [.. ] given each game's led color. cards: [B,K]."""
        col = self.COLOR[cards]
        rnk = self.RANK[cards]
        tr = self.TRUMP[cards]
        led = self.led[:, None]
        s = torch.zeros_like(cards)
        s = torch.where(col == led, 100 + rnk, s)   # following led color
        s = torch.where(tr, 200 + rnk, s)           # trump dominates
        s = torch.where(cards < 0, torch.full_like(s, -1), s)  # empty seats
        return s

    # ---------------------------------------------------------------- observe
    def observe(self):
        B, dev = self.B, self.device
        turn = self.turn
        blocks = []
        # 1 hand
        blocks.append((self.owner == turn[:, None]).float())
        # 2 captured (out of play)
        blocks.append((self.captured_by >= 0).float())
        # 3 cards on table
        table_oh = torch.zeros((B, N_C), device=dev)
        for j in range(N_P):
            c = self.table[:, j]
            valid = c >= 0
            table_oh[self.arangeB[valid], c[valid]] = 1.0
        blocks.append(table_oh)
        # 4 current winning card
        win_oh = torch.zeros((B, N_C), device=dev)
        stg = self._strength(self.table)                  # [B,N_P]
        any_play = self.plays > 0
        seat = stg.argmax(dim=1)
        wcard = self.table[self.arangeB, seat]
        sel = any_play & (wcard >= 0)
        win_oh[self.arangeB[sel], wcard[sel]] = 1.0
        blocks.append(win_oh)
        # 5 led color one-hot (length N_COLORS+1 = 5)
        led_oh = torch.zeros((B, E.N_COLORS + 1), device=dev)
        has_led = self.led >= 0
        led_oh[self.arangeB[has_led], self.led[has_led]] = 1.0
        blocks.append(led_oh)
        # 6/7/8 tasks mine / other / done
        assigned, dt = self.assigned, self.done_tasks
        mine = (assigned == turn[:, None]) & (~dt)
        other = (assigned >= 0) & (assigned != turn[:, None]) & (~dt)
        blocks.append(mine.float()); blocks.append(other.float()); blocks.append(dt.float())
        # 9 order position normalized
        max_order = self.order_pos.max(dim=1).values.clamp(min=1)
        blocks.append((self.order_pos.float() / max_order[:, None].float()))
        # 10 ready
        op = self.order_pos
        ordered = op > 0
        open_task = (assigned >= 0) & (~dt)
        less = (op[:, None, :] > 0) & (op[:, None, :] < op[:, :, None]) & dt[:, None, :]
        earlier_done = less.sum(dim=2)                    # [B,N_C]
        ready_ordered = ordered & open_task & (earlier_done == (op - 1))
        ready_unordered = (~ordered) & open_task
        blocks.append((ready_ordered | ready_unordered).float())
        # 11 communication, relative seating
        for off in range(N_P):
            q = (turn + off) % N_P
            card = self.comm_card[self.arangeB, q]
            htype = self.comm_type[self.arangeB, q]
            card_oh = torch.zeros((B, N_C), device=dev)
            m = card >= 0
            card_oh[self.arangeB[m], card[m]] = 1.0
            type_oh = torch.zeros((B, 3), device=dev)
            mt = htype >= 0
            type_oh[self.arangeB[mt], htype[mt]] = 1.0
            valid = self.comm_valid[self.arangeB, q].float()[:, None]
            blocks.append(card_oh); blocks.append(type_oh); blocks.append(valid)
        # 12 relative hand sizes
        counts = torch.stack([(self.owner == p).sum(dim=1) for p in range(N_P)], dim=1).float()
        rel = torch.stack([counts[self.arangeB, (turn + off) % N_P] for off in range(N_P)], dim=1)
        blocks.append(rel / E.CARDS_PER_PLAYER_MAX)
        # 13 scalars (incl. communication-phase flag)
        sc = torch.stack([
            self.tricks.float() / TOTAL_TRICKS,
            (turn == self.leader).float(),
            self.plays.float() / N_P,
            self.comm_phase.float(),
        ], dim=1)
        blocks.append(sc)
        return torch.cat(blocks, dim=1)

    # ------------------------------------------------------------------- step
    def step(self, actions):
        """Apply actions [B]. During the comm phase: pass (PASS_ACTION) or
        communicate (N_C..2N_C-1); otherwise play (0..N_C-1)."""
        active = ~self.done
        in_comm = active & self.comm_phase
        in_play = active & (~self.comm_phase)

        # --- communication round: record signal (if not a pass), then advance
        #     the per-game decision pointer; exit to play after N_P decisions ---
        if bool(in_comm.any()):
            ctype = getattr(self, "_ctype_cache", None)
            if ctype is None:
                _, ctype = self._comm_table()
            ci = in_comm.nonzero(as_tuple=True)[0]
            real = actions[ci] < E.PASS_ACTION          # comm action (not a pass)
            rci = ci[real]
            if rci.numel() > 0:
                rtp = self.turn[rci]
                rc = actions[rci] - N_C
                self.comm_card[rci, rtp] = rc
                self.comm_type[rci, rtp] = ctype[rci, rc]
                self.comm_valid[rci, rtp] = True
            self.comm_count[ci] += 1
            done_comm = self.comm_count[ci] >= N_P
            self.comm_phase[ci] = ~done_comm
            self.turn[ci] = torch.where(done_comm, self.leader[ci],
                                        (self.leader[ci] + self.comm_count[ci]) % N_P)

        # --- play: clear a consumed signal, place card, advance, maybe resolve ---
        pidx = in_play.nonzero(as_tuple=True)[0]
        if pidx.numel() > 0:
            pturn = self.turn[pidx]
            pa = actions[pidx]
            clr = self.comm_card[pidx, pturn] == pa     # played the communicated card
            cci = pidx[clr]
            if cci.numel() > 0:
                ct = self.turn[cci]
                self.comm_card[cci, ct] = -1
                self.comm_type[cci, ct] = -1
                self.comm_valid[cci, ct] = False
        safe_a = torch.where(actions < N_C, actions, torch.zeros_like(actions))
        first = in_play & (self.plays == 0)
        self.led = torch.where(first, self.COLOR[safe_a], self.led)
        self.table[pidx, self.plays[pidx]] = actions[pidx]
        self.owner[pidx, actions[pidx]] = -1
        self.plays = torch.where(in_play, self.plays + 1, self.plays)
        self.turn = torch.where(in_play, (self.turn + 1) % N_P, self.turn)

        resolve = in_play & (self.plays == N_P)
        if bool(resolve.any()):
            self._resolve(resolve)
        self._ctype_cache = None   # state changed: invalidate comm_type cache
        return self

    def _resolve(self, R):
        ridx = R.nonzero(as_tuple=True)[0]
        winner = (self.leader + self._strength(self.table).argmax(dim=1)) % N_P  # [B]
        wr = winner[ridx]                                  # winners for resolved games

        # Pass 1 (scalar parity): capture every card in the trick.
        for j in range(N_P):
            self.captured_by[ridx, self.table[ridx, j]] = wr

        # Pass 2: evaluate task cards in play order, growing done_tasks
        # incrementally; once a game has failed, stop crediting its tasks.
        failed_acc = torch.zeros(len(ridx), dtype=torch.bool, device=self.device)
        for j in range(N_P):
            c = self.table[ridx, j]
            asn = self.assigned[ridx, c]
            is_task = asn >= 0
            wrong = is_task & (wr != asn)
            op = self.order_pos[ridx, c]
            less = (self.order_pos[ridx] > 0) & (self.order_pos[ridx] < op[:, None]) & self.done_tasks[ridx]
            order_bad = is_task & (op > 0) & (less.sum(dim=1) != (op - 1))
            can_done = is_task & (~wrong) & (~order_bad) & (~failed_acc)
            gi = can_done.nonzero(as_tuple=True)[0]
            self.done_tasks[ridx[gi], c[gi]] = True
            failed_acc = failed_acc | wrong | order_bad
        self.failed[ridx] = self.failed[ridx] | failed_acc

        # reset trick, winner leads next
        self.table[ridx] = -1
        self.led[ridx] = -1
        self.plays[ridx] = 0
        self.leader[ridx] = wr
        self.turn[ridx] = wr
        self.tricks[ridx] += 1

        # terminal checks
        n_tasks = (self.assigned >= 0).sum(dim=1)
        n_done = self.done_tasks.sum(dim=1)
        failed = R & self.failed
        succ = R & (~self.failed) & (n_done == n_tasks)
        timeout = R & (~self.failed) & (self.tricks >= TOTAL_TRICKS)
        self.done = self.done | failed | succ | timeout
        self.success = self.success | succ
