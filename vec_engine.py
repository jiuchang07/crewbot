"""
Vectorized (batched) Crew environment for GPU self-play.

Steps B independent games in lockstep as tensors so policy inference AND env
transitions batch on the GPU. Reproduces crew_engine.py EXACTLY (verified by
test_vec_consistency.py): task distribution (claiming), communication, trick
resolution, priority-tile ordering (predecessor matrix), rewards, and the full
observation. Phase order per game: distribute -> communicate -> play.

State is a bag of [B, ...] tensors run for E.MAX_PLIES plies; finished games are
masked no-ops until the wave ends.
"""

import numpy as np
import torch

import crew_engine as E

N_P = E.N_PLAYERS
N_C = E.N_CARDS
N_TC = E.N_TASK_CARDS          # 36 claimable task cards
TOTAL_TRICKS = E.TOTAL_TRICKS

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
        """Build a batch from crew_engine.GameState list (NumPy first, one transfer)."""
        B = len(states)
        owner = np.full((B, N_C), -1, np.int64)
        assigned = np.full((B, N_C), -1, np.int64)
        is_task = np.zeros((B, N_C), bool)
        pred = np.zeros((B, N_C, N_C), bool)          # pred[b,c,d] = d must precede c
        done_tasks = np.zeros((B, N_C), bool)
        captured = np.full((B, N_C), -1, np.int64)
        table = np.full((B, N_P), -1, np.int64)
        comm_card = np.full((B, N_P), -1, np.int64)
        comm_type = np.full((B, N_P), -1, np.int64)
        comm_valid = np.zeros((B, N_P), bool)
        allow_comm = np.zeros(B, bool)
        dist_phase = np.zeros(B, bool); dist_count = np.zeros(B, np.int64)
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
            for c in s.task_cards:
                is_task[b, c] = True
                for d in s.card_preds.get(c, ()):
                    pred[b, c, d] = True
            for c in s.done_tasks:
                done_tasks[b, c] = True
            captured[b] = s.captured_by
            for j, (pl, c) in enumerate(s.on_table):
                table[b, j] = c
            plays[b] = len(s.on_table)
            for p in range(N_P):
                card, htype, valid = s.comm[p]
                comm_card[b, p] = card; comm_type[b, p] = htype; comm_valid[b, p] = bool(valid)
            allow_comm[b] = bool(s.allow_comm)
            dist_phase[b] = bool(s.dist_phase); dist_count[b] = s.dist_count
            comm_phase[b] = bool(s.comm_phase); comm_count[b] = s.comm_count
            led[b] = s.led_color; leader[b] = s.leader; turn[b] = s.turn
            tricks[b] = s.tricks_played
            done[b] = s.done; success[b] = s.success; failed[b] = s.failed

        color, rank, trump = (_COLOR.to(device), _RANK.to(device), _TRUMP.to(device))
        self = cls(B, device, color, rank, trump)
        t = lambda a: torch.from_numpy(a).to(device)
        self.owner = t(owner); self.assigned = t(assigned)
        self.is_task = t(is_task); self.pred = t(pred)
        self.done_tasks = t(done_tasks); self.captured_by = t(captured); self.table = t(table)
        self.comm_card = t(comm_card); self.comm_type = t(comm_type); self.comm_valid = t(comm_valid)
        self.allow_comm = t(allow_comm)
        self.dist_phase = t(dist_phase); self.dist_count = t(dist_count)
        self.comm_phase = t(comm_phase); self.comm_count = t(comm_count)
        self.led = t(led); self.leader = t(leader); self.turn = t(turn)
        self.plays = t(plays); self.tricks = t(tricks)
        self.done = t(done); self.success = t(success); self.failed = t(failed)
        return self

    @classmethod
    def new_games(cls, B, mission_ids, rng, device="cpu", solvable_only=False):
        states = [E.new_game(rng, int(mission_ids[b]), solvable_only=solvable_only)
                  for b in range(B)]
        return cls.from_scalar(states, device=device)

    # --------------------------------------------------------------- Other-Play
    @staticmethod
    def build_perm_table(color_perms, device="cpu"):
        """Build a full card-index permutation table from color permutations.

        Args:
            color_perms: [B, 4] long tensor — a permutation of {0,1,2,3} per game.
        Returns:
            perm: [B, N_C] long tensor mapping old card index → new card index.
            inv:  [B, N_C] long tensor mapping new card index → old card index.
        """
        B = color_perms.shape[0]
        RPC = E.RANKS_PER_COLOR  # 9
        # For non-trump cards: new_index = perm_color[old_color] * 9 + old_rank_offset
        old_idx = torch.arange(N_C, device=device).unsqueeze(0).expand(B, -1)  # [B, N_C]
        perm = old_idx.clone()
        # Only remap non-trump cards (indices 0..35)
        old_color = old_idx[:, :N_TC] // RPC          # [B, 36]
        rank_off  = old_idx[:, :N_TC] % RPC           # [B, 36]
        new_color = color_perms.gather(1, old_color)   # [B, 36]
        perm[:, :N_TC] = new_color * RPC + rank_off
        # Build inverse: inv[perm[i]] = i
        inv = torch.zeros_like(perm)
        inv.scatter_(1, perm, old_idx)
        return perm, inv

    def permute_colors(self, color_perms):
        """Apply a random color permutation (Other-Play) to all game state tensors.

        Args:
            color_perms: [B, 4] long tensor — a permutation of {0,1,2,3} per game.

        Mutates self in-place and returns self for chaining.
        """
        perm, inv = self.build_perm_table(color_perms, device=self.device)

        # Helper: remap a [B, N_C] tensor indexed by card slot.
        # card-indexed tensors have semantics like "value at card slot c" — we want
        # the value that was at old slot c to move to new slot perm[c].
        # Equivalently, new[b, perm[b,c]] = old[b, c]  →  new[b, j] = old[b, inv[b,j]]
        def remap_card_slots(t):
            return t.gather(1, inv)

        # Helper: remap a tensor that *stores* card indices as values.
        # E.g. table[b, j] = card_index → should become perm[card_index].
        # Entries with sentinel -1 are left unchanged.
        def remap_card_values(t):
            valid = t >= 0
            flat_perm = perm  # [B, N_C]
            # Clamp for safe indexing, then mask
            safe = t.clamp(min=0)
            # Gather from perm along the card dimension
            remapped = flat_perm.gather(1, safe) if safe.shape[1] == perm.shape[1] else \
                       torch.stack([perm[b, safe[b]] for b in range(self.B)])
            return torch.where(valid, remapped, t)

        def remap_card_values_small(t):
            """Remap a [B, K] tensor of card-index values (K < N_C) using perm."""
            valid = t >= 0
            safe = t.clamp(min=0)
            remapped = torch.stack([perm[b].gather(0, safe[b]) for b in range(self.B)])
            return torch.where(valid, remapped, t)

        # 1. owner[B, N_C]: who holds card c → remap card slots
        self.owner = remap_card_slots(self.owner)

        # 2. assigned[B, N_C]: task assignment per card → remap card slots
        self.assigned = remap_card_slots(self.assigned)

        # 3. is_task[B, N_C]: bool per card → remap card slots
        self.is_task = remap_card_slots(self.is_task)

        # 4. pred[B, N_C, N_C]: pred[b,c,d] = d must precede c → remap both axes
        # new_pred[b, perm[c], perm[d]] = old_pred[b, c, d]
        # → new_pred[b, i, j] = old_pred[b, inv[i], inv[j]]
        self.pred = self.pred.gather(1, inv.unsqueeze(2).expand_as(self.pred))
        self.pred = self.pred.gather(2, inv.unsqueeze(1).expand_as(self.pred))

        # 5. done_tasks[B, N_C]: bool per card → remap card slots
        self.done_tasks = remap_card_slots(self.done_tasks)

        # 6. captured_by[B, N_C]: who captured card c → remap card slots
        self.captured_by = remap_card_slots(self.captured_by)

        # 7. table[B, N_P]: stores card indices → remap card values
        self.table = remap_card_values_small(self.table)

        # 8. comm_card[B, N_P]: stores card indices → remap card values
        self.comm_card = remap_card_values_small(self.comm_card)

        # 9. led[B]: stores color index → remap via color_perms
        has_led = self.led >= 0
        # led stores a color (0-3) or -1; trumps have led_color = TRUMP_COLOR (4) 
        is_normal_color = has_led & (self.led < E.N_COLORS)
        safe_led = self.led.clamp(min=0)
        new_led = color_perms[self.arangeB, safe_led]
        self.led = torch.where(is_normal_color, new_led, self.led)

        return self

    # ------------------------------------------------------------------- moves
    def _comm_table(self):
        """Per current player: (comm_legal [B,N_C] bool, comm_type [B,N_C] long)."""
        B, dev = self.B, self.device
        hand = self.owner == self.turn[:, None]
        comm_legal = torch.zeros((B, N_C), dtype=torch.bool, device=dev)
        comm_type = torch.zeros((B, N_C), dtype=torch.long, device=dev)
        rpc = E.RANKS_PER_COLOR
        for col in range(E.N_COLORS):
            lo, hi = col * rpc, col * rpc + rpc
            in_col = hand[:, lo:hi]
            ranks = self.RANK[lo:hi]
            cnt = in_col.sum(dim=1, keepdim=True)
            big = torch.where(in_col, ranks[None, :], torch.zeros_like(ranks)[None, :])
            small = torch.where(in_col, ranks[None, :], torch.full_like(ranks, 99)[None, :])
            maxr = big.max(dim=1, keepdim=True).values
            minr = small.min(dim=1, keepdim=True).values
            only = in_col & (cnt == 1)
            ishigh = in_col & (cnt >= 2) & (ranks[None, :] == maxr)
            islow = in_col & (cnt >= 2) & (ranks[None, :] == minr)
            comm_legal[:, lo:hi] = only | ishigh | islow
            comm_type[:, lo:hi] = ishigh.long() + 2 * islow.long()
        self._ctype_cache = comm_type
        return comm_legal, comm_type

    def legal_mask(self):
        dev = self.device
        hand = self.owner == self.turn[:, None]
        leading = self.led == -1
        same = self.COLOR[None, :] == self.led[:, None]
        follow = hand & same & (~leading[:, None])
        has_follow = follow.any(dim=1)
        use_follow = (~leading) & has_follow
        play = torch.where(use_follow[:, None], follow, hand)
        comm_legal, _ = self._comm_table()

        in_play = (~self.comm_phase) & (~self.dist_phase)
        full = torch.zeros((self.B, E.ACT_DIM), dtype=torch.bool, device=dev)
        full[:, :N_C] = play & in_play[:, None]
        full[:, N_C:2 * N_C] = comm_legal & self.comm_phase[:, None]
        full[:, E.PASS_ACTION] = self.comm_phase
        # claim actions (0..35): unclaimed task cards, during the distribution phase
        claimable = self.is_task[:, :N_TC] & (self.assigned[:, :N_TC] == -1) & self.dist_phase[:, None]
        full[:, E.CLAIM_OFFSET:E.CLAIM_OFFSET + N_TC] = claimable
        empty = ~full.any(dim=1)
        if empty.any():
            full[empty, 0] = True
        return full

    def _strength(self, cards):
        col = self.COLOR[cards]; rnk = self.RANK[cards]; tr = self.TRUMP[cards]
        led = self.led[:, None]
        s = torch.zeros_like(cards)
        s = torch.where(col == led, 100 + rnk, s)
        s = torch.where(tr, 200 + rnk, s)
        s = torch.where(cards < 0, torch.full_like(s, -1), s)
        return s

    # ---------------------------------------------------------------- observe
    def observe(self):
        B, dev = self.B, self.device
        turn = self.turn
        blocks = []
        # 1 hand
        blocks.append((self.owner == turn[:, None]).float())
        # 2 captured
        blocks.append((self.captured_by >= 0).float())
        # 3 table
        table_oh = torch.zeros((B, N_C), device=dev)
        for j in range(N_P):
            c = self.table[:, j]; valid = c >= 0
            table_oh[self.arangeB[valid], c[valid]] = 1.0
        blocks.append(table_oh)
        # 4 winning card
        win_oh = torch.zeros((B, N_C), device=dev)
        seat = self._strength(self.table).argmax(dim=1)
        wcard = self.table[self.arangeB, seat]
        sel = (self.plays > 0) & (wcard >= 0)
        win_oh[self.arangeB[sel], wcard[sel]] = 1.0
        blocks.append(win_oh)
        # 5 led color
        led_oh = torch.zeros((B, E.N_COLORS + 1), device=dev)
        has_led = self.led >= 0
        led_oh[self.arangeB[has_led], self.led[has_led]] = 1.0
        blocks.append(led_oh)
        # 6/7/8 tasks mine / other / done (by claimer)
        assigned, dt = self.assigned, self.done_tasks
        mine = (assigned == turn[:, None]) & (~dt)
        other = (assigned >= 0) & (assigned != turn[:, None]) & (~dt)
        blocks.append(mine.float()); blocks.append(other.float()); blocks.append(dt.float())
        # 8b distribution pool: revealed tasks, and which remain unclaimed
        blocks.append(self.is_task.float())
        blocks.append((self.is_task & (assigned == -1)).float())
        # 9 blocked-ness + 10 ready (over all revealed tasks)
        open_task = self.is_task & (~dt)
        unmet = self.pred & (~dt[:, None, :])            # [B,N_C,N_C]
        n_unmet = unmet.sum(dim=2).float()
        n_pred = self.pred.sum(dim=2).float()
        blocked = torch.where(n_pred > 0, n_unmet / n_pred.clamp(min=1.0),
                              torch.zeros_like(n_pred)) * open_task.float()
        ready = (open_task & (n_unmet == 0)).float()
        blocks.append(blocked); blocks.append(ready)
        # 11 communication, relative seating
        for off in range(N_P):
            q = (turn + off) % N_P
            card = self.comm_card[self.arangeB, q]; htype = self.comm_type[self.arangeB, q]
            card_oh = torch.zeros((B, N_C), device=dev)
            m = card >= 0; card_oh[self.arangeB[m], card[m]] = 1.0
            type_oh = torch.zeros((B, 3), device=dev)
            mt = htype >= 0; type_oh[self.arangeB[mt], htype[mt]] = 1.0
            valid = self.comm_valid[self.arangeB, q].float()[:, None]
            blocks.append(card_oh); blocks.append(type_oh); blocks.append(valid)
        # 12 relative hand sizes
        counts = torch.stack([(self.owner == p).sum(dim=1) for p in range(N_P)], dim=1).float()
        rel = torch.stack([counts[self.arangeB, (turn + off) % N_P] for off in range(N_P)], dim=1)
        blocks.append(rel / E.CARDS_PER_PLAYER_MAX)
        # 13 scalars (comm-phase + dist-phase flags)
        sc = torch.stack([
            self.tricks.float() / TOTAL_TRICKS,
            (turn == self.leader).float(),
            self.plays.float() / N_P,
            self.comm_phase.float(),
            self.dist_phase.float(),
        ], dim=1)
        blocks.append(sc)
        return torch.cat(blocks, dim=1)

    # ------------------------------------------------------------------- step
    def step(self, actions):
        """Claim (81..116) during distribution; pass/communicate during the comm
        round; otherwise play. Phase masks are snapshotted before any mutation."""
        active = ~self.done
        in_dist = active & self.dist_phase
        in_comm = active & self.comm_phase
        in_play = active & (~self.dist_phase) & (~self.comm_phase)

        # --- distribution: claim a task, advance claimer; exit when pool empty ---
        if bool(in_dist.any()):
            di = in_dist.nonzero(as_tuple=True)[0]
            c = actions[di] - E.CLAIM_OFFSET
            self.assigned[di, c] = self.turn[di]
            self.dist_count[di] += 1
            unclaimed = (self.is_task[di] & (self.assigned[di] == -1)).any(dim=1)
            done_dist = ~unclaimed
            self.dist_phase[di] = ~done_dist
            self.comm_phase[di] = done_dist & self.allow_comm[di]
            self.turn[di] = torch.where(done_dist, self.leader[di],
                                        (self.leader[di] + self.dist_count[di]) % N_P)

        # --- communication: record signal (or pass), advance; exit after N_P ---
        if bool(in_comm.any()):
            ctype = getattr(self, "_ctype_cache", None)
            if ctype is None:
                _, ctype = self._comm_table()
            ci = in_comm.nonzero(as_tuple=True)[0]
            real = actions[ci] < E.PASS_ACTION
            rci = ci[real]
            if rci.numel() > 0:
                rtp = self.turn[rci]; rc = actions[rci] - N_C
                self.comm_card[rci, rtp] = rc
                self.comm_type[rci, rtp] = ctype[rci, rc]
                self.comm_valid[rci, rtp] = True
            self.comm_count[ci] += 1
            done_comm = self.comm_count[ci] >= N_P
            self.comm_phase[ci] = ~done_comm
            self.turn[ci] = torch.where(done_comm, self.leader[ci],
                                        (self.leader[ci] + self.comm_count[ci]) % N_P)

        # --- play: clear consumed signal, place card, advance, maybe resolve ---
        pidx = in_play.nonzero(as_tuple=True)[0]
        if pidx.numel() > 0:
            pturn = self.turn[pidx]; pa = actions[pidx]
            clr = self.comm_card[pidx, pturn] == pa
            cci = pidx[clr]
            if cci.numel() > 0:
                ct = self.turn[cci]
                self.comm_card[cci, ct] = -1; self.comm_type[cci, ct] = -1
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
        self._ctype_cache = None
        return self

    def _resolve(self, R):
        ridx = R.nonzero(as_tuple=True)[0]
        winner = (self.leader + self._strength(self.table).argmax(dim=1)) % N_P
        wr = winner[ridx]
        for j in range(N_P):
            self.captured_by[ridx, self.table[ridx, j]] = wr

        failed_acc = torch.zeros(len(ridx), dtype=torch.bool, device=self.device)
        for j in range(N_P):
            c = self.table[ridx, j]
            asn = self.assigned[ridx, c]
            is_task = asn >= 0                       # during play, claimed == is_task
            wrong = is_task & (wr != asn)
            preds_c = self.pred[ridx, c]             # [m, N_C] predecessors of c
            unmet = (preds_c & (~self.done_tasks[ridx])).any(dim=1)
            order_bad = is_task & unmet
            can_done = is_task & (~wrong) & (~order_bad) & (~failed_acc)
            gi = can_done.nonzero(as_tuple=True)[0]
            self.done_tasks[ridx[gi], c[gi]] = True
            failed_acc = failed_acc | wrong | order_bad
        self.failed[ridx] = self.failed[ridx] | failed_acc

        self.table[ridx] = -1; self.led[ridx] = -1; self.plays[ridx] = 0
        self.leader[ridx] = wr; self.turn[ridx] = wr; self.tricks[ridx] += 1

        n_tasks = self.is_task.sum(dim=1)
        n_done = self.done_tasks.sum(dim=1)
        failed = R & self.failed
        succ = R & (~self.failed) & (n_done == n_tasks)
        timeout = R & (~self.failed) & (self.tricks >= TOTAL_TRICKS)
        self.done = self.done | failed | succ | timeout
        self.success = self.success | succ
