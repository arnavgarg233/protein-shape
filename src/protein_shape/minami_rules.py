"""Independent implementation of the three beta-sheet topology rules of Minami et al. (Nat. Struct.
Mol. Biol. 30, 1132-1140, 2023), written from the published rule statements: the connection
jump-distance rule, the connection overlap rule and the connection ending rule. Entry point: evaluate().
"""

from itertools import combinations

__all__ = ["evaluate", "Connection"]

# Thresholds quoted in the rule statement ("less than four", "less than two").
_MAX_PARA_JUMP = 3
_MAX_ANTI_JUMP = 1


class Connection(object):
    """One beta-X-beta motif, i.e. one sequence-consecutive strand pair.

    A connection leaves the C-terminal end of its first strand and
    enters the N-terminal end of its second.  For a parallel pair those
    two points sit on opposite edges of the sheet, so the connection
    runs across the face as a diagonal; for an antiparallel pair they
    sit on the same edge, so it stays there as an arch.  ``edge`` is the
    edge an antiparallel connection hangs from, written as the
    orientation of its first strand, and is meaningless for a parallel
    one.
    """

    __slots__ = ("first", "left", "right", "jump", "parallel", "face",
                 "edge")

    def __init__(self, first, start, end, parallel, face, edge):
        self.first = first            # sequence index of the earlier strand
        self.left = min(start, end)   # lower sheet position spanned
        self.right = max(start, end)  # higher sheet position spanned
        self.jump = self.right - self.left - 1
        self.parallel = parallel
        self.face = face              # +1 / -1
        self.edge = edge              # +1 / -1 (antiparallel only)

    def nested_in(self, other):
        """True when one connection reaches right across the other."""
        return ((self.left < other.left and other.right < self.right) or
                (other.left < self.left and self.right < other.right))

    def spans_overlap(self, other):
        """True when the two connections cover a common stretch of sheet.

        Sharing a single strand (one connection ending where the other
        begins) is not an overlap: the loops meet at the two ends of
        that strand rather than over the same piece of sheet.
        """
        return max(self.left, other.left) < min(self.right, other.right)


def _unpack(order, signs):
    count = len(order)
    if len(signs) != count:
        raise ValueError("order and signs must have the same length")
    seat = [0] * (count + 1)          # sequence index -> sheet position
    for seat_index, strand in enumerate(order):
        seat[strand] = seat_index
    polarity = [0] * (count + 1)      # sequence index -> +1 / -1
    for seat_index, strand in enumerate(order):
        mark = signs[seat_index]
        polarity[strand] = 1 if mark in ("+", 1, "1", True) else -1
    return count, seat, polarity


def _connections(count, seat, polarity):
    found = []
    for strand in range(1, count):
        start, end = seat[strand], seat[strand + 1]
        parallel = polarity[strand] == polarity[strand + 1]
        step = 1 if end > start else -1
        face = -(polarity[strand] * step)
        found.append(Connection(strand, start, end, parallel, face,
                                polarity[strand]))
    return found


def _greek_key_excused(link, order, seat, polarity):
    """Is this anti motif the long connection of a Greek key?

    The Greek key places four strands on four neighbouring sheet
    positions so that, read in sequence order, they step to a neighbour,
    to a neighbour, and once right across all four: one hop of three
    positions and two hops of one.  The long hop is the connection being
    tested here, and the three circular permutations of the Greek key
    just move it to a different place in the quartet.  Taking the four
    strands in their sequence order (rather than insisting they be
    consecutive in sequence) is what makes the exception carry over from
    the four-stranded sheets of Fig. 3d to larger ones, where other
    strands of the chain may be threaded through the quartet.  The
    strands of a Greek key alternate in orientation across the sheet.
    """
    low, high = link.left, link.right
    spanned = sorted(order[p] for p in range(low, high + 1))
    hops = [abs(seat[spanned[k + 1]] - seat[spanned[k]]) for k in range(3)]
    if sorted(hops) != [1, 1, 3]:
        return False
    for position in range(low, high):
        if polarity[order[position]] == polarity[order[position + 1]]:
            return False
    return True


def evaluate(order, signs):
    """Apply the three Minami rules to one beta-sheet topology.

    Parameters
    ----------
    order : sequence of int
        ``order[k]`` is the sequence index (1-based, 1 = N-terminal) of
        the strand at sheet position ``k``.
    signs : sequence of str
        ``signs[k]`` is ``'+'`` or ``'-'``, the orientation of the strand
        at sheet position ``k`` relative to the strand at position 0.

    Returns
    -------
    dict with the three rule flags (``jump``, ``overlap``, ``ending``),
    the number of times each fired, and the three-way split of the
    overlap count by motif pair type.
    """
    order = list(order)
    count, seat, polarity = _unpack(order, signs)
    links = _connections(count, seat, polarity)

    jump_hits = 0
    for link in links:
        limit = _MAX_PARA_JUMP if link.parallel else _MAX_ANTI_JUMP
        if link.jump <= limit:
            continue
        if (not link.parallel) and link.jump == 2 and \
                _greek_key_excused(link, order, seat, polarity):
            continue
        jump_hits += 1

    # Only connections whose face is fixed by a measured preference take
    # part: every parallel one, by Richardson's rule, and antiparallel
    # ones with a single intervening strand, by Extended Data Fig. 1.
    # Antiparallel hairpins and longer antiparallel connections showed no
    # bending preference, so they cannot be called S-type or D-type.
    placed = [link for link in links if link.parallel or link.jump == 1]

    overlap_pp = overlap_aa = overlap_pa = 0
    for one, two in combinations(placed, 2):
        if one.face != two.face:
            continue
        if one.parallel and two.parallel:
            # Two parallel crossovers on one face run as parallel
            # diagonals, so they meet only where one reaches right
            # across the other.
            if one.nested_in(two):
                overlap_pp += 1
        elif one.parallel or two.parallel:
            # A diagonal and an arch meet wherever they cover a common
            # stretch of the sheet.
            if one.spans_overlap(two):
                overlap_pa += 1
        else:
            # Two arches hanging from the same edge of the same face
            # compete for the same strip of surface.
            if one.edge == two.edge:
                overlap_aa += 1
    overlap_hits = overlap_pp + overlap_aa + overlap_pa

    para_links = [link for link in links if link.parallel]
    ending_hits = 0
    for one, two in combinations(para_links, 2):
        tail_one = seat[one.first + 1]
        tail_two = seat[two.first + 1]
        if abs(tail_one - tail_two) != 1:
            continue
        if polarity[one.first + 1] != polarity[two.first + 1]:
            continue
        if one.face != two.face:
            ending_hits += 1

    return {
        "jump": jump_hits > 0,
        "overlap": overlap_hits > 0,
        "ending": ending_hits > 0,
        "jump_count": jump_hits,
        "overlap_count": overlap_hits,
        "ending_count": ending_hits,
        "overlap_para_para": overlap_pp,
        "overlap_anti_anti": overlap_aa,
        "overlap_para_anti": overlap_pa,
    }
