import chess
import pytest
import torch

from badgyal.board2planes import board2planes, bulk_board2planes

# Plane index -> (piece, square) for the all-pieces tests.
# Plane mapping (Lc0 classical): 0-5 = white P,N,B,R,Q,K; 6-11 = black P,N,B,R,Q,K.
# Squares are all distinct so each plane has exactly one set bit.
ALL_PIECES = (
    (0, chess.Piece(chess.PAWN, chess.WHITE), chess.A2),
    (1, chess.Piece(chess.KNIGHT, chess.WHITE), chess.B1),
    (2, chess.Piece(chess.BISHOP, chess.WHITE), chess.C1),
    (3, chess.Piece(chess.ROOK, chess.WHITE), chess.D1),
    (4, chess.Piece(chess.QUEEN, chess.WHITE), chess.E1),
    (5, chess.Piece(chess.KING, chess.WHITE), chess.F1),
    (6, chess.Piece(chess.PAWN, chess.BLACK), chess.A7),
    (7, chess.Piece(chess.KNIGHT, chess.BLACK), chess.B8),
    (8, chess.Piece(chess.BISHOP, chess.BLACK), chess.C8),
    (9, chess.Piece(chess.ROOK, chess.BLACK), chess.D8),
    (10, chess.Piece(chess.QUEEN, chess.BLACK), chess.E8),
    (11, chess.Piece(chess.KING, chess.BLACK), chess.F8),
)

# ---------------------------------------------------------------------------
# Castling-plane regression tests
#
# These tests pin the **current** badgyal castling-plane behavior.  The
# kingside/queenside planes (104/105 and 106/107) are **swapped** relative
# to Lc0 classical INPUT_CLASSICAL_112_PLANE.
#
# Known issue: see docs/research/112planes.md §4.1.
# Any future fix that reorders the castling planes in board2planes.py MUST
# update these tests in lockstep — these tests will fail by design when the
# swap is corrected.  They are regression (behavior-pinning) tests, not
# Lc0-correctness tests.
# ---------------------------------------------------------------------------

PLANE_OUR_KINGSIDE = 104
PLANE_OUR_QUEENSIDE = 105
PLANE_THEIR_KINGSIDE = 106
PLANE_THEIR_QUEENSIDE = 107
PLANE_SIDE_TO_MOVE = 108
PLANE_HALFMOVE_CLOCK = 109
PLANE_UNUSED = 110
PLANE_BIAS = 111


def _planes_for_fen(fen: str) -> torch.Tensor:
    """Build the 112-plane tensor for a FEN string.

    Args:
        fen: A valid chess FEN.

    Returns:
        The (112, 8, 8) tensor produced by ``board2planes`` for that position,
        with the leading batch dimension dropped.
    """
    return board2planes(chess.Board(fen))[0]


def _build_all_pieces_board(turn: chess.Color) -> chess.Board:
    """Build a board with exactly one piece of each type/color.

    Args:
        turn: Side to move.

    Returns:
        A chess.Board with all 12 piece types on distinct squares.
    """
    board = chess.Board.empty()
    for _plane, piece, square in ALL_PIECES:
        board.set_piece_at(square, piece)
    board.turn = turn
    return board


def test_board2planes_shape():
    """Test that board2planes returns a tensor of the correct shape."""
    board = chess.Board()
    planes = board2planes(board)

    assert isinstance(planes, torch.Tensor), "Output should be a torch.Tensor"
    assert planes.shape == (1, 112, 8, 8), (
        f"Expected shape (1, 112, 8, 8), got {planes.shape}"
    )


def test_board2planes_white_pawns():
    """Test that white pawns are placed correctly on the 2nd rank."""
    board = chess.Board()
    planes = board2planes(board)

    # Plane 0 is white pawns
    white_pawns = planes[0, 0, :, :]

    # Should have 8 pawns
    assert white_pawns.sum() == 8, f"Expected 8 white pawns, got {white_pawns.sum()}"

    # Pawns should be on rank 2 (row 1 in 0-indexed, since row 0 is rank 1)
    assert torch.all(white_pawns[1, :] == 1.0), (
        "White pawns should be on row 1 (rank 2)"
    )


def test_board2planes_black_pawns():
    """Test that black pawns are placed correctly on the 7th rank."""
    board = chess.Board()
    planes = board2planes(board)

    # Plane 6 is black pawns
    black_pawns = planes[0, 6, :, :]

    # Should have 8 pawns
    assert black_pawns.sum() == 8, f"Expected 8 black pawns, got {black_pawns.sum()}"

    # Rank 7 -> row = 6
    assert torch.all(black_pawns[6, :] == 1.0), (
        "Black pawns should be on row 6 (rank 7)"
    )


def test_board2planes_all_pieces_white():
    """Test all 12 piece planes (0-11) for a white-to-move position.

    Each plane should have exactly one piece at the expected (row, col).
    """
    board = _build_all_pieces_board(chess.WHITE)
    planes = board2planes(board)

    for plane, _piece, square in ALL_PIECES:
        row, col = square // 8, square % 8
        assert planes[0, plane].sum() == 1, (
            f"Plane {plane} should have 1 piece, got {planes[0, plane].sum()}"
        )
        assert planes[0, plane, row, col] == 1.0, (
            f"Plane {plane} piece should be at ({row}, {col})"
        )


def test_board2planes_all_pieces_black():
    """Test all 12 piece planes for a black-to-move (mirrored) position.

    board2planes calls board.mirror() when black is to move. mirror() swaps
    piece colors (white <-> black) and flips the board vertically (row r ->
    7-r). So a piece that was in plane p lands in plane (p+6) % 12 at row
    7-row, same col.
    """
    board = _build_all_pieces_board(chess.BLACK)
    planes = board2planes(board)

    for plane, _piece, square in ALL_PIECES:
        mirrored_plane = (plane + 6) % 12
        mirrored_row = 7 - (square // 8)
        col = square % 8
        assert planes[0, mirrored_plane].sum() == 1, (
            f"Mirrored plane {mirrored_plane} should have 1 piece, "
            f"got {planes[0, mirrored_plane].sum()}"
        )
        assert planes[0, mirrored_plane, mirrored_row, col] == 1.0, (
            f"Mirrored plane {mirrored_plane} piece should be at "
            f"({mirrored_row}, {col})"
        )

    # Sanity: exactly 12 set bits across the 12 piece planes.
    assert planes[0, :12].sum() == 12, (
        f"Expected 12 set bits in piece planes 0-11, got {planes[0, :12].sum()}"
    )


# ---------------------------------------------------------------------------
# Castling-plane tests (planes 104-107)
# Pinning current badgyal behavior; swap fix is out of scope.
# ---------------------------------------------------------------------------


def test_castling_plane_104_white_kingside_only():
    """Test plane 104 is all-1s when white kingside (K) is the only castling right.

    Pins current badgyal behavior: K maps to PLANE_OUR_KINGSIDE (104).
    See docs/research/112planes.md §4.1 — the swap vs Lc0 classical is a
    known issue; this test detects any future change to the plane order.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w K - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_OUR_KINGSIDE].sum() == 64, (
        f"Plane {PLANE_OUR_KINGSIDE} should be all 1s for white kingside-only"
    )
    assert planes[PLANE_OUR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_QUEENSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_QUEENSIDE} should be all 0s"
    )


def test_castling_plane_105_white_queenside_only():
    """Test plane 105 is all-1s when white queenside (Q) is the only castling right.

    Pins current badgyal behavior: Q maps to PLANE_OUR_QUEENSIDE (105).
    See docs/research/112planes.md §4.1 — the swap vs Lc0 classical is a
    known issue; this test detects any future change to the plane order.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Q - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_OUR_QUEENSIDE].sum() == 64, (
        f"Plane {PLANE_OUR_QUEENSIDE} should be all 1s for white queenside-only"
    )
    assert planes[PLANE_OUR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_QUEENSIDE} should be all 0s"
    )


def test_castling_plane_106_black_kingside_only():
    """Test plane 106 is all-1s when black kingside (k) is the only castling right.

    Pins current badgyal behavior: k maps to PLANE_THEIR_KINGSIDE (106)
    when white is to move.
    See docs/research/112planes.md §4.1 — the swap vs Lc0 classical is a
    known issue; this test detects any future change to the plane order.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w k - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_THEIR_KINGSIDE].sum() == 64, (
        f"Plane {PLANE_THEIR_KINGSIDE} should be all 1s for black kingside-only"
    )
    assert planes[PLANE_OUR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_OUR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_QUEENSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_QUEENSIDE} should be all 0s"
    )


def test_castling_plane_107_black_queenside_only():
    """Test plane 107 is all-1s when black queenside (q) is the only castling right.

    Pins current badgyal behavior: q maps to PLANE_THEIR_QUEENSIDE (107)
    when white is to move.
    See docs/research/112planes.md §4.1 — the swap vs Lc0 classical is a
    known issue; this test detects any future change to the plane order.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w q - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_THEIR_QUEENSIDE].sum() == 64, (
        f"Plane {PLANE_THEIR_QUEENSIDE} should be all 1s for black queenside-only"
    )
    assert planes[PLANE_OUR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_OUR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_QUEENSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_KINGSIDE} should be all 0s"
    )


def test_castling_planes_no_rights():
    """Test planes 104-107 are all-0s when no castling rights are set."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_OUR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_OUR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_QUEENSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_QUEENSIDE} should be all 0s"
    )


def test_castling_planes_all_rights():
    """Test planes 104-107 are all-1s when all four castling rights are set."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_OUR_KINGSIDE].sum() == 64, (
        f"Plane {PLANE_OUR_KINGSIDE} should be all 1s"
    )
    assert planes[PLANE_OUR_QUEENSIDE].sum() == 64, (
        f"Plane {PLANE_OUR_QUEENSIDE} should be all 1s"
    )
    assert planes[PLANE_THEIR_KINGSIDE].sum() == 64, (
        f"Plane {PLANE_THEIR_KINGSIDE} should be all 1s"
    )
    assert planes[PLANE_THEIR_QUEENSIDE].sum() == 64, (
        f"Plane {PLANE_THEIR_QUEENSIDE} should be all 1s"
    )


def test_castling_plane_104_black_to_move_kingside():
    """Test plane 104 is all-1s when black to move with only black kingside (k).

    board2planes calls board.mirror() when black is to move.  mirror() swaps
    piece colors and flips the board vertically; castling rights are also
    mirrored.  After the mirror, black's BB_H8 kingside right maps to
    PLANE_OUR_KINGSIDE (104) — empirically confirmed in
    docs/research/112planes.md §4.1.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b k - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_OUR_KINGSIDE].sum() == 64, (
        f"Plane {PLANE_OUR_KINGSIDE} should be all 1s for black-to-move kingside"
    )
    assert planes[PLANE_OUR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_QUEENSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_QUEENSIDE} should be all 0s"
    )


def test_castling_plane_105_black_to_move_queenside():
    """Test plane 105 is all-1s when black to move with only black queenside (q).

    Symmetry partner of test_castling_plane_104_black_to_move_kingside.
    After board.mirror(), black's BB_A8 queenside right maps to
    PLANE_OUR_QUEENSIDE (105) — empirically confirmed in
    docs/research/112planes.md §4.1.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b q - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_OUR_QUEENSIDE].sum() == 64, (
        f"Plane {PLANE_OUR_QUEENSIDE} should be all 1s for black-to-move queenside"
    )
    assert planes[PLANE_OUR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_OUR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_KINGSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_KINGSIDE} should be all 0s"
    )
    assert planes[PLANE_THEIR_QUEENSIDE].sum() == 0, (
        f"Plane {PLANE_THEIR_QUEENSIDE} should be all 0s"
    )


def test_side_to_move_plane_white():
    """Test plane 108 is all 0s when white is to move.

    Plane 108 = ``not board_.turn`` — ``board_.turn`` is True (white),
    so ``not board_.turn`` is False → all zeros.  Uses the original
    board's turn (not the mirrored board's), so the plane reflects the
    true side to move regardless of mirroring.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_SIDE_TO_MOVE].sum() == 0, (
        f"Plane {PLANE_SIDE_TO_MOVE} should be all 0s for white-to-move"
    )
    assert torch.all(planes[PLANE_SIDE_TO_MOVE] == 0.0), (
        f"Plane {PLANE_SIDE_TO_MOVE} should be all 0.0 for white-to-move"
    )


def test_side_to_move_plane_black():
    """Test plane 108 is all 1s when black is to move.

    Plane 108 = ``not board_.turn`` — ``board_.turn`` is False (black),
    so ``not board_.turn`` is True → all ones.  Uses the original
    board's turn (not the mirrored board's), so the plane reflects the
    true side to move regardless of mirroring.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    planes = _planes_for_fen(fen)

    assert planes[PLANE_SIDE_TO_MOVE].sum() == 64, (
        f"Plane {PLANE_SIDE_TO_MOVE} should be all 1s for black-to-move"
    )
    assert torch.all(planes[PLANE_SIDE_TO_MOVE] == 1.0), (
        f"Plane {PLANE_SIDE_TO_MOVE} should be all 1.0 for black-to-move"
    )

def test_startpos_black_to_move_mirrors():
    """Test startpos with black to move: pieces are encoded from side-to-move's perspective.

    When black is to move, ``board2planes`` calls ``board.mirror()`` so the
    112-plane tensor is always encoded from the side-to-move's viewpoint.
    After mirroring:

    - Plane 0 (our pawns) should contain **black** pawns, now on row 1
      (mirrored from rank 7 → row 1).
    - Plane 6 (their pawns) should contain **white** pawns, now on row 6
      (mirrored from rank 2 → row 6).

    This is the inverse of ``test_board2planes_white_pawns`` and
    ``test_board2planes_black_pawns`` which use white-to-move startpos.

    See: docs/research/112planes.md §1.1, §3.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    planes = _planes_for_fen(fen)  # shape: (112, 8, 8) — batch dim dropped

    # Plane 0 = our pawns (black pawns after mirror) on row 1 (rank 2)
    assert planes[0, 1, :].sum() == 8, (
        "Plane 0 (our pawns) should have 8 pawns on row 1 after mirror"
    )
    assert torch.all(planes[0, 1, :] == 1.0), (
        "Plane 0 (our pawns) should be all 1s on row 1"
    )

    # Plane 6 = their pawns (white pawns after mirror) on row 6 (rank 7)
    assert planes[6, 6, :].sum() == 8, (
        "Plane 6 (their pawns) should have 8 pawns on row 6 after mirror"
    )
    assert torch.all(planes[6, 6, :] == 1.0), (
        "Plane 6 (their pawns) should be all 1s on row 6"
    )

    # Mirror sanity: side-to-move plane should be all 1s (black to move)
    assert planes[PLANE_SIDE_TO_MOVE].sum() == 64, (
        f"Plane {PLANE_SIDE_TO_MOVE} should be all 1s for black-to-move"
    )

    # Mirror sanity: no pawns on the "wrong" rows
    assert planes[0, 6, :].sum() == 0, (
        "Plane 0 pawns should NOT be on row 6 (mirrored position)"
    )
    assert planes[6, 1, :].sum() == 0, (
        "Plane 6 pawns should NOT be on row 1 (mirrored position)"
    )


# ---------------------------------------------------------------------------
# Bias-plane test (plane 111)
# ---------------------------------------------------------------------------


_BIAS_PLANE_FENS = (
    # Standard startpos, white to move — default production path.
    chess.STARTING_FEN,
    # Standard startpos, black to move — exercises mirror() branch (line 63-64).
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    # Kings-only minimal board — baseline invariance check.
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    # Complex middlegame — varied piece types, both castling rights present.
    "r1bq1rk1/pp2nppp/2n1p3/3pP3/1b1P4/2NB1N2/PPP2PPP/R1BQ1RK1 w - - 0 9",
    # Non-zero halfmove clock — confirms bias plane is independent of move counters.
    "4k3/8/8/8/8/8/8/4K3 w - - 25 10",
)


def test_board2planes_constant_ones_plane():
    """Test that plane 111 is all 1s for any position (Lc0 bias plane).

    Plane 111 is appended unconditionally as a constant ones plane in
    board2planes.py:93 (the final ``append_plane(retval, True)``).  It must
    not depend on piece placement, side to move, castling rights, en-passant
    state, or move counters.  See docs/research/112planes.md.
    """
    for fen in _BIAS_PLANE_FENS:
        planes = _planes_for_fen(fen)
        assert torch.all(planes[PLANE_BIAS] == 1.0), (
            f"Plane 111 should be all 1s for FEN {fen!r}"
        )
        assert planes[PLANE_BIAS].sum().item() == 64, (
            f"Plane 111 should have 64 set cells for FEN {fen!r}, "
            f"got {int(planes[PLANE_BIAS].sum().item())}"
        )


# ---------------------------------------------------------------------------
# Zeroed-plane tests (planes 109 halfmove clock, 110 unused)
#
# These tests pin the **current** badgyal behavior: planes 109 and 110 are
# deliberately forced to zero in ``board2planes.py`` (lines 90, 92).  Plane
# 109 is the Lc0 classical halfmove clock (rule-50); the real
# ``board_.halfmove_clock`` fill is commented out with the note
# "# half-move clock goes to zero".  Plane 110 is unused.
#
# A future ticket will implement the real halfmove clock in plane 109
# (Lc0 classical).  When that lands, this test MUST be updated to assert
# the real value instead of zero — these are behavior-pinning (regression)
# tests, not Lc0-correctness tests.
# See: docs/research/112planes.md §4.2
# ---------------------------------------------------------------------------


_ZEROED_PLANE_FENS = (
    # Standard startpos, white to move — halfmove clock is 0.
    chess.STARTING_FEN,
    # Halfmove clock = 2 — proves zeroing even when the real clock is non-zero.
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    # Halfmove clock = 20 — larger value, same expectation.
    "8/8/8/8/8/8/8/k1K5 w - - 20 30",
    # Black to move with non-zero clock — exercises the mirror() branch to
    # confirm the zeroing is independent of side-to-move / mirroring.
    "8/8/8/8/8/8/8/k1K5 b - - 20 30",
)


def test_board2planes_zeroed_planes():
    """Test that planes 109 (halfmove clock) and 110 (unused) are always zeroed.

    ``board2planes.py`` deliberately forces both planes to zero (lines 90, 92).
    Plane 109 is the Lc0 classical halfmove clock (rule-50); the real
    ``board_.halfmove_clock`` fill is commented out with the note
    "# half-move clock goes to zero".  Plane 110 is unused.

    This test pins the current always-zero behavior so that a future
    implementation of the real halfmove clock is **intentional and detected**.
    When the real clock is implemented, this test must be updated to assert
    the real value instead of zero.

    See: docs/research/112planes.md §4.2
    """
    for fen in _ZEROED_PLANE_FENS:
        planes = _planes_for_fen(fen)
        assert torch.all(planes[PLANE_HALFMOVE_CLOCK] == 0.0), (
            f"Plane {PLANE_HALFMOVE_CLOCK} (halfmove clock) is currently always "
            f"zeroed (FEN: {fen!r})"
        )
        assert planes[PLANE_HALFMOVE_CLOCK].sum() == 0, (
            f"Plane {PLANE_HALFMOVE_CLOCK} sum should be 0 (FEN: {fen!r})"
        )
        assert torch.all(planes[PLANE_UNUSED] == 0.0), (
            f"Plane {PLANE_UNUSED} (unused) is always zeroed (FEN: {fen!r})"
        )
        assert planes[PLANE_UNUSED].sum() == 0, (
            f"Plane {PLANE_UNUSED} sum should be 0 (FEN: {fen!r})"
        )


# ---------------------------------------------------------------------------
# bulk_board2planes tests
#
# These tests pin the shape, dtype, contiguity, and per-element equivalence
# contracts of ``bulk_board2planes``.  No production code changes are needed
# — the function already exists and works correctly.
# ---------------------------------------------------------------------------


def test_bulk_board2planes_shape():
    """Test that bulk_board2planes returns a tensor of the correct shape.

    Args:
        boards: List of chess.Board instances (3 identical startpos boards).

    Returns:
        None. Asserts shape, dtype, and contiguity.
    """
    boards = [chess.Board(), chess.Board(), chess.Board()]
    planes = bulk_board2planes(boards)

    assert planes.shape == (3, 112, 8, 8), (
        f"Expected shape (3, 112, 8, 8), got {planes.shape}"
    )
    assert planes.dtype == torch.float32, (
        f"Expected dtype torch.float32, got {planes.dtype}"
    )
    assert isinstance(planes, torch.Tensor), (
        f"Expected torch.Tensor, got {type(planes)}"
    )
    assert planes.is_contiguous(), "bulk_board2planes output must be contiguous"


def test_bulk_board2planes_matches_individual():
    """Test that bulk_board2planes matches individual board2planes calls.

    Uses 3 distinct FENs including one black-to-move to exercise the
    mirror branch inside board2planes.

    Args:
        None. Builds boards from FEN strings.

    Returns:
        None. Asserts full-tensor and per-element equality.
    """
    fens = [
        chess.STARTING_FEN,  # startpos white-to-move
        "rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # midgame WTM
        "8/8/8/4k3/8/8/4K3/R6r b - - 0 1",  # endgame BTM (mirror branch)
    ]
    boards = [chess.Board(fen) for fen in fens]

    bulk = bulk_board2planes(boards)
    expected = torch.cat([board2planes(b) for b in boards], dim=0)

    # Full-tensor check (belt-and-suspenders)
    assert torch.equal(bulk, expected), (
        "Bulk output must equal individual concatenation (full tensor)"
    )

    # Per-element checks (ticket Step 2 requirement)
    for i in range(len(boards)):
        assert torch.equal(bulk[i], board2planes(boards[i])[0]), (
            f"bulk[{i}] must equal board2planes(boards[{i}])[0]"
        )


def test_bulk_board2planes_single_board():
    """Test bulk_board2planes with a single board (N=1).

    Uses a black-to-move FEN to exercise the mirror branch in the N=1 path.

    Args:
        None. Builds a single board from FEN.

    Returns:
        None. Asserts shape, dtype, and per-element equality.
    """
    fen = "8/8/8/4k3/8/8/4K3/R6r b - - 0 1"
    board = chess.Board(fen)

    bulk = bulk_board2planes([board])

    assert bulk.shape == (1, 112, 8, 8), (
        f"Expected shape (1, 112, 8, 8), got {bulk.shape}"
    )
    assert bulk.dtype == torch.float32, (
        f"Expected dtype torch.float32, got {bulk.dtype}"
    )
    assert torch.equal(bulk[0], board2planes(board)[0]), (
        "Single-board bulk output must equal board2planes(board)[0]"
    )


def test_bulk_board2planes_empty_list_raises():
    """Test that bulk_board2planes raises RuntimeError for an empty list.

    Documents the current behavior: torch.cat((), dim=0) raises RuntimeError.

    Args:
        None.

    Returns:
        None. Asserts that RuntimeError is raised.
    """
    with pytest.raises(RuntimeError):
        bulk_board2planes([])


if __name__ == "__main__":
    test_board2planes_shape()
    test_board2planes_white_pawns()
    test_board2planes_black_pawns()
    test_board2planes_all_pieces_white()
    test_board2planes_all_pieces_black()
    test_castling_plane_104_white_kingside_only()
    test_castling_plane_105_white_queenside_only()
    test_castling_plane_106_black_kingside_only()
    test_castling_plane_107_black_queenside_only()
    test_castling_planes_no_rights()
    test_castling_planes_all_rights()
    test_castling_plane_104_black_to_move_kingside()
    test_castling_plane_105_black_to_move_queenside()
    test_side_to_move_plane_white()
    test_side_to_move_plane_black()
    test_startpos_black_to_move_mirrors()
    test_bulk_board2planes_shape()
    test_bulk_board2planes_matches_individual()
    test_bulk_board2planes_single_board()
    test_bulk_board2planes_empty_list_raises()
    test_board2planes_constant_ones_plane()
    test_board2planes_zeroed_planes()
    print("All tests passed!")
