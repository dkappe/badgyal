import chess
import numpy as np
import torch
from time import time
from math import exp
from badgyal.policy_index import policy_index
import re

MOVE_MAP = dict(list(zip(policy_index, range(len(policy_index)))))

WPAWN = chess.Piece(chess.PAWN, chess.WHITE)
WKNIGHT = chess.Piece(chess.KNIGHT, chess.WHITE)
WBISHOP = chess.Piece(chess.BISHOP, chess.WHITE)
WROOK = chess.Piece(chess.ROOK, chess.WHITE)
WQUEEN = chess.Piece(chess.QUEEN, chess.WHITE)
WKING = chess.Piece(chess.KING, chess.WHITE)
BPAWN = chess.Piece(chess.PAWN, chess.BLACK)
BKNIGHT = chess.Piece(chess.KNIGHT, chess.BLACK)
BBISHOP = chess.Piece(chess.BISHOP, chess.BLACK)
BROOK = chess.Piece(chess.ROOK, chess.BLACK)
BQUEEN = chess.Piece(chess.QUEEN, chess.BLACK)
BKING = chess.Piece(chess.KING, chess.BLACK)

# Standard starting position board for startpos detection.
_STARTPOS = chess.Board()


def assign_piece(planes, piece_step, row, col):
    planes[piece_step][row][col] = 1


DISPATCH = {}

DISPATCH[str(WPAWN)] = lambda retval, row, col: assign_piece(retval, 0, row, col)
DISPATCH[str(WKNIGHT)] = lambda retval, row, col: assign_piece(retval, 1, row, col)
DISPATCH[str(WBISHOP)] = lambda retval, row, col: assign_piece(retval, 2, row, col)
DISPATCH[str(WROOK)] = lambda retval, row, col: assign_piece(retval, 3, row, col)
DISPATCH[str(WQUEEN)] = lambda retval, row, col: assign_piece(retval, 4, row, col)
DISPATCH[str(WKING)] = lambda retval, row, col: assign_piece(retval, 5, row, col)
DISPATCH[str(BPAWN)] = lambda retval, row, col: assign_piece(retval, 6, row, col)
DISPATCH[str(BKNIGHT)] = lambda retval, row, col: assign_piece(retval, 7, row, col)
DISPATCH[str(BBISHOP)] = lambda retval, row, col: assign_piece(retval, 8, row, col)
DISPATCH[str(BROOK)] = lambda retval, row, col: assign_piece(retval, 9, row, col)
DISPATCH[str(BQUEEN)] = lambda retval, row, col: assign_piece(retval, 10, row, col)
DISPATCH[str(BKING)] = lambda retval, row, col: assign_piece(retval, 11, row, col)

MOVE_RE = re.compile(r"^([a-h])(\d)([a-h])(\d)(.*)$")


def dump(planes):
    for i in range(112):
        print(i)
        print(planes[0][i])


def mirrorMoveUCI(move):
    m = MOVE_RE.match(move)
    return "{}{}{}{}{}".format(
        m.group(1), 9 - int(m.group(2)), m.group(3), 9 - int(m.group(4)), m.group(5)
    )


def mirrorMove(move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion,
    )


def append_plane(planes, ones):
    if ones:
        return np.append(planes, np.ones((1, 8, 8), dtype=float), axis=0)
    else:
        return np.append(planes, np.zeros((1, 8, 8), dtype=float), axis=0)


def _undo_en_passant(block: np.ndarray, board: chess.Board) -> np.ndarray:
    """Reconstruct the pre-push board for an e.p. position.

    When Lc0 synthesizes historical blocks and the current position has an
    en-passant target square, it removes the just-pushed pawn from the
    e.p. destination and places it back on its pre-push square.

    After ``board2planes`` applies ``chess.Board.mirror()`` for
    black-to-move originals, the internal board is always white-to-move
    with the e.p. square at row 5 and the pushed pawn on plane 6 at row 4.
    The undo simply moves that pawn from row 4 to row 6.

    Args:
        block: A 13×8×8 plane block.
        board: The (already mirrored if needed) board used to build *block*.

    Returns:
        A new 13×8×8 block with the e.p. undo applied.
    """
    ep_square = board.ep_square
    if ep_square is None:
        return block

    block = np.copy(block)
    ep_row, ep_col = ep_square // 8, ep_square % 8

    # After mirroring in board2planes:
    #   - e.p. square is always at row 5 (6th rank)
    #   - pushed pawn is always on plane 6 at row 4
    #   - pre-push square is always at row 6
    block[6, ep_row - 1, ep_col] = 0.0
    block[6, ep_row + 1, ep_col] = 1.0
    return block


def board2planes(board_: chess.Board) -> torch.Tensor:
    """Convert a chess.Board to the 112-plane Lc0 input tensor.

    The output tensor has shape (1, 112, 8, 8).  Planes 0-11 encode the
    current position (one plane per piece type per color), planes 12-103
    are historical blocks (``--history-fill=fen_only``), and planes 104-111
    are auxiliary planes (castling rights, side-to-move, etc.).

    For the standard starting position planes 13-103 are all-zero.
    For positions with an en-passant target square the historical blocks
    undo the just-pushed pawn (moving it from its post-push square back
    to its pre-push square).

    For black-to-move positions the board is mirrored via
    ``chess.Board.mirror()`` so the internal representation is always
    white-to-move.

    Args:
        board_: The chess position to encode.

    Returns:
        A float tensor of shape (1, 112, 8, 8).
    """
    if not board_.turn:
        board = board_.mirror()
    else:
        board = board_

    retval = np.zeros((13, 8, 8), dtype=float)
    for row in range(8):
        for col in range(8):
            piece = str(board.piece_at(chess.SQUARES[row * 8 + col]))
            if piece != "None":
                DISPATCH[piece](retval, row, col)

    # Lc0 --history-fill=fen_only: startpos has no meaningful prior
    # history at ply 0; planes 13..103 are zeros.
    if board == _STARTPOS:
        zero_block = np.zeros((13, 8, 8), dtype=float)
        for _i in range(7):
            retval = np.append(retval, zero_block, axis=0)
    else:
        temp = np.copy(retval)
        # Undo the en-passant pawn double-push once, then copy the
        # resulting block 7 times into planes 13-103.
        if board.ep_square is not None:
            temp = _undo_en_passant(temp, board)
        for _i in range(7):
            retval = np.append(retval, temp, axis=0)

    retval = append_plane(retval, bool(board.castling_rights & chess.BB_A1))
    retval = append_plane(retval, bool(board.castling_rights & chess.BB_H1))
    retval = append_plane(retval, bool(board.castling_rights & chess.BB_A8))
    retval = append_plane(retval, bool(board.castling_rights & chess.BB_H8))
    retval = append_plane(retval, not board_.turn)

    # half-move clock goes to zero
    retval = append_plane(retval, False)

    retval = append_plane(retval, False)
    retval = append_plane(retval, True)
    return torch.from_numpy(np.expand_dims(retval, axis=0)).float()


def board2planes_np(board_: chess.Board) -> np.ndarray:
    """Numpy variant of ``board2planes`` returning a (1, 112, 8, 8) ndarray.

    Identical encoding logic, but returns a numpy float32 array so callers
    that do not depend on PyTorch (e.g. an ONNX-runtime backend) can use it.

    Args:
        board_: The chess position to encode.

    Returns:
        A float32 ndarray of shape (1, 112, 8, 8).
    """
    if not board_.turn:
        board = board_.mirror()
    else:
        board = board_

    retval = np.zeros((13, 8, 8), dtype=np.float32)
    for row in range(8):
        for col in range(8):
            piece = str(board.piece_at(chess.SQUARES[row * 8 + col]))
            if piece != "None":
                DISPATCH[piece](retval, row, col)

    if board == _STARTPOS:
        zero_block = np.zeros((13, 8, 8), dtype=np.float32)
        for _i in range(7):
            retval = np.append(retval, zero_block, axis=0)
    else:
        temp = np.copy(retval)
        if board.ep_square is not None:
            temp = _undo_en_passant(temp, board)
        for _i in range(7):
            retval = np.append(retval, temp, axis=0)

    retval = append_plane(retval, bool(board.castling_rights & chess.BB_A1))
    retval = append_plane(retval, bool(board.castling_rights & chess.BB_H1))
    retval = append_plane(retval, bool(board.castling_rights & chess.BB_A8))
    retval = append_plane(retval, bool(board.castling_rights & chess.BB_H8))
    retval = append_plane(retval, not board_.turn)
    retval = append_plane(retval, False)
    retval = append_plane(retval, False)
    retval = append_plane(retval, True)
    return np.expand_dims(retval, axis=0).astype(np.float32)


def bulk_board2planes_np(boards) -> np.ndarray:
    """Numpy variant of ``bulk_board2planes``.

    Args:
        boards: Iterable of ``chess.Board``.

    Returns:
        A float32 ndarray of shape (N, 112, 8, 8).
    """
    planes = [board2planes_np(b) for b in boards]
    return np.concatenate(planes, axis=0).astype(np.float32)


def bulk_board2planes(boards):
    planes = []
    for b in boards:
        temp = board2planes(b)
        planes.append(temp)
    pl = tuple(planes)
    retval = torch.cat(pl, dim=0).contiguous()
    return retval


def policy2moves(board_, policy_tensor, softmax_temp=1.61):
    if not board_.turn:
        board = board_.mirror()
    else:
        board = board_
    policy = policy_tensor.numpy()

    moves = list(board.legal_moves)
    retval = {}
    max_p = float("-inf")
    for m in moves:
        uci = m.uci()
        fixed_uci = uci
        # piece = str(board.piece_at(m.from_square))
        # fix the uci
        # if (piece == 'K') and (uci == "e1g1"):
        #    fixed_uci = "e1h1"
        # if (piece == 'K') and (uci == "e1c1"):
        #    fixed_uci = "e1a1"
        if (uci == "e1g1") and board.is_kingside_castling(m):
            fixed_uci = "e1h1"
        elif (uci == "e1c1") and board.is_queenside_castling(m):
            fixed_uci = "e1a1"
        if uci[-1] == "n":
            # we are promoting to knight, so trim the character
            fixed_uci = uci[0:-1]
        # now mirror the uci
        if not board_.turn:
            uci = mirrorMoveUCI(uci)
        p = policy[0][MOVE_MAP[fixed_uci]]
        retval[uci] = p
        if p > max_p:
            max_p = p
    total = 0.0
    for uci in retval:
        retval[uci] = exp((retval[uci] - max_p) / softmax_temp)
        total = total + retval[uci]

    if total > 0.0:
        for uci in retval:
            retval[uci] = retval[uci] / total
    return retval


def policy2moves_np(board_, policy: np.ndarray, softmax_temp: float = 1.61):
    """Numpy variant of ``policy2moves`` accepting a raw ndarray.

    Args:
        board_: The chess position (same perspective conventions as
            ``policy2moves``).
        policy: A 1-D ndarray of length 1858 (the policy logits for one
            position) or a 2-D array of shape (1, 1858).
        softmax_temp: Temperature for the softmax normalization.

    Returns:
        Dict mapping UCI move strings to normalized probabilities.
    """
    if not board_.turn:
        board = board_.mirror()
    else:
        board = board_
    policy = np.asarray(policy).reshape(-1)

    moves = list(board.legal_moves)
    retval = {}
    max_p = float("-inf")
    for m in moves:
        uci = m.uci()
        fixed_uci = uci
        if (uci == "e1g1") and board.is_kingside_castling(m):
            fixed_uci = "e1h1"
        elif (uci == "e1c1") and board.is_queenside_castling(m):
            fixed_uci = "e1a1"
        if uci[-1] == "n":
            fixed_uci = uci[0:-1]
        if not board_.turn:
            uci = mirrorMoveUCI(uci)
        p = policy[MOVE_MAP[fixed_uci]]
        retval[uci] = p
        if p > max_p:
            max_p = p
    total = 0.0
    for uci in retval:
        retval[uci] = exp((retval[uci] - max_p) / softmax_temp)
        total = total + retval[uci]

    if total > 0.0:
        for uci in retval:
            retval[uci] = retval[uci] / total
    return retval


if __name__ == "__main__":
    print(MOVE_MAP)

    board = chess.Board(
        fen="rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    )

    start = time()
    REPS = 1

    for i in range(0, REPS):
        planes = board2planes(board)

    end = time()

    print(end - start)
    print((end - start) / REPS)
    print(planes.shape)
    # dump(planes)
