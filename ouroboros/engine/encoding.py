"""Board -> tensor encoding and move <-> policy-index mapping.

AlphaZero-style: 8x8x19 input planes, 8x8x73 policy space (4672 total).
All indices are in the side-to-move frame (board flipped if Black to move).
"""
import chess
import numpy as np
import torch

# Optional native acceleration (built from rust_ext/). The pure-Python
# implementations below remain the source of truth and the guaranteed fallback;
# the native module only replaces the hot inner loops and is validated to
# produce byte-identical output. If it is missing (e.g. the Rust build was
# skipped), everything still works.
try:
    import ouroboros_native as _native
    HAS_NATIVE = True
except Exception:  # pragma: no cover - depends on build environment
    _native = None
    HAS_NATIVE = False

# -- Policy index layout (per from-square, 73 planes) --------------------------
# [0..55]  queen-style: 8 directions x 7 distances
# [56..63] knight moves: 8 deltas
# [64..72] underpromotions: 3 pieces x 3 directions (fwd, cap-L, cap-R)
#          pieces: N=0, B=1, R=2

_QUEEN_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]   # N,S,W,E,NW,NE,SW,SE
_KNIGHT_DELTAS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]

POLICY_SIZE = 4672  # 64 x 73


def _queen_plane(direction_idx: int, distance: int) -> int:
    """direction_idx in [0,7], distance in [1,7] -> plane in [0,55]"""
    return direction_idx * 7 + (distance - 1)


def _knight_plane(delta_idx: int) -> int:
    return 56 + delta_idx


def _underprom_plane(piece_idx: int, direction: int) -> int:
    """piece_idx: N=0,B=1,R=2; direction: fwd=0, cap-left=1, cap-right=2"""
    return 64 + piece_idx * 3 + direction


def _sq(rank: int, col: int) -> int:
    return rank * 8 + col


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Map a legal move to a policy index [0, 4671].

    Always uses the side-to-move frame: if Black to move the board is viewed
    as if flipped (rank 0 <-> rank 7), so the side to move always attacks
    'upward' (increasing rank).
    """
    if HAS_NATIVE:
        idx = _native.move_to_index(
            move.from_square, move.to_square, move.promotion or 0,
            board.turn == chess.BLACK,
        )
        if idx < 0:
            raise ValueError(f"Cannot encode move {move} on board\n{board}")
        return idx
    return _py_move_to_index(board, move)


def _py_move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Pure-Python reference for move_to_index (fallback + source of truth)."""
    flip = board.turn == chess.BLACK

    from_sq = move.from_square
    to_sq = move.to_square

    if flip:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    from_rank, from_col = divmod(from_sq, 8)
    to_rank, to_col = divmod(to_sq, 8)
    dr = to_rank - from_rank
    dc = to_col - from_col

    promotion = move.promotion

    # Underpromotion: any promotion that is NOT queen
    if promotion is not None and promotion != chess.QUEEN:
        piece_map = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
        piece_idx = piece_map[promotion]
        if dc == 0:
            direction = 0  # forward
        elif dc == -1:
            direction = 1  # capture-left
        else:
            direction = 2  # capture-right
        plane = _underprom_plane(piece_idx, direction)
        return from_sq * 73 + plane

    # Knight move
    delta = (dr, dc)
    if delta in _KNIGHT_DELTAS:
        plane = _knight_plane(_KNIGHT_DELTAS.index(delta))
        return from_sq * 73 + plane

    # Queen-style move (includes queen promotion)
    dist = max(abs(dr), abs(dc))
    dir_vec = (dr // dist if dr != 0 else 0, dc // dist if dc != 0 else 0)
    if dir_vec in _QUEEN_DIRS:
        dir_idx = _QUEEN_DIRS.index(dir_vec)
        plane = _queen_plane(dir_idx, dist)
        return from_sq * 73 + plane

    raise ValueError(f"Cannot encode move {move} on board\n{board}")


def index_to_move(board: chess.Board, idx: int) -> chess.Move:
    """Map a policy index back to a chess.Move (in board coordinates).

    Returns the move if it is legal, else raises ValueError.
    """
    flip = board.turn == chess.BLACK
    from_sq_frame = idx // 73
    plane = idx % 73

    if flip:
        from_sq = chess.square_mirror(from_sq_frame)
    else:
        from_sq = from_sq_frame

    from_rank_f, from_col_f = divmod(from_sq_frame, 8)

    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        ddr, ddc = _QUEEN_DIRS[dir_idx]
        to_rank_f = from_rank_f + ddr * dist
        to_col_f = from_col_f + ddc * dist
        if not (0 <= to_rank_f < 8 and 0 <= to_col_f < 8):
            raise ValueError(f"idx {idx} leads off-board")
        to_sq_frame = _sq(to_rank_f, to_col_f)
        to_sq = chess.square_mirror(to_sq_frame) if flip else to_sq_frame
        # Auto queen-promote if pawn reaches last rank
        promotion = None
        piece = board.piece_at(from_sq)
        if (piece and piece.piece_type == chess.PAWN and
                chess.square_rank(to_sq) in (0, 7)):
            promotion = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promotion)

    if plane < 64:
        delta_idx = plane - 56
        ddr, ddc = _KNIGHT_DELTAS[delta_idx]
        to_rank_f = from_rank_f + ddr
        to_col_f = from_col_f + ddc
        if not (0 <= to_rank_f < 8 and 0 <= to_col_f < 8):
            raise ValueError(f"idx {idx} knight off-board")
        to_sq_frame = _sq(to_rank_f, to_col_f)
        to_sq = chess.square_mirror(to_sq_frame) if flip else to_sq_frame
        return chess.Move(from_sq, to_sq)

    # Underpromotion
    up_idx = plane - 64
    piece_idx = up_idx // 3
    direction = up_idx % 3
    piece_map = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    promotion = piece_map[piece_idx]
    ddc = [0, -1, 1][direction]
    # Pawns always promote going forward (rank+1 in frame)
    to_rank_f = from_rank_f + 1
    to_col_f = from_col_f + ddc
    if not (0 <= to_rank_f < 8 and 0 <= to_col_f < 8):
        raise ValueError(f"idx {idx} underprom off-board")
    to_sq_frame = _sq(to_rank_f, to_col_f)
    to_sq = chess.square_mirror(to_sq_frame) if flip else to_sq_frame
    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a bool mask of shape (4672,) with True at legal move indices."""
    if HAS_NATIVE:
        froms: list[int] = []
        tos: list[int] = []
        promos: list[int] = []
        for move in board.legal_moves:
            froms.append(move.from_square)
            tos.append(move.to_square)
            promos.append(move.promotion or 0)
        mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
        if froms:
            idxs = _native.legal_indices(froms, tos, promos, board.turn == chess.BLACK)
            for idx in idxs:
                if idx >= 0:
                    mask[idx] = True
        return mask
    return _py_legal_move_mask(board)


def _py_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Pure-Python reference for legal_move_mask (fallback + source of truth)."""
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        try:
            idx = _py_move_to_index(board, move)
            mask[idx] = True
        except ValueError:
            pass
    return mask


# -- Board -> tensor planes ----------------------------------------------------
_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Return float32 tensor of shape (19, 8, 8).

    Always from the side-to-move's perspective.
    """
    if HAS_NATIVE:
        return _native_board_to_tensor(board)
    return _py_board_to_tensor(board)


def _native_board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Native-backed board encoding. python-chess supplies the bitboards and
    rule-dependent flags; the native module fills the plane buffer."""
    own_color = board.turn
    opp_color = not own_color
    flip = board.turn == chess.BLACK

    own = [board.pieces_mask(pt, own_color) for pt in _PIECE_ORDER]
    opp = [board.pieces_mask(pt, opp_color) for pt in _PIECE_ORDER]

    cr = board.castling_rights
    if board.turn == chess.WHITE:
        castle = [bool(cr & chess.BB_H1), bool(cr & chess.BB_A1),
                  bool(cr & chess.BB_H8), bool(cr & chess.BB_A8)]
    else:
        castle = [bool(cr & chess.BB_H8), bool(cr & chess.BB_A8),
                  bool(cr & chess.BB_H1), bool(cr & chess.BB_A1)]

    ep_file = chess.square_file(board.ep_square) if board.ep_square is not None else -1
    halfmove = min(board.halfmove_clock / 100.0, 1.0)
    repetition = board.is_repetition(2)

    buf = _native.board_to_planes(
        own, opp, castle, ep_file, float(halfmove), repetition, flip,
    )
    planes = np.frombuffer(buf, dtype="<f4").reshape(19, 8, 8).copy()
    return torch.from_numpy(planes)


def _py_board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Pure-Python reference for board_to_tensor (fallback + source of truth)."""
    flip = board.turn == chess.BLACK
    planes = np.zeros((19, 8, 8), dtype=np.float32)

    def sq2rc(sq: int) -> tuple[int, int]:
        r, c = divmod(sq, 8)
        if flip:
            r = 7 - r
        return r, c

    # Planes 0-5: own pieces; 6-11: opponent pieces
    own_color = board.turn
    opp_color = not own_color
    for i, pt in enumerate(_PIECE_ORDER):
        for sq in board.pieces(pt, own_color):
            r, c = sq2rc(sq)
            planes[i, r, c] = 1.0
        for sq in board.pieces(pt, opp_color):
            r, c = sq2rc(sq)
            planes[6 + i, r, c] = 1.0

    # Castling rights (planes 12-15)
    cr = board.castling_rights
    if board.turn == chess.WHITE:
        if cr & chess.BB_H1:
            planes[12, :, :] = 1.0  # own K-side
        if cr & chess.BB_A1:
            planes[13, :, :] = 1.0  # own Q-side
        if cr & chess.BB_H8:
            planes[14, :, :] = 1.0  # opp K-side
        if cr & chess.BB_A8:
            planes[15, :, :] = 1.0  # opp Q-side
    else:
        if cr & chess.BB_H8:
            planes[12, :, :] = 1.0  # own K-side (Black)
        if cr & chess.BB_A8:
            planes[13, :, :] = 1.0  # own Q-side (Black)
        if cr & chess.BB_H1:
            planes[14, :, :] = 1.0  # opp K-side
        if cr & chess.BB_A1:
            planes[15, :, :] = 1.0  # opp Q-side

    # En-passant file (plane 16)
    if board.ep_square is not None:
        col = chess.square_file(board.ep_square)
        planes[16, :, col] = 1.0

    # Halfmove clock (plane 17)
    planes[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    # Repetition (plane 18)
    if board.is_repetition(2):
        planes[18, :, :] = 1.0

    return torch.from_numpy(planes)
