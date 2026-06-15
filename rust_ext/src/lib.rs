//! Native acceleration for OUROBOROS board encoding.
//!
//! These functions mirror `ouroboros/engine/encoding.py` exactly so that the
//! network sees byte-identical input whether or not this module is present.
//! All chess-rule logic (move generation, repetition) stays in python-chess;
//! this crate only does the index math and array filling that dominate the
//! per-node cost of MCTS.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

// Same orderings as encoding.py.
const KNIGHT_DELTAS: [(i32, i32); 8] = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
];
const QUEEN_DIRS: [(i32, i32); 8] = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
];

// python-chess piece type ids.
const KNIGHT: u8 = 2;
const BISHOP: u8 = 3;
const ROOK: u8 = 4;
const QUEEN: u8 = 5;

/// Map a move (raw squares + promotion) to a policy index, or -1 if unencodable.
/// `flip` mirrors squares vertically (sq ^ 56), matching the side-to-move frame.
#[inline]
fn move_index(from_sq: u8, to_sq: u8, promotion: u8, flip: bool) -> i64 {
    let mut f = from_sq as i32;
    let mut t = to_sq as i32;
    if flip {
        f ^= 56;
        t ^= 56;
    }
    let fr = f / 8;
    let fc = f % 8;
    let tr = t / 8;
    let tc = t % 8;
    let dr = tr - fr;
    let dc = tc - fc;

    // Underpromotion (anything that is not a queen promotion).
    if promotion != 0 && promotion != QUEEN {
        let piece_idx = match promotion {
            KNIGHT => 0,
            BISHOP => 1,
            ROOK => 2,
            _ => return -1,
        };
        let direction = if dc == 0 {
            0
        } else if dc == -1 {
            1
        } else {
            2
        };
        let plane = 64 + piece_idx * 3 + direction;
        return (f * 73 + plane) as i64;
    }

    // Knight move.
    for (i, (kr, kc)) in KNIGHT_DELTAS.iter().enumerate() {
        if *kr == dr && *kc == dc {
            return (f * 73 + 56 + i as i32) as i64;
        }
    }

    // Queen-style move (includes queen promotion).
    let dist = dr.abs().max(dc.abs());
    if dist == 0 {
        return -1;
    }
    let dvr = if dr != 0 { dr / dist } else { 0 };
    let dvc = if dc != 0 { dc / dist } else { 0 };
    for (i, (qr, qc)) in QUEEN_DIRS.iter().enumerate() {
        if *qr == dvr && *qc == dvc {
            let plane = i as i32 * 7 + (dist - 1);
            return (f * 73 + plane) as i64;
        }
    }

    -1
}

#[pyfunction]
fn move_to_index(from_sq: u8, to_sq: u8, promotion: u8, flip: bool) -> i64 {
    move_index(from_sq, to_sq, promotion, flip)
}

/// Batch version: one index per (from, to, promotion) triple.
#[pyfunction]
fn legal_indices(froms: Vec<u8>, tos: Vec<u8>, promos: Vec<u8>, flip: bool) -> Vec<i64> {
    let n = froms.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(move_index(froms[i], tos[i], promos[i], flip));
    }
    out
}

/// Fill the (19, 8, 8) input planes and return them as little-endian f32 bytes.
///
/// `own`/`opp` are the six piece bitboards (PAWN..KING order) for the side to
/// move and its opponent. `castle` is [own_K, own_Q, opp_K, opp_Q]. `ep_file`
/// is the en-passant file or -1. `halfmove` is already normalised to [0, 1].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn board_to_planes<'py>(
    py: Python<'py>,
    own: Vec<u64>,
    opp: Vec<u64>,
    castle: Vec<bool>,
    ep_file: i32,
    halfmove: f32,
    repetition: bool,
    flip: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut planes = vec![0f32; 19 * 64];

    for i in 0..6 {
        let mut bb = own[i];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let mut r = sq / 8;
            let c = sq % 8;
            if flip {
                r = 7 - r;
            }
            planes[i * 64 + r * 8 + c] = 1.0;
        }
        let mut bb = opp[i];
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let mut r = sq / 8;
            let c = sq % 8;
            if flip {
                r = 7 - r;
            }
            planes[(6 + i) * 64 + r * 8 + c] = 1.0;
        }
    }

    for k in 0..4 {
        if castle[k] {
            let base = (12 + k) * 64;
            for x in 0..64 {
                planes[base + x] = 1.0;
            }
        }
    }

    if ep_file >= 0 {
        let c = ep_file as usize;
        for r in 0..8 {
            planes[16 * 64 + r * 8 + c] = 1.0;
        }
    }

    let base17 = 17 * 64;
    for x in 0..64 {
        planes[base17 + x] = halfmove;
    }

    if repetition {
        let base18 = 18 * 64;
        for x in 0..64 {
            planes[base18 + x] = 1.0;
        }
    }

    let mut bytes = Vec::with_capacity(planes.len() * 4);
    for v in &planes {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Ok(PyBytes::new_bound(py, &bytes))
}

#[pymodule]
fn ouroboros_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(move_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(legal_indices, m)?)?;
    m.add_function(wrap_pyfunction!(board_to_planes, m)?)?;
    Ok(())
}
