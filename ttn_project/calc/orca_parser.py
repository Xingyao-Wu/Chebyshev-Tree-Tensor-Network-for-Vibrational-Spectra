#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Dict, Tuple, List, Optional
import pathlib

Float = float
Idx3 = Tuple[int, int, int]
Idx4 = Tuple[int, int, int, int]

FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

def parse_block(lines: List[str], header: str) -> Tuple[int, int, int, List[Tuple[int,int,int,Float]]]:
    """Parse a 3-index numeric block (i j k value) that starts with `header` and a dims line."""
    out = []
    it = iter(range(len(lines)))
    for p in it:
        if header in lines[p]:
            ni, nj, nk = map(int, lines[p+1].split())
            row_pat = re.compile(rf"^\s*(\d+)\s+(\d+)\s+(\d+)\s+({FLOAT_RE})\s*$")
            q = p + 2
            while q < len(lines):
                m = row_pat.match(lines[q])
                if not m: break
                i_, j_, k_, v = m.groups()
                out.append((int(i_), int(j_), int(k_), float(v)))
                q += 1
            return ni, nj, nk, out
    raise ValueError(f"Block not found: {header}")

def read_orca_vpt2(path: pathlib.Path):
    """Return n_modes, Phi3 dict[(i,j,k)], Phi4semi dict[(i,j,k)] where Phi4semi = Φ_{ij,kk}."""
    with open(str(path)+".vpt2", "r", encoding="utf-8") as f:
        lines = f.readlines()

    ni, nj, nk, cub_rows = parse_block(lines, "# Cubic[i][j][k] force field in 1/cm")
    if not (ni == nj == nk): raise ValueError("Cubic dims not cubic")
    n_modes = ni

    si, sj, sk, q_rows = parse_block(lines, "# Semi-quartic[i][j][k][k] force field in 1/cm")
    if not (si == sj == sk == n_modes): raise ValueError("Semi-quartic dims mismatch")

    Phi3: Dict[Idx3, Float] = {(i,j,k): v for i,j,k,v in cub_rows}
    Phi4semi: Dict[Idx3, Float] = {(i,j,k): v for i,j,k,v in q_rows}  # represents Φ_{ij,kk}

    return n_modes, Phi3, Phi4semi

def read_orca_vib_frequencies(path: pathlib.Path) -> List[Float]:
    """
    Parse an ORCA VPT2 `vib.out` file to extract normal mode frequencies.
    The `path` may be:
      - a full path to the `vib.out` file
      - a basename without extension (e.g. '/path/to/vib') -> '/path/to/vib.out'
      - a directory that contains 'vib.out'
    Returns a list of frequencies in cm^-1 ordered by mode index.
    """
    p = pathlib.Path(path)
    candidate_paths: List[pathlib.Path] = []

    if p.is_dir():
        candidate_paths.append(p / "vib.out")
    else:
        candidate_paths.append(p)
        candidate_paths.append(p.with_suffix(".out"))
        candidate_paths.append(p.parent / (p.name + ".out"))

    file_path = next((c for c in candidate_paths if c.exists() and c.is_file()), None)
    if file_path is None:
        raise FileNotFoundError(f"Could not locate vib.out. Tried: {', '.join(str(c) for c in candidate_paths)}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the "mode frequency" table
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"^\s*mode\s+frequency\b", line, flags=re.IGNORECASE):
            header_idx = i
            break
    if header_idx is None:
        # Prefer the block after this marker if present
        marker_idx = None
        for i, line in enumerate(lines):
            if "Computing mean and mean square displacements of normal modes" in line:
                marker_idx = i
                break
        if marker_idx is not None:
            for j in range(marker_idx, min(marker_idx + 500, len(lines))):
                if re.search(r"^\s*mode\s+freq(?:uency)?\b", lines[j], flags=re.IGNORECASE):
                    header_idx = j
                    break
    if header_idx is None:
        raise ValueError("Could not find 'mode frequency' table in vib.out")

    # Data typically starts two lines after header (after a dashed line)
    start = header_idx + 2
    freqs: List[Float] = []
    row_re = re.compile(rf"^\s*(\d+)\s+({FLOAT_RE})\b")
    for k in range(start, len(lines)):
        s = lines[k].strip()
        if not s or s.startswith("---"):
            # end of table or separator
            if freqs:  # stop after we parsed at least one block
                break
            continue
        m = row_re.match(lines[k])
        if not m:
            # stop at first non-matching line after data started
            if freqs:
                break
            continue
        mode_idx, freq = m.groups()
        # mode index is available but we rely on order
        try:
            freqs.append(float(freq))
        except ValueError:
            # skip malformed numbers
            continue

    if not freqs:
        raise ValueError("No frequencies parsed from 'mode frequency' table in vib.out")

    return freqs

def get_dipole_moment_1(path: pathlib.Path) -> List[Tuple[Float, Float, Float]]:
    """
    Parse ORCA VPT2 `vib.out` IR intensities block and return (Tx, Ty, Tz) in atomic units
    for the three mode rows immediately above the line that starts
    'Analysis of possible Fermi resonances'.

    The `path` resolution matches `read_orca_vib_frequencies` (directory, `vib.out`, or basename).
    Returns three (Tx, Ty, Tz) tuples in ascending mode index order.
    """
    p = pathlib.Path(path)
    candidate_paths: List[pathlib.Path] = []
    if p.is_dir():
        candidate_paths.append(p / "vib.out")
    else:
        candidate_paths.append(p)
        candidate_paths.append(p.with_suffix(".out"))
        candidate_paths.append(p.parent / (p.name + ".out"))

    file_path = next((c for c in candidate_paths if c.exists() and c.is_file()), None)
    if file_path is None:
        raise FileNotFoundError(f"Could not locate vib.out. Tried: {', '.join(str(c) for c in candidate_paths)}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fermi_idx = None
    for i, line in enumerate(lines):
        if "Analysis of possible Fermi resonances" in line:
            fermi_idx = i
            break
    if fermi_idx is None:
        raise ValueError("Could not find 'Analysis of possible Fermi resonances' in vib.out")

    # Mode freq Int T2 ( Tx Ty Tz ) — skip rows where floats are -nan (not matched by FLOAT_RE)
    row_re = re.compile(
        rf"^\s*(\d+)\s+({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})\s+\(\s*({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})\s*\)\s*$"
    )

    collected: List[Tuple[Float, Float, Float]] = []
    for j in range(fermi_idx - 1, -1, -1):
        raw = lines[j]
        if not raw.strip():
            continue
        m = row_re.match(raw)
        if not m:
            if collected:
                break
            continue
        tx, ty, tz = map(float, m.groups()[4:7])
        collected.append((tx, ty, tz))
        if len(collected) == 3:
            break

    if len(collected) != 3:
        raise ValueError(
            "Expected 3 IR intensity rows with (Tx, Ty, Tz) immediately before Fermi resonance section; "
            f"found {len(collected)} matching row(s)."
        )

    collected.reverse()
    return collected

def canonical3(i,j,k) -> Tuple[Idx3, str]:
    """Return nondecreasing triple and class: 'iii', 'iij', 'ijk'."""
    s = tuple(sorted((i,j,k)))
    if s[0]==s[2]: return s, "iii"
    if s[0]==s[1] or s[1]==s[2]: return s, "iij"
    return s, "ijk"

def sign_factor(indices: Tuple[int, ...], signs_1based: List[int]) -> int:
    s = 1
    for idx in indices:
        s *= signs_1based[idx]
    return s

def map_paper_to_orca(idx_list_1based: Tuple[int, ...], paper_to_orca_0based: List[int]) -> Tuple[int, ...]:
    """Map paper 1..M to ORCA 0..M-1."""
    return tuple(paper_to_orca_0based[i-1] for i in idx_list_1based)

def build_k_cubic(n_modes: int,
                  Phi3: Dict[Idx3, Float],
                  paper_to_orca: List[int],
                  paper_signs: List[int]) -> Dict[Idx3, Float]:
    """
    Build unique k_ijk (paper 1-based, i<=j<=k). Conversion (no factorials in k):
      k_iii  = Φ_iii / 6
      k_iij  = Φ_iij / 2
      k_ijk  = Φ_ijk
    """
    k: Dict[Idx3, Float] = {}
    for ip in range(1, n_modes+1):
        for jp in range(ip, n_modes+1):
            for kp in range(jp, n_modes+1):
                io, jo, ko = map_paper_to_orca((ip,jp,kp), paper_to_orca)
                # ORCA prints many permutations; try canonical first, then permutations.
                Phi = None
                for a,b,c in {(io,jo,ko),(io,ko,jo),(jo,io,ko),(jo,ko,io),(ko,io,jo),(ko,jo,io)}:
                    if (a,b,c) in Phi3:
                        Phi = Phi3[(a,b,c)]
                        break
                if Phi is None:  # shouldn’t happen, but be robust
                    continue

                cls = canonical3(ip,jp,kp)[1]
                if cls == "iii": val = Phi
                elif cls == "iij": val = Phi
                else: val = Phi

                sf = sign_factor((ip,jp,kp), [None]+paper_signs)
                k[(ip,jp,kp)] = val*sf
    return k

def build_k_quartic(n_modes: int,
                    Phi4semi: Dict[Idx3, Float],
                    paper_to_orca: List[int],
                    paper_signs: List[int]) -> Dict[Idx4, Float]:
    """
    Build unique quartic k_ijkl (paper 1-based, i<=j<=k<=l) from *semi-quartic* Φ_{ij,kk}.
    Available patterns and divisors:
      iiii: k = Φ_{ii,ii} / 24
      iijj: k = Φ_{ii,jj} / 4
      ijkk: k = Φ_{ij,kk} / 2
    We ensure ijkk does NOT overwrite iiii/iijj entries.
    """
    kq: Dict[Idx4, Float] = {}

    signs1 = [None] + paper_signs  # 1-based

    # iiii: use entries where (i=j=k) in semi-quartic key (io,io,io) -> Φ_{ii,ii}
    for ip in range(1, n_modes+1):
        io = paper_to_orca[ip-1]
        Phi = Phi4semi.get((io, io, io))
        if Phi is None: continue
        key = (ip, ip, ip, ip)
        sf = sign_factor(key, signs1)
        kq[key] = (Phi) * sf

    # iijj: use (io,io,jo) -> Φ_{ii,jj}, with ip<jp to keep uniqueness
    for ip in range(1, n_modes+1):
        for jp in range(ip+1, n_modes+1):
            io, jo = paper_to_orca[ip-1], paper_to_orca[jp-1]
            # ORCA semi-quartic is symmetric in (i,j) and typically stored with i<=j
            a, b = (io, jo) if io <= jo else (jo, io)
            Phi = Phi4semi.get((a, a, b))
            if Phi is None: continue
            key = (ip, ip, jp, jp)
            sf = sign_factor(key, signs1)
            # only set if not already present (it shouldn’t be)
            if key not in kq:
                kq[key] = (Phi) * sf

    # ijkk: general case from Φ_{ij,kk} with any i<=j, k arbitrary,
    # but skip combinations that collapse to iiii or iijj to avoid overwrites.
    for ip in range(1, n_modes+1):
        for jp in range(ip, n_modes+1):
            for kp in range(1, n_modes+1):
                # Skip if this is iiii: ip==jp==kp
                if ip == jp == kp: 
                    continue
                # Skip if this is iijj type: (two of one index, two of another)
                if (ip == jp and kp != ip):
                    # This ijkk would sort to (ip, ip, kp, kp) which we already set as iijj
                    continue

                io, jo, ko = paper_to_orca[ip-1], paper_to_orca[jp-1], paper_to_orca[kp-1]
                # Canonicalize first two indices (i,j) as ORCA stores with i<=j
                a, b = (io, jo) if io <= jo else (jo, io)
                Phi = Phi4semi.get((a, b, ko))
                if Phi is None: continue

                # Canonical 4-tuple (sorted) with two k's
                key = tuple(sorted((ip, jp, kp, kp)))
                sf = sign_factor((ip, jp, kp, kp), signs1)
                
                if key not in kq:
                    kq[key] = (Phi) * sf

    return kq

def get_anharmonic_constants(path: pathlib.Path, map_modes: Optional[List[int]] = None, signs: Optional[List[int]] = None):
    n_modes, Phi3, Phi4semi = read_orca_vpt2(path)

    paper_to_orca = map_modes if map_modes else list(range(n_modes))
    paper_signs = signs if signs else [1]*n_modes
    if len(paper_to_orca) != n_modes or len(paper_signs) != n_modes:
        raise ValueError("Lengths of map_modes and signs must equal number of modes in the file.")

    kc = build_k_cubic(n_modes, Phi3, paper_to_orca, paper_signs)
    kq = build_k_quartic(n_modes, Phi4semi, paper_to_orca, paper_signs)

    cubic = " ".join([f"{i} {j} {k} {kc[(i,j,k)]: .8f}\n" for (i,j,k) in sorted(kc.keys()) if abs(kc[(i,j,k)]) > 1.0])

    quartic = " ".join([f"{i} {j} {k} {l} {kq[(i,j,k,l)]: .8f}\n" for (i,j,k,l) in sorted(kq.keys()) if abs(kq[(i,j,k,l)]) > 1.0])
    
    freq = read_orca_vib_frequencies(path)
    return freq, cubic, quartic



if __name__ == "__main__":
    freq,cubic, quartic = get_anharmonic_constants(path=pathlib.Path(__file__).parent / "../orca" / "h2o"/"vib", map_modes=[0,1,2], signs=[1,1,1])
    print(freq)
    # print(cubic)
    # print(quartic)
    dipole_moment = get_dipole_moment_1(path=pathlib.Path(__file__).parent / "../orca" / "h2o"/"vib")
    print(dipole_moment)