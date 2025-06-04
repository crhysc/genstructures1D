# preprocess.py
"""Unified dataset preprocessing CLI (AtomGPT, CDVAE, FlowMM)
----------------------------------------------------------------
Reads a *single* JSON file that contains a top‑level key ``"entries"`` with
COMPUTED‑STRUCTURE style records (as shown in *pretty.json*).  For each
sub‑command the script materialises the *exact* output formats produced by the
original stand‑alone scripts you provided, but **deduplicated so that only the
*last* occurrence of each material is kept** (the final relaxation / lowest
energy entry).

Sub‑commands
~~~~~~~~~~~
- **atomgpt** →  ``<OUTPUT>/id_prop.csv``  + one `<mat_id>.vasp` POSCAR per
  entry.  CSV has *no header* and two columns: ``structure_path`` (relative
  file‑name) and the target property (default ``Tc_supercon``).

- **cdvae**   →  ``train.csv``, ``val.csv``, ``test.csv`` in *OUTPUT*.
  Each CSV contains: ``material_id``, ``cif`` (the raw CIF string), and
  ``prop`` (target value).  Splits follow an 80/10/10 ratio with a
  reproducible shuffle (``--seed``).

- **flowmm**  →  the same three CSVs but with the richer schema described in
  the reference script: ``material_id``, ``pretty_formula``, ``elements``
  (JSON list of *all* site species), ``cif`` (raw), ``spacegroup.number``,
  ``spacegroup.number.conv``, ``cif.conv`` (canonical), and the target.

Example
-------
```bash
python preprocess.py atomgpt -i pretty.json -o ./atomgpt_out
python preprocess.py cdvae   -i pretty.json -o ./cdvae_out --seed 42
python preprocess.py flowmm  -i pretty.json -o ./flowmm_out --seed 42
```
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

###############################################################################
# General helpers                                                             #
###############################################################################

def _load_entries(path: Path | str) -> List[Dict[str, Any]]:
    """Return the list found under the ``"entries"`` key of *path*."""
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"[ERROR] failed to read dataset '{path}': {exc}")

    entries: Any = payload.get("entries")
    if not isinstance(entries, list):
        sys.exit("[ERROR] dataset must be a JSON object with key 'entries' -> list")
    return entries


def _deduplicate_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep **only the last occurrence** for each ``mat_id``."""
    dedup: Dict[str, Dict[str, Any]] = {}
    for item in entries:  # later duplicates overwrite earlier ones
        mat_id = item.get("data", {}).get("mat_id")
        if mat_id is None:
            continue  # skip bad record
        dedup[mat_id] = item
    return list(dedup.values())


def _entries_to_dataframe(entries: List[Dict[str, Any]], target_key: str) -> pd.DataFrame:
    """Light frame holding only what we need downstream."""
    rows: List[Dict[str, Any]] = []
    for e in entries:
        data_blk = e.get("data", {})
        tgt_val = data_blk.get(target_key)
        if tgt_val in (None, "na"):
            continue
        rows.append(
            {
                "mat_id": data_blk.get("mat_id"),
                target_key: tgt_val,
                "structure_dict": e.get("structure"),  # to be re‑instantiated later
            }
        )
    return pd.DataFrame(rows)


def _split_indices(n: int, seed: int, train: float = 0.8, val: float = 0.1, test: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train + val + test - 1) < 1e-8
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    n_train = int(train * n)
    n_test = int(test * n)
    n_val = int(val * n)
    if n_train + n_val + n_test > n:
        raise ValueError("train/val/test sizes exceed dataset length")
    id_train = idx[:n_train]
    id_val = idx[-(n_val + n_test): -n_test]
    id_test = idx[-n_test:]
    return id_train, id_val, id_test


def _sha10(strings: List[str]) -> str:
    h = sha256()
    for s in strings:
        h.update(s.encode())
        h.update(b",")
    return h.hexdigest()[:10]

###############################################################################
# Abstract + concrete preprocessors                                           #
###############################################################################

class Preprocessor(ABC):
    def __init__(self, df: pd.DataFrame, target_key: str, seed: int):
        self.df = df.reset_index(drop=True)
        self.target_key = target_key
        self.seed = seed

    @abstractmethod
    def preprocess(self, output_dir: Path) -> None:  # pragma: no cover
        ...


# --------------------------------------------------------------------------- #
# AtomGPT                                                                     #
# --------------------------------------------------------------------------- #
class AtomGPTPreprocessor(Preprocessor):
    """Writes one <mat_id>.vasp POSCAR per entry + id_prop.csv (no header)."""

    def preprocess(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_paths: List[str] = []
        targets: List[float] = []

        for _, row in self.df.iterrows():
            mat_id = row["mat_id"]
            struct_dict = row["structure_dict"]
            target_val = row[self.target_key]
            struct = Structure.from_dict(struct_dict)

            fname = f"{mat_id}.vasp"
            fpath = output_dir / fname
            Poscar(struct).write_file(str(fpath))

            structure_paths.append(fname)
            targets.append(target_val)

        # Build id_prop.csv (no header)
        pd.DataFrame({"structure_path": structure_paths, self.target_key: targets}).to_csv(
            output_dir / "id_prop.csv", index=False, header=False
        )
        print(f"[AtomGPT] wrote {len(structure_paths)} POSCARs and id_prop.csv | sha10={_sha10(structure_paths)}")


# --------------------------------------------------------------------------- #
# CD‑VAE                                                                      #
# --------------------------------------------------------------------------- #
class CDVAEPreprocessor(Preprocessor):
    """Creates train/val/test CSVs with material_id, cif, prop."""

    def preprocess(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        records: List[Dict[str, Any]] = []
        for _, row in self.df.iterrows():
            struct = Structure.from_dict(row["structure_dict"])
            try:
                cif_raw = struct.to(fmt="cif")
            except Exception:
                continue  # skip uncif‑able structures
            records.append(
                {
                    "material_id": row["mat_id"],
                    "cif": cif_raw,
                    "prop": row[self.target_key],
                }
            )

        frame = pd.DataFrame(records)
        id_train, id_val, id_test = _split_indices(len(frame), self.seed)
        frame.iloc[id_train].to_csv(output_dir / "train.csv", index=False)
        frame.iloc[id_val].to_csv(output_dir / "val.csv", index=False)
        frame.iloc[id_test].to_csv(output_dir / "test.csv", index=False)

        print(
            "[CDVAE] wrote train/val/test CSVs | "
            f"train:{_sha10(frame.iloc[id_train]['material_id'].tolist())} "
            f"val:{_sha10(frame.iloc[id_val]['material_id'].tolist())} "
            f"test:{_sha10(frame.iloc[id_test]['material_id'].tolist())}"
        )


# --------------------------------------------------------------------------- #
# FlowMM                                                                      #
# --------------------------------------------------------------------------- #
class FlowMMPreprocessor(Preprocessor):
    """Richer CSVs with canonicalised CIFs and space‑group numbers."""

    @staticmethod
    def _canonicalise(struct: Structure, symprec: float = 0.1) -> Tuple[str, int, int]:
        try:
            sga = SpacegroupAnalyzer(struct, symprec=symprec)
            spg_num = sga.get_space_group_number()
            conv = sga.get_conventional_standard_structure()
            spg_conv = SpacegroupAnalyzer(conv, symprec=symprec).get_space_group_number()
            return conv.to(fmt="cif"), spg_num, spg_conv
        except Exception:
            return "", -1, -1

    def preprocess(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        records: List[Dict[str, Any]] = []
        for _, row in self.df.iterrows():
            struct = Structure.from_dict(row["structure_dict"])
            try:
                cif_raw = struct.to(fmt="cif")
            except Exception:
                continue
            cif_conv, spg, spg_conv = self._canonicalise(struct)
            records.append(
                {
                    "material_id": row["mat_id"],
                    "pretty_formula": struct.composition.reduced_formula,
                    "elements": json.dumps([el.symbol for el in struct.species]),
                    "cif": cif_raw,
                    "spacegroup.number": spg,
                    "spacegroup.number.conv": spg_conv,
                    "cif.conv": cif_conv,
                    self.target_key: row[self.target_key],
                }
            )

        frame = pd.DataFrame(records)
        id_train, id_val, id_test = _split_indices(len(frame), self.seed)
        frame.iloc[id_train].to_csv(output_dir / "train.csv", index=False)
        frame.iloc[id_val].to_csv(output_dir / "val.csv", index=False)
        frame.iloc[id_test].to_csv(output_dir / "test.csv", index=False)
        print(
            "[FlowMM] wrote train/val/test CSVs | "
            f"train:{_sha10(frame.iloc[id_train]['material_id'].tolist())} "
            f"val:{_sha10(frame.iloc[id_val]['material_id'].tolist())} "
            f"test:{_sha10(frame.iloc[id_test]['material_id'].tolist())}"
        )

###############################################################################
# Preprocessor factory                                                        #
###############################################################################

class PreprocessorFactory:
    _REGISTRY = {
        "atomgpt": AtomGPTPreprocessor,
        "cdvae": CDVAEPreprocessor,
        "flowmm": FlowMMPreprocessor,
    }

    @staticmethod
    def create(name: str, df: pd.DataFrame, target_key: str, seed: int) -> Preprocessor:
        try:
            return PreprocessorFactory._REGISTRY[name.lower()](df, target_key, seed)
        except KeyError:
            valid = ", ".join(sorted(PreprocessorFactory._REGISTRY))
            raise ValueError(f"unknown subcommand '{name}'. valid choices: {valid}")

###############################################################################
# CLI                                                                         #
###############################################################################

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess materials dataset")
    p.add_argument("subcommand", choices=["atomgpt", "cdvae", "flowmm"], help="output format")
    p.add_argument("-i", "--input", type=Path, required=True, help="Path to JSON dataset (pretty.json)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Directory for artefacts")
    p.add_argument("--target", default="Tc_supercon", help="Target property key in the 'data' blob")
    p.add_argument("--seed", type=int, default=123, help="Random seed for dataset splits")
    return p


def main(argv: List[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    raw_entries = _load_entries(args.input)
    raw_entries = _deduplicate_entries(raw_entries)
    df = _entries_to_dataframe(raw_entries, args.target)
    print(f"[INFO] loaded {len(df)} unique structures with target '{args.target}' from {args.input}")
    if df.empty:
        sys.exit("[ERROR] no valid entries found – check the target key or dataset contents.")

    preproc = PreprocessorFactory.create(args.subcommand, df, args.target, args.seed)
    preproc.preprocess(args.output)


if __name__ == "__main__":  # pragma: no cover
    main()

