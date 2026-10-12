"""
experiments.py — Grade de experimentos com o MarkovModel.

1. Para cada ordem (padrão: 1, 2, 3, 4) e cada seed (padrão: 0 a 49),
   gera uma melodia com a mesma temperatura.
2. Salva tudo em outputs/experiments.json, com os metadados de cada melodia.
3. Exporta para MIDI apenas as combinações pedidas em --export.

Sintaxe de --export: uma ou mais seleções ORDEM:SEEDS
    ORDEM  um número (2) ou * para todas as ordens
    SEEDS  lista separada por vírgulas, com intervalos opcionais: 1  |  2,3  |  0-4,10

Exemplos:
    # seed 1 de cada ordem + seeds 2 e 3 da ordem 2
    python scripts/experiments.py --export '*:1' 2:2,3

    # só reexporta a partir do JSON já gerado, sem treinar de novo
    python scripts/experiments.py --skip-generate --export 4:0-9

(No PowerShell e no cmd do Windows, use aspas duplas: "*:1".)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Raiz do projeto: este arquivo fica em scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Os módulos do modelo ficam em src/.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from midi_io import melody_to_midi, pad_last_measure  # noqa: E402
DEFAULT_JSON = PROJECT_ROOT / "outputs" / "experiments.json"
DEFAULT_MIDI_DIR = PROJECT_ROOT / "outputs" / "midi"


# ---------------------------------------------------------------------------
# Seleção do subconjunto a exportar
# ---------------------------------------------------------------------------

def parse_seeds(text: str) -> set[int]:
    """'2,3' -> {2, 3};  '0-4,10' -> {0, 1, 2, 3, 4, 10}."""
    seeds: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if lo > hi:
                raise ValueError(f"intervalo invertido: {part}")
            seeds.update(range(lo, hi + 1))
        else:
            seeds.add(int(part))
    return seeds


def parse_selection(text: str) -> tuple[int | None, set[int]]:
    """
    Tipo para argparse: 'ORDEM:SEEDS' -> (ordem ou None para '*', seeds).
    """
    try:
        order_txt, seeds_txt = text.split(":", 1)
        order = None if order_txt.strip() == "*" else int(order_txt)
        return order, parse_seeds(seeds_txt)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"seleção inválida '{text}' (formato ORDEM:SEEDS, ex.: 2:1,3 ou '*:0-4'): {err}"
        )


def selected_keys(
    selections: list[tuple[int | None, set[int]]],
    orders: list[int],
) -> set[tuple[int, int]]:
    """Expande as seleções em pares (ordem, seed)."""
    keys: set[tuple[int, int]] = set()
    for order, seeds in selections:
        for o in (orders if order is None else [order]):
            keys.update((o, s) for s in seeds)
    return keys


# ---------------------------------------------------------------------------
# Geração e persistência
# ---------------------------------------------------------------------------

def run_grid(orders: list[int], seeds: list[int],
             temperature: float, max_len: int, min_len: int) -> list[dict]:
    """Treina um modelo por ordem e gera uma melodia por seed."""
    # Importados aqui para que --skip-generate funcione sem treinar nada.
    from corpus import load_bach_sopranos
    from markov import MarkovModel

    corpus = load_bach_sopranos()
    print(f"Corpus: {len(corpus)} melodias")

    records: list[dict] = []
    for order in orders:
        model = MarkovModel(order)
        model.fit(corpus)                      # um treino por ordem
        for seed in seeds:
            melody = model.generate(max_len, temperature, seed, min_len)
            records.append({
                "order": order,
                "temperature": temperature,
                "seed": seed,
                "max_len": max_len,
                "min_len": min_len,
                # JSON não tem tuplas: cada evento vira [altura, duração].
                "melody": [[int(p), float(d)] for p, d in melody],
            })
        print(f"  ordem {order}: {len(seeds)} melodias geradas")
    return records


def save_json(records: list[dict], path: Path, config: dict) -> None:
    """
    Grava o JSON com uma melodia por linha: o arquivo continua JSON válido,
    mas fica legível e gera diffs pequenos no git.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ",\n    ".join(json.dumps(r, ensure_ascii=False) for r in records)
    with path.open("w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "config": {json.dumps(config, ensure_ascii=False)},\n')
        f.write(f'  "melodies": [\n    {lines}\n  ]\n')
        f.write("}\n")


def load_json(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"ERRO: {path} não existe; rode sem --skip-generate primeiro.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)["melodies"]


# ---------------------------------------------------------------------------
# Exportação MIDI
# ---------------------------------------------------------------------------

def export_midis(records: list[dict], keys: set[tuple[int, int]],
                 midi_dir: Path) -> None:
    """Grava em MIDI só as melodias cujos (ordem, seed) estão em `keys`."""
    by_key = {(r["order"], r["seed"]): r for r in records}

    missing = sorted(keys - by_key.keys())
    if missing:
        print(f"AVISO: combinações pedidas que não existem no experimento: {missing}")

    for key in sorted(keys & by_key.keys()):
        r = by_key[key]
        if not r["melody"]:
            print(f"  ! ordem {r['order']}, seed {r['seed']}: melodia vazia")
            continue
        name = f"ord{r['order']}_temp{r['temperature']}_seed{r['seed']}.mid"
        melody = [(int(p), float(d)) for p, d in r["melody"]]   # volta a tuplas
        melody = pad_last_measure(melody)                       # fecha o compasso
        melody_to_midi(melody, str(midi_dir / name))
        print(f"  ✓ {name}")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grade ordem × seed com o MarkovModel; salva JSON e exporta MIDIs.",
        epilog="Exemplo: --export '*:1' 2:2,3  (seed 1 de cada ordem + seeds 2 e 3 da ordem 2)",
    )
    p.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="ordens a testar (padrão: 1 2 3 4)")
    p.add_argument("--n-seeds", type=int, default=50,
                   help="seeds de 0 a N-1 (padrão: 50)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="temperatura de amostragem (padrão: 1.0)")
    p.add_argument("--max-len", type=int, default=64,
                   help="máximo de eventos por melodia (padrão: 64)")
    p.add_argument("--min-len", type=int, default=24,
                   help="mínimo de eventos antes de permitir o fim (padrão: 24)")
    p.add_argument("--json", type=Path, default=DEFAULT_JSON,
                   help="arquivo JSON (padrão: outputs/experiments.json)")
    p.add_argument("--midi-dir", type=Path, default=DEFAULT_MIDI_DIR,
                   help="pasta dos MIDIs (padrão: outputs/midi)")
    p.add_argument("--export", type=parse_selection, nargs="*", default=[],
                   metavar="ORDEM:SEEDS",
                   help="combinações a exportar para MIDI, ex.: '*:1' 2:2,3")
    p.add_argument("--skip-generate", action="store_true",
                   help="não gera nada; lê o JSON existente e só exporta")
    args = p.parse_args()

    if args.n_seeds < 1 or args.max_len < 1 or args.temperature <= 0:
        p.error("--n-seeds e --max-len devem ser >= 1 e --temperature > 0")
    if any(o < 1 for o in args.orders):
        p.error("as ordens devem ser >= 1")
    return args


def main() -> None:
    args = parse_args()

    if args.skip_generate:
        records = load_json(args.json)
        print(f"{len(records)} melodias lidas de {args.json}")
    else:
        seeds = list(range(args.n_seeds))
        records = run_grid(args.orders, seeds, args.temperature,
                           args.max_len, args.min_len)
        config = {
            "orders": args.orders,
            "seeds": [seeds[0], seeds[-1]],
            "temperature": args.temperature,
            "max_len": args.max_len,
            "min_len": args.min_len,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_json(records, args.json, config)
        print(f"{len(records)} melodias salvas em {args.json}")

    if args.export:
        # '*' se refere às ordens presentes nos dados (útil com --skip-generate).
        orders_in_data = sorted({r["order"] for r in records})
        keys = selected_keys(args.export, orders_in_data)
        available = {(r["order"], r["seed"]) for r in records}
        print(f"\nExportando {len(keys & available)} melodias para {args.midi_dir}:")
        export_midis(records, keys, args.midi_dir)


if __name__ == "__main__":
    main()
