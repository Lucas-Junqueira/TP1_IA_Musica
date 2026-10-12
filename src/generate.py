"""
generate.py — Treina o MarkovModel nos sopranos de Bach e grava melodias em MIDI.

Uso:
    python src/generate.py --order 2 --temperature 0.8 --seed 42 --n 5
    python src/generate.py --order 3 --max-len 48 --out-dir outputs/midi/teste

Com --n maior que 1, cada melodia usa uma semente diferente
(seed, seed+1, seed+2, ...), então os resultados são reprodutíveis e cada
arquivo é nomeado pela semente que o gerou:
    ord{order}_temp{temperature}_seed{seed}.mid
"""

from __future__ import annotations

import argparse
from pathlib import Path

from corpus import load_bach_sopranos
from markov import MarkovModel
from midi_io import melody_to_midi, pad_last_measure

# Raiz do projeto: este arquivo fica em src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "midi"


def positive_int(value: str) -> int:
    """Tipo para argparse: inteiro >= 1."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"deve ser >= 1, recebi {n}")
    return n


def positive_float(value: str) -> float:
    """Tipo para argparse: número real > 0."""
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"deve ser > 0, recebi {x}")
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera melodias com o MarkovModel treinado nos corais de Bach.",
    )
    parser.add_argument("--order", type=positive_int, default=2,
                        help="ordem da cadeia de Markov (padrão: 2)")
    parser.add_argument("--temperature", type=positive_float, default=1.0,
                        help="temperatura da amostragem (padrão: 1.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="semente da primeira melodia (padrão: 0)")
    parser.add_argument("--max-len", type=positive_int, default=64,
                        help="número máximo de eventos por melodia (padrão: 64)")
    parser.add_argument("--min-len", type=int, default=24,
                        help="número mínimo de eventos antes de permitir o fim (padrão: 24)")
    parser.add_argument("--n", type=positive_int, default=1,
                        help="quantas melodias gerar (padrão: 1)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="pasta de saída (padrão: outputs/midi)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Corpus de treino (vem do cache depois da primeira execução).
    melodies = load_bach_sopranos()
    print(f"Corpus: {len(melodies)} melodias")

    # 2. Treina o modelo uma única vez para todas as gerações.
    model = MarkovModel(args.order)
    model.fit(melodies)
    print(f"Modelo de ordem {args.order} treinado")

    # 3. Gera e grava cada melodia com sua própria semente.
    for i in range(args.n):
        seed = args.seed + i
        melody = model.generate(args.max_len, args.temperature, seed, args.min_len)

        if not melody:
            print(f"  ! seed {seed}: melodia vazia, nada gravado")
            continue

        # Fecha o último compasso 4/4 estendendo a última nota.
        melody = pad_last_measure(melody)

        name = f"ord{args.order}_temp{args.temperature}_seed{seed}.mid"
        path = args.out_dir / name
        melody_to_midi(melody, str(path))   # cria a pasta se necessário

        total = sum(dur for _, dur in melody)
        print(f"  ✓ {path.name}  ({len(melody)} eventos, {total:g} semínimas)")


if __name__ == "__main__":
    main()
