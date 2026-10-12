"""
run_analysis.py — Mede o trade-off de ordem nas melodias de outputs/experiments.json.

Para cada ordem do experimento calcula, com as funções de src/analysis.py:
  - fração de n-gramas copiados do corpus (padrão n = 6);
  - maior trecho copiado (em eventos);
  - diversidade (fração de melodias distintas).
e, com o próprio MarkovModel, estatísticas do modelo treinado:
  - número de contextos, ramificação média e entropia condicional.

Também faz uma varredura curta de temperatura na ordem padrão (2), para
mostrar o efeito do segundo hiperparâmetro.

Saídas:
    paper/figures/metrics_by_order.pdf   gráfico para o resumo
    paper/tables/results.tex             tabela LaTeX (booktabs)
    paper/tables/temperature.tex         tabela da varredura de temperatura
    outputs/analysis.json                todos os números, para consulta

Uso:
    python scripts/experiments.py         # gera outputs/experiments.json
    python scripts/run_analysis.py        # mede e grava as saídas acima
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Raiz do projeto: este arquivo fica em scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis import (  # noqa: E402
    CorpusIndex,
    copied_ngram_ratio,
    evaluate_by_order,
    longest_copied_run,
    plot_metrics_by_order,
    results_to_latex,
    diversity,
)
from corpus import load_bach_sopranos  # noqa: E402
from markov import MarkovModel  # noqa: E402

DEFAULT_JSON = PROJECT_ROOT / "outputs" / "experiments.json"
PAPER_DIR = PROJECT_ROOT / "paper"


def load_by_order(path: Path) -> tuple[dict, dict[int, list]]:
    """Lê o JSON do experimento e agrupa as melodias por ordem."""
    if not path.is_file():
        raise SystemExit(f"ERRO: {path} não existe; rode scripts/experiments.py antes.")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    by_order: dict[int, list] = {}
    for r in data["melodies"]:
        melody = [(int(p), float(d)) for p, d in r["melody"]]
        by_order.setdefault(r["order"], []).append(melody)
    return data["config"], by_order


def model_stats(corpus: list, orders: list[int]) -> dict[int, dict[str, float]]:
    """Esparsidade e previsibilidade do modelo treinado em cada ordem."""
    stats = {}
    for order in orders:
        model = MarkovModel(order).fit(corpus)
        stats[order] = {
            "contexts": model.n_contexts(),
            "branching": model.mean_branching(),
            "entropy_bits": model.conditional_entropy(),
        }
    return stats


def temperature_sweep(corpus: list, index: CorpusIndex, order: int,
                      temperatures: list[float], n_seeds: int, max_len: int,
                      min_len: int, n: int) -> dict[float, dict[str, float]]:
    """Mesmas métricas, variando só a temperatura numa ordem fixa."""
    model = MarkovModel(order).fit(corpus)
    out = {}
    for t in temperatures:
        mels = [model.generate(max_len, t, s, min_len) for s in range(n_seeds)]
        out[t] = {
            "copied_ratio": copied_ngram_ratio(mels, index, n),
            "longest_run": statistics.fmean(longest_copied_run(m, index) for m in mels),
            "diversity": diversity(mels),
            "mean_len": statistics.fmean(len(m) for m in mels),
        }
    return out


def temperature_to_latex(sweep: dict[float, dict[str, float]], order: int, n: int) -> str:
    rows = [
        f"    {t:g} & ${v['copied_ratio']:.2f}$ & ${v['longest_run']:.1f}$ & "
        f"${v['diversity']:.2f}$ & ${v['mean_len']:.1f}$ \\\\"
        for t, v in sweep.items()
    ]
    return "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{ccccc}",
        r"    \toprule",
        f"    $T$ & Copied {n}-grams & Longest run & Diversity & Length \\\\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{Effect of temperature $T$ at order {order} "
        f"(copied {n}-grams pooled over all melodies; means for the other columns).}}",
        r"  \label{tab:temperature}",
        r"\end{table}",
    ]) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Métricas de memorização por ordem.")
    p.add_argument("--json", type=Path, default=DEFAULT_JSON,
                   help="saída de scripts/experiments.py (padrão: outputs/experiments.json)")
    # n = 6: um modelo de ordem k só emite (k+1)-gramas vistos no corpus,
    # então com n <= 5 a métrica fica trivialmente em 1 para as ordens 3 e 4.
    p.add_argument("--n", type=int, default=6,
                   help="tamanho dos n-gramas comparados com o corpus (padrão: 6)")
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0],
                   help="temperaturas da varredura (padrão: 0.5 1.0 1.5 2.0)")
    p.add_argument("--temperature-order", type=int, default=2,
                   help="ordem usada na varredura de temperatura (padrão: 2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config, by_order = load_by_order(args.json)
    orders = sorted(by_order)
    n_seeds = len(by_order[orders[0]])
    print(f"{sum(len(v) for v in by_order.values())} melodias, ordens {orders}")

    corpus = load_bach_sopranos()
    index = CorpusIndex(corpus)   # construído uma vez para todas as métricas

    # 1. Métricas de memorização por ordem.
    results = evaluate_by_order(by_order, index, n=args.n)
    # No gráfico só entram as métricas que variam com a ordem (a diversidade
    # é 1,00 em todas e ocuparia um painel inteiro).
    plot_path = PAPER_DIR / "figures" / "metrics_by_order.pdf"
    plot_metrics_by_order(
        {o: {k: r[k] for k in ("copied_ratio", "longest_run")} for o, r in results.items()},
        plot_path,
        labels={"copied_ratio": f"Copied\n{args.n}-grams"},
    )

    # 2. Estatísticas dos modelos.
    stats = model_stats(corpus, orders)

    # Tabela do resumo: esparsidade do modelo + memorização. A diversidade
    # fica de fora porque é 1,00 em todas as ordens (citada no texto).
    table_results = {
        o: {
            "contexts": [float(stats[o]["contexts"])],
            "entropy": [stats[o]["entropy_bits"]],
            "copied_ratio": results[o]["copied_ratio"],
            "longest_run": results[o]["longest_run"],
        }
        for o in orders
    }
    table = results_to_latex(
        table_results,
        labels={"contexts": "Contexts", "entropy": "$H$ (bits)",
                "copied_ratio": f"Copied {args.n}-grams"},
        decimals={"contexts": 0, "entropy": 2, "copied_ratio": 2, "longest_run": 1},
        caption=(f"Model sparsity and memorization by Markov order. Contexts: distinct "
                 f"contexts seen in training; $H$: mean conditional entropy of the next "
                 f"event. The last two columns are mean $\\pm$ s.d.\\ over {n_seeds} "
                 f"melodies ($T = {config['temperature']:g}$); longest run is the longest "
                 "span, in events, copied verbatim from one chorale."),
    )
    table_dir = PAPER_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "results.tex").write_text(table, encoding="utf-8")

    # 3. Varredura de temperatura.
    sweep = temperature_sweep(corpus, index, args.temperature_order, args.temperatures,
                              n_seeds, config["max_len"], config.get("min_len", 0), args.n)
    (table_dir / "temperature.tex").write_text(
        temperature_to_latex(sweep, args.temperature_order, args.n), encoding="utf-8")

    # 4. Resumo no terminal e em JSON.
    print(f"\n{'ordem':>5} {'copiados':>9} {'maior':>7} {'divers.':>8} "
          f"{'contextos':>10} {'ramif.':>7} {'H (bits)':>9} {'tam.':>6}")
    summary = {"config": config, "n": args.n, "by_order": {}, "temperature": {}}
    for o in orders:
        r = results[o]
        row = {
            "copied_ratio_mean": statistics.fmean(r["copied_ratio"]),
            "longest_run_mean": statistics.fmean(r["longest_run"]),
            "longest_run_max": max(r["longest_run"]),
            "diversity": r["diversity"][0],
            "mean_len": statistics.fmean(len(m) for m in by_order[o]),
            **stats[o],
        }
        summary["by_order"][o] = row
        print(f"{o:>5} {row['copied_ratio_mean']:>9.2f} {row['longest_run_mean']:>7.1f} "
              f"{row['diversity']:>8.2f} {row['contexts']:>10} {row['branching']:>7.2f} "
              f"{row['entropy_bits']:>9.2f} {row['mean_len']:>6.1f}")

    print(f"\nTemperatura (ordem {args.temperature_order}):")
    for t, v in sweep.items():
        summary["temperature"][t] = v
        print(f"  T={t:<4g} copiados={v['copied_ratio']:.2f}  maior={v['longest_run']:.1f}  "
              f"diversidade={v['diversity']:.2f}  tamanho={v['mean_len']:.1f}")

    out_json = PROJECT_ROOT / "outputs" / "analysis.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGráfico: {plot_path}\nTabelas: {table_dir}\nNúmeros: {out_json}")


if __name__ == "__main__":
    main()
