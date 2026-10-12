"""
analysis.py — Métricas de originalidade e diversidade de melodias geradas.

Uma melodia é uma lista de eventos (altura_midi, duracao), como em corpus.py.
As comparações com o corpus são literais: dois eventos são iguais quando a
altura e a duração coincidem (a duração é arredondada a 6 casas decimais
para evitar diferenças de ponto flutuante).

Métricas
--------
copied_ngram_ratio(generated, corpus, n)
    Fração dos n-gramas gerados que aparecem tal e qual em alguma melodia
    do corpus. Perto de 1 = o modelo está copiando trechos.
longest_copied_run(melody, corpus)
    Comprimento (em eventos) do maior trecho contíguo da melodia que existe
    idêntico em alguma melodia do corpus.
diversity(generated)
    Fração de melodias distintas entre si.

Saídas para o artigo
--------------------
plot_metrics_by_order(results, path)   gráfico PDF, média ± desvio por ordem
results_to_latex(results)              tabela LaTeX com booktabs

Eficiência
----------
As duas métricas que consultam o corpus usam um índice (CorpusIndex) baseado
num autômato de sufixos. Ele é construído uma vez em O(N), com N = total de
eventos do corpus, e responde, para cada posição de uma melodia de tamanho m,
qual o maior trecho terminando ali que existe no corpus — tudo em O(m).
É o mesmo resultado de testar conjuntos de n-gramas para n = 1, 2, 3, …,
sem precisar guardar um conjunto para cada n.

Para avaliar muitas melodias, construa o índice uma vez e passe-o no lugar
da lista do corpus:

    index = CorpusIndex(corpus)
    runs = [longest_copied_run(m, index) for m in generated]

(Passar a lista diretamente também funciona; o índice da última lista usada
fica em cache. Não modifique essa lista depois de usá-la.)
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

Event = tuple[int, float]
Melody = Sequence[Event]

_SEP = None  # separador entre melodias no índice; nunca aparece numa consulta


def _key(event: Event) -> tuple[int, float]:
    """Forma canônica de um evento (duração arredondada) para comparação."""
    pitch, dur = event
    return (int(pitch), round(float(dur), 6))


# ---------------------------------------------------------------------------
# Índice do corpus: autômato de sufixos
# ---------------------------------------------------------------------------

class CorpusIndex:
    """
    Índice de todas as subsequências contíguas das melodias do corpus.

    As melodias são concatenadas com um separador entre elas, para que
    nenhum trecho encontrado atravesse a fronteira entre duas melodias.
    """

    def __init__(self, corpus: Sequence[Melody]):
        # Estado 0 é a raiz. Cada estado tem: transições, link de sufixo
        # e comprimento da maior cadeia que ele representa.
        self._next: list[dict] = [{}]
        self._link: list[int] = [-1]
        self._len: list[int] = [0]
        self._last = 0

        for melody in corpus:
            for event in melody:
                self._extend(_key(event))
            self._extend(_SEP)

    def _extend(self, c) -> None:
        """Acrescenta um símbolo ao autômato (algoritmo clássico, O(1) amortizado)."""
        nxt, link, length = self._next, self._link, self._len

        cur = len(length)
        nxt.append({})
        link.append(-1)
        length.append(length[self._last] + 1)

        p = self._last
        while p != -1 and c not in nxt[p]:
            nxt[p][c] = cur
            p = link[p]

        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][c]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                # Divide o estado q: cria um clone com o comprimento correto.
                clone = len(length)
                nxt.append(dict(nxt[q]))
                link.append(link[q])
                length.append(length[p] + 1)
                while p != -1 and nxt[p].get(c) == q:
                    nxt[p][c] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone

        self._last = cur

    def match_lengths(self, melody: Melody) -> list[int]:
        """
        Para cada posição i da melodia, o comprimento do maior trecho
        melody[i-L+1 : i+1] que existe idêntico em alguma melodia do corpus.
        """
        nxt, link, length = self._next, self._link, self._len
        state, run = 0, 0
        out: list[int] = []

        for event in melody:
            c = _key(event)
            # Encurta o trecho atual até que ele possa ser estendido com c.
            while state != 0 and c not in nxt[state]:
                state = link[state]
                run = length[state]
            if c in nxt[state]:
                state = nxt[state][c]
                run += 1
            else:
                state, run = 0, 0   # evento que nunca aparece no corpus
            out.append(run)
        return out


# Cache do índice da última lista de corpus recebida diretamente.
_cache: tuple[object, int, CorpusIndex] | None = None


def _as_index(corpus: Sequence[Melody] | CorpusIndex) -> CorpusIndex:
    """Aceita um CorpusIndex pronto ou uma lista de melodias (com cache)."""
    global _cache
    if isinstance(corpus, CorpusIndex):
        return corpus
    # Guardamos a própria lista no cache: enquanto ela estiver lá, seu id
    # não pode ser reutilizado por outro objeto.
    if _cache is not None and _cache[0] is corpus and _cache[1] == len(corpus):
        return _cache[2]
    index = CorpusIndex(corpus)
    _cache = (corpus, len(corpus), index)
    return index


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def copied_ngram_ratio(
    generated: Sequence[Melody],
    corpus: Sequence[Melody] | CorpusIndex,
    n: int,
) -> float:
    """
    Fração dos n-gramas das melodias geradas que aparecem literalmente em
    alguma melodia do corpus. Todos os n-gramas contam, inclusive repetidos.

    O n-grama que termina na posição i está no corpus se, e só se, o maior
    trecho copiado terminando em i tem comprimento >= n.

    Devolve NaN se nenhuma melodia tiver pelo menos n eventos.
    """
    if n < 1:
        raise ValueError("n deve ser >= 1")
    index = _as_index(corpus)

    total = copied = 0
    for melody in generated:
        lengths = index.match_lengths(melody)
        for i in range(n - 1, len(lengths)):   # posições onde termina um n-grama
            total += 1
            copied += lengths[i] >= n
    return copied / total if total else math.nan


def longest_copied_run(
    melody: Melody,
    corpus: Sequence[Melody] | CorpusIndex,
) -> int:
    """
    Comprimento, em eventos, do maior trecho contíguo da melodia que existe
    idêntico em alguma melodia do corpus. 0 se nenhum evento aparece no corpus.
    """
    lengths = _as_index(corpus).match_lengths(melody)
    return max(lengths, default=0)


def diversity(generated: Sequence[Melody]) -> float:
    """
    Fração de melodias distintas: número de melodias únicas / total.
    1.0 = todas diferentes; 1/N = todas iguais. NaN para lista vazia.
    """
    if not generated:
        return math.nan
    unique = {tuple(_key(e) for e in melody) for melody in generated}
    return len(unique) / len(generated)


# ---------------------------------------------------------------------------
# Avaliação por ordem
# ---------------------------------------------------------------------------

# results: ordem -> nome da métrica -> lista de valores.
# Métricas por melodia têm um valor por melodia (média ± desvio);
# métricas do conjunto, como a diversidade, têm um único valor (desvio 0).
Results = Mapping[int, Mapping[str, Sequence[float]]]

# Rótulos curtos: precisam caber numa coluna de artigo em duas colunas.
# No gráfico, \n quebra o rótulo do eixo y em duas linhas.
PLOT_LABELS = {
    "copied_ratio": "Copied\n$n$-grams",
    "longest_run": "Longest run\n(events)",
    "diversity": "Diversity",
}
TABLE_LABELS = {
    "copied_ratio": "Copied $n$-grams",
    "longest_run": "Longest run",
    "diversity": "Diversity",
}


def evaluate_by_order(
    generated_by_order: Mapping[int, Sequence[Melody]],
    corpus: Sequence[Melody] | CorpusIndex,
    n: int = 4,
) -> dict[int, dict[str, list[float]]]:
    """
    Calcula as três métricas para cada ordem, no formato que
    plot_metrics_by_order e results_to_latex esperam.

    - copied_ratio: copied_ngram_ratio de cada melodia isoladamente
      (melodias com menos de n eventos são ignoradas);
    - longest_run: longest_copied_run de cada melodia;
    - diversity: um único valor para o conjunto da ordem.
    """
    index = _as_index(corpus)
    results: dict[int, dict[str, list[float]]] = {}
    for order in sorted(generated_by_order):
        melodies = generated_by_order[order]
        ratios = [copied_ngram_ratio([m], index, n) for m in melodies]
        results[order] = {
            "copied_ratio": [r for r in ratios if not math.isnan(r)],
            "longest_run": [float(longest_copied_run(m, index)) for m in melodies],
            "diversity": [diversity(melodies)],
        }
    return results


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Média e desvio padrão amostral (0 com um único valor; NaN se vazio)."""
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------

def plot_metrics_by_order(
    results: Results,
    path: str | Path,
    labels: Mapping[str, str] | None = None,
    width_in: float = 3.4,
) -> None:
    """
    Um painel por métrica, empilhados, com a ordem no eixo x e
    média ± desvio padrão (barras de erro) no eixo y.

    O padrão de 3,4 polegadas de largura corresponde a uma coluna de um
    artigo em duas colunas (\\columnwidth em IEEE/ACM). Inclua no LaTeX com
    \\includegraphics[width=\\columnwidth]{...}, sem reescalar, para que a
    fonte de 8 pt fique com o tamanho certo.

    Métricas com um único valor por ordem (diversidade) aparecem sem barra.
    """
    import matplotlib
    matplotlib.use("Agg")                    # sem janela; só gera o arquivo
    import matplotlib.pyplot as plt

    labels = {**PLOT_LABELS, **(labels or {})}
    orders = sorted(results)
    metrics = list(results[orders[0]].keys())

    style = {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",   # $n$ na mesma família do texto
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "pdf.fonttype": 42,      # fontes TrueType embutidas (exigido por IEEE/ACM)
        "ps.fonttype": 42,
    }

    with plt.rc_context(style):
        height = 1.15 * len(metrics) + 0.35
        fig, axes = plt.subplots(len(metrics), 1, sharex=True,
                                 figsize=(width_in, height), squeeze=False)

        for ax, metric in zip(axes[:, 0], metrics):
            stats = [_mean_std(results[o][metric]) for o in orders]
            means = [m for m, _ in stats]
            stds = [s for _, s in stats]
            has_spread = any(len(results[o][metric]) > 1 for o in orders)

            ax.errorbar(orders, means, yerr=stds if has_spread else None,
                        marker="o", color="0.15", ecolor="0.45",
                        capsize=2.5, elinewidth=0.8)
            ax.set_ylabel(labels.get(metric, metric))

            # Frações ficam em [0, 1] com uma pequena margem.
            if metric in ("copied_ratio", "diversity"):
                ax.set_ylim(-0.05, 1.05)

        axes[-1, 0].set_xlabel("Markov order")
        axes[-1, 0].set_xticks(orders)
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout(pad=0.3, h_pad=0.5)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tabela LaTeX
# ---------------------------------------------------------------------------

def results_to_latex(
    results: Results,
    labels: Mapping[str, str] | None = None,
    decimals: Mapping[str, int] | int = 2,
    caption: str = "Originality and diversity of generated melodies by Markov order "
                   "(mean $\\pm$ standard deviation). Longest run in events.",
    label: str = "tab:results",
) -> str:
    """
    Tabela LaTeX (pacote booktabs) com uma linha por ordem e uma coluna por
    métrica, no formato média ± desvio. Métricas com um único valor por
    ordem aparecem sem o desvio.

    `decimals` pode ser um inteiro para todas as métricas ou um dicionário
    por métrica, ex.: {"longest_run": 1}. Os rótulos são inseridos como
    LaTeX, sem escape — podem conter $...$.

    No preâmbulo: \\usepackage{booktabs}
    """
    labels = {**TABLE_LABELS, **(labels or {})}
    orders = sorted(results)
    metrics = list(results[orders[0]].keys())

    def places(metric: str) -> int:
        if isinstance(decimals, int):
            return decimals
        return decimals.get(metric, 2)

    def cell(metric: str, values: Sequence[float]) -> str:
        mean, std = _mean_std(values)
        d = places(metric)
        if math.isnan(mean):
            return "--"
        if len(values) == 1:
            return f"${mean:.{d}f}$"
        return f"${mean:.{d}f} \\pm {std:.{d}f}$"

    header = " & ".join(["Order"] + [labels.get(m, m) for m in metrics])
    rows = [
        " & ".join([str(o)] + [cell(m, results[o][m]) for m in metrics]) + r" \\"
        for o in orders
    ]

    # O template do ISMIR pede a legenda abaixo da tabela.
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{c" + "c" * len(metrics) + "}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
        *(f"    {row}" for row in rows),
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"
