"""
markov.py — Cadeia de Markov de ordem n sobre eventos (altura_midi, duracao).

Treino
------
Cada melodia do corpus é cercada por tokens especiais:

    START × n, e1, e2, ..., ek, END

e, para cada ordem k = 0, 1, ..., n, contamos quantas vezes cada evento
seguiu cada contexto formado pelos k tokens anteriores. As probabilidades
de transição são as frequências normalizadas (máxima verossimilhança),
ou seja, aprendidas do corpus — nada é definido à mão.

Os tokens START fazem o modelo aprender como as melodias começam
(anacruses típicas dos corais) e o token END, como elas terminam (cadências).

Geração
-------
Começa com n tokens START e sorteia o próximo evento a partir do contexto
dos n últimos tokens, até sortear END ou atingir o tamanho máximo.

- Backoff: se o contexto de ordem n nunca apareceu no corpus, usa o de
  ordem n-1, depois n-2, ..., até a ordem 0 (frequência global dos eventos).
- Temperatura T: as probabilidades são elevadas a 1/T e renormalizadas.
  T < 1 concentra a escolha nas transições mais frequentes; T > 1 achata a
  distribuição e dá mais chance às raras. T = 1 é a distribuição do corpus.
- Tamanho mínimo: antes de `min_len` eventos, END é retirado das opções
  (com backoff se o contexto só levava a END), para evitar peças curtas.
- Semente: cada chamada usa seu próprio random.Random(seed), então a mesma
  configuração com a mesma semente sempre gera a mesma melodia.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence

Event = tuple[int, float]
Melody = list[Event]

# Tokens especiais. São strings, então nunca colidem com eventos (tuplas).
START = "<START>"
END = "<END>"


class MarkovModel:
    """
    Modelo de Markov de ordem `order` com backoff para ordens menores.

    Uso:
        model = MarkovModel(2)
        model.fit(melodies)
        melody = model.generate(max_len=64, temperature=1.0, seed=1)
    """

    def __init__(self, order: int):
        if order < 1:
            raise ValueError(f"a ordem deve ser >= 1, recebi {order}")
        self.order = order
        # tables[k]: contexto (tupla com k tokens) -> (próximos, contagens).
        # Guardamos listas em vez de Counter para sortear sem reconverter.
        self.tables: list[dict[tuple, tuple[list, list[int]]]] = []

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def fit(self, melodies: Sequence[Sequence[Event]]) -> "MarkovModel":
        """Conta as transições de ordem 0 a `order` em todas as melodias."""
        n = self.order
        counts: list[defaultdict[tuple, Counter]] = [
            defaultdict(Counter) for _ in range(n + 1)
        ]

        for melody in melodies:
            tokens = [START] * n + [(int(p), float(d)) for p, d in melody] + [END]
            # Cada posição i >= n é um "próximo" token; os n anteriores são
            # o contexto de ordem n, e seus sufixos, os de ordem menor.
            for i in range(n, len(tokens)):
                nxt = tokens[i]
                for k in range(n + 1):
                    ctx = tuple(tokens[i - k:i])
                    counts[k][ctx][nxt] += 1

        # Congela em listas. A ordem de inserção depende só do corpus,
        # então o sorteio é reprodutível entre execuções.
        self.tables = [
            {ctx: (list(c.keys()), list(c.values())) for ctx, c in table.items()}
            for table in counts
        ]
        return self

    # ------------------------------------------------------------------
    # Geração
    # ------------------------------------------------------------------

    def _distribution(
        self, history: list, allow_end: bool = True,
    ) -> tuple[list, list[int], int]:
        """
        Próximos tokens e contagens para o maior contexto conhecido.
        Devolve também a ordem efetivamente usada (< order indica backoff).

        Com allow_end=False, END é retirado das opções; se o contexto só
        tiver END como continuação (p.ex. uma cadência final), conta como
        contexto desconhecido e o backoff passa para a ordem de baixo.
        """
        for k in range(self.order, -1, -1):
            ctx = tuple(history[len(history) - k:]) if k else ()
            entry = self.tables[k].get(ctx)
            if entry is None:
                continue
            options, counts = entry
            if not allow_end and END in options:
                pairs = [(o, c) for o, c in zip(options, counts) if o != END]
                if not pairs:
                    continue
                options = [o for o, _ in pairs]
                counts = [c for _, c in pairs]
            return options, counts, k
        raise RuntimeError("modelo vazio: chame fit() antes de generate()")

    @staticmethod
    def _apply_temperature(counts: list[int], temperature: float) -> list[float]:
        """p_i ∝ c_i^(1/T). Divide pelo máximo antes para evitar overflow."""
        if temperature == 1.0:
            return [float(c) for c in counts]
        top = max(counts)
        return [(c / top) ** (1.0 / temperature) for c in counts]

    def generate(
        self,
        max_len: int = 64,
        temperature: float = 1.0,
        seed: int | None = None,
        min_len: int = 0,
    ) -> Melody:
        """
        Gera uma melodia com no máximo `max_len` eventos.

        Para quando sorteia END. Enquanto a melodia tiver menos de `min_len`
        eventos, END não pode ser sorteado: sem isso, contextos cadenciais
        (p.ex. uma nota longa na tônica) encerram a peça com poucos eventos.
        Se o tamanho máximo for atingido antes de END, a melodia termina onde
        parou (o fechamento do compasso é feito na exportação, por
        midi_io.pad_last_measure).
        """
        if not self.tables:
            raise RuntimeError("modelo não treinado: chame fit() antes")
        if temperature <= 0:
            raise ValueError("a temperatura deve ser > 0")

        rng = random.Random(seed)
        history: list = [START] * self.order
        melody: Melody = []
        self.last_backoffs = 0   # quantas vezes caiu para uma ordem menor

        while len(melody) < max_len:
            allow_end = len(melody) >= max(min_len, 1)
            options, counts, k = self._distribution(history, allow_end)
            if k < self.order:
                self.last_backoffs += 1

            weights = self._apply_temperature(counts, temperature)
            nxt = rng.choices(options, weights=weights, k=1)[0]

            if nxt == END:
                break
            if nxt == START:   # nunca é "próximo" no treino; por segurança
                continue

            melody.append(nxt)
            history.append(nxt)

        return melody

    # ------------------------------------------------------------------
    # Estatísticas do modelo (para o resumo)
    # ------------------------------------------------------------------

    def n_contexts(self, k: int | None = None) -> int:
        """Número de contextos distintos vistos na ordem k (padrão: a do modelo)."""
        return len(self.tables[self.order if k is None else k])

    def conditional_entropy(self, k: int | None = None) -> float:
        """
        Entropia condicional média H(próximo | contexto), em bits, na ordem k,
        ponderada pela frequência de cada contexto. Quanto menor, mais
        determinística é a escolha do próximo evento — sinal de que o modelo
        tende a reproduzir o corpus.
        """
        table = self.tables[self.order if k is None else k]
        total = 0
        acc = 0.0
        for _, counts in table.values():
            n = sum(counts)
            h = -sum((c / n) * math.log2(c / n) for c in counts)
            acc += n * h
            total += n
        return acc / total if total else math.nan

    def mean_branching(self, k: int | None = None) -> float:
        """Número médio de continuações distintas por contexto na ordem k."""
        table = self.tables[self.order if k is None else k]
        if not table:
            return math.nan
        return sum(len(opts) for opts, _ in table.values()) / len(table)
