"""
corpus.py — Carregamento das melodias de soprano dos corais de Bach.

Percorre os corais do corpus do music21, extrai a linha do soprano,
normaliza a tonalidade (tudo em Dó) e converte cada melodia numa lista
de tuplas (altura_midi, duracao_em_quarterLength), prontas para alimentar
um modelo de Markov. O resultado fica em cache num arquivo pickle.
"""

from __future__ import annotations

import pickle
import re
import warnings
from collections import Counter
from pathlib import Path

from music21 import corpus, interval, pitch, stream
from music21 import note as m21note
from music21 import chord as m21chord

# Tipo de uma melodia: sequência de (altura MIDI, duração em semínimas).
# Altura 0 representa pausa.
Melody = list[tuple[int, float]]

REST = 0  # código usado para pausas

# O cache fica em <raiz do projeto>/.cache/ (este arquivo está em src/).
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

# Incremente quando mudar a lógica de extração: invalida caches antigos.
CACHE_VERSION = 2

# Nomes de parte (normalizados) que identificam o soprano.
SOPRANO_NAMES = {"soprano", "sopran", "s", "sop", "canto", "cantus", "discantus"}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _normalize(name: str | None) -> str:
    """Minúsculas, sem pontos nem espaços: 'S.' -> 's', 'Soprano ' -> 'soprano'."""
    if not name:
        return ""
    return name.lower().replace(".", "").replace(" ", "").strip()


def _words(name: str | None) -> set[str]:
    """
    Palavras de um nome, só letras e em minúsculas:
    'Soprano Oboe 1 Violin1' -> {'soprano', 'oboe', 'violin'}.
    """
    if not name:
        return set()
    return set(re.findall(r"[a-z]+", name.lower()))


def _find_soprano(score: stream.Score) -> stream.Part | None:
    """
    Devolve a parte do soprano de um coral, ou None se não houver partes.

    Ordem de tentativa:
      1. parte cujo nome (partName), abreviação (partAbbreviation) ou id
         seja exatamente um dos nomes de SOPRANO_NAMES ('Soprano', 'S.');
      2. parte em cujo nome apareça a palavra 'soprano' (ou variante),
         p.ex. 'Soprano 1' ou 'Soprano Oboe 1 Violin1' em corais com
         orquestra, onde a primeira parte costuma ser um instrumento;
      3. parte cujo instrumento seja um soprano;
      4. primeira parte da partitura (nos corais, o soprano vem no topo).
    """
    parts = list(score.parts)
    if not parts:
        return None

    def labels(part: stream.Part) -> tuple:
        return (part.partName, part.partAbbreviation, str(part.id))

    # 1. Correspondência exata do nome, abreviação ou id.
    for part in parts:
        if any(_normalize(c) in SOPRANO_NAMES for c in labels(part)):
            return part

    # 2. A palavra aparece dentro do nome. A abreviação de uma letra ('s')
    #    fica de fora aqui para não casar com palavras soltas por acaso.
    long_names = {n for n in SOPRANO_NAMES if len(n) > 1}
    for part in parts:
        if any(_words(c) & long_names for c in labels(part)):
            return part

    # 3. Pelo instrumento associado à parte.
    for part in parts:
        inst = part.getInstrument(returnDefault=False)
        if inst is not None and _words(inst.instrumentName) & long_names:
            return part

    # 4. Último recurso: a primeira parte.
    return parts[0]


def _semitones_to_c(tonic: pitch.Pitch) -> int:
    """
    Quantos semitons transpor para levar a tônica a Dó, escolhendo o
    caminho mais curto (entre -6 e +5) para não deslocar demais a tessitura.
    """
    shift = (0 - tonic.pitchClass) % 12   # 0..11, sempre para cima
    if shift > 6:
        shift -= 12                        # prefere descer se for mais perto
    return shift


def _part_to_tuples(part: stream.Part) -> Melody:
    """
    Converte uma parte em lista de (altura_midi, duração).

    - Pausas viram altura REST (0); pausas consecutivas são fundidas.
    - Notas ligadas (tie 'continue'/'stop') somam sua duração à nota anterior.
    - Fermatas são ignoradas: no music21 elas são apenas expressões e não
      alteram quarterLength, então basta não tratá-las.
    - Notas de ornamento (duração 0) são descartadas.
    - Acordes, se aparecerem, são reduzidos à nota mais aguda.
    """
    melody: Melody = []

    for el in part.flatten().notesAndRests:
        dur = float(el.quarterLength)
        if dur == 0:  # appoggiaturas/grace notes
            continue

        if isinstance(el, m21note.Rest):
            if melody and melody[-1][0] == REST:
                # Funde com a pausa anterior.
                melody[-1] = (REST, melody[-1][1] + dur)
            else:
                melody.append((REST, dur))
            continue

        if isinstance(el, m21chord.Chord):
            midi = max(p.midi for p in el.pitches)
        else:
            midi = el.pitch.midi

        # Continuação de ligadura: soma à nota anterior, se for a mesma altura.
        is_tie_continuation = el.tie is not None and el.tie.type in ("continue", "stop")
        if is_tie_continuation and melody and melody[-1][0] == midi:
            melody[-1] = (midi, melody[-1][1] + dur)
        else:
            melody.append((midi, dur))

    # Remove pausas no início e no fim (geralmente artefatos de anacruse/codas).
    while melody and melody[0][0] == REST:
        melody.pop(0)
    while melody and melody[-1][0] == REST:
        melody.pop()

    return melody


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def load_bach_sopranos(major_only: bool = True) -> list[Melody]:
    """
    Carrega as melodias de soprano dos corais de Bach, transpostas para Dó.

    Parâmetros
    ----------
    major_only : bool
        Se True, descarta corais analisados como menores.

    Retorna
    -------
    list[list[tuple[int, float]]]
        Uma lista de melodias; cada melodia é uma lista de
        (altura_midi, duracao_em_quarterLength), com pausas como altura 0.
    """
    suffix = "major" if major_only else "all"
    cache_file = CACHE_DIR / f"bach_sopranos_{suffix}_v{CACHE_VERSION}.pkl"

    # Se já processamos antes, devolve direto do cache.
    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)

    melodies: list[Melody] = []
    skipped_no_part = skipped_minor = skipped_empty = 0

    # Iterator() percorre os corais na numeração de Riemenschneider e
    # devolve cada um como music21.stream.Score.
    for score in corpus.chorales.Iterator():
        soprano = _find_soprano(score)
        if soprano is None:
            skipped_no_part += 1
            continue

        # Analisa a tonalidade do coral inteiro (mais confiável que só o
        # soprano, já que a harmonia está nas outras vozes).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = score.analyze("key")

        if major_only and k.mode != "major":
            skipped_minor += 1
            continue

        # Transpõe a parte para Dó (maior ou menor, conforme o modo).
        shift = _semitones_to_c(k.tonic)
        if shift != 0:
            soprano = soprano.transpose(interval.Interval(shift))

        melody = _part_to_tuples(soprano)
        if not melody:
            skipped_empty += 1
            continue

        melodies.append(melody)

    print(
        f"[corpus] {len(melodies)} melodias carregadas "
        f"(ignorados: {skipped_no_part} sem parte, "
        f"{skipped_minor} menores, {skipped_empty} vazios)"
    )

    # Salva o cache para as próximas execuções.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as f:
        pickle.dump(melodies, f)

    return melodies


# ---------------------------------------------------------------------------
# Execução direta: estatísticas rápidas do corpus
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mels = load_bach_sopranos(major_only=True)

    # Número de melodias e de eventos.
    n_events = sum(len(m) for m in mels)
    print(f"\nMelodias: {len(mels)}")
    print(f"Eventos (notas + pausas): {n_events}")

    # Extensão de alturas (desconsiderando pausas).
    pitches = [p for m in mels for p, _ in m if p != REST]
    lo, hi = min(pitches), max(pitches)
    print(
        f"Extensão: {pitch.Pitch(midi=lo).nameWithOctave} (MIDI {lo}) "
        f"a {pitch.Pitch(midi=hi).nameWithOctave} (MIDI {hi}) "
        f"— {hi - lo} semitons"
    )

    # Histograma textual das durações (notas e pausas).
    durations = Counter(d for m in mels for _, d in m)
    total = sum(durations.values())
    max_count = max(durations.values())
    bar_width = 40

    print("\nHistograma de durações (quarterLength):")
    for dur, count in sorted(durations.items()):
        bar = "█" * max(1, round(count / max_count * bar_width))
        print(f"  {dur:>6.3g}  {count:>6}  {count / total:6.1%}  {bar}")
