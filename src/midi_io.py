"""
midi_io.py — Conversão de melodias (listas de tuplas) para arquivos MIDI.

Uma melodia é uma lista de (altura_midi, duracao_em_seminimas), onde
altura 0 representa pausa — o mesmo formato produzido por corpus.py.
"""

from __future__ import annotations

from pathlib import Path

from music21 import instrument, meter, note, stream, tempo

REST = 0                 # código usado para pausas
BEATS_PER_MEASURE = 4.0  # compasso 4/4, em semínimas
EPS = 1e-6               # tolerância para comparar durações em ponto flutuante


def pad_last_measure(
    melody: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """
    Estende a última nota (ou pausa) para que a duração total da melodia
    seja um múltiplo de 4 semínimas, completando o último compasso 4/4.

    Devolve uma nova lista; a melodia original não é modificada.
    Se a melodia já termina no fim de um compasso, ou se está vazia,
    devolve uma cópia sem alterações.

    Observação: a conta assume que a melodia começa no tempo 1 do compasso.
    Melodias com anacruse terão o último compasso calculado a partir do
    início da anacruse, e não do primeiro compasso completo.
    """
    if not melody:
        return []

    padded = list(melody)
    total = sum(dur for _, dur in padded)

    # Quanto falta para fechar o compasso atual.
    remainder = total % BEATS_PER_MEASURE
    missing = BEATS_PER_MEASURE - remainder

    # Já fechado? (remainder ~ 0 ou ~ 4 por causa de arredondamento)
    if remainder < EPS or missing < EPS:
        return padded

    last_pitch, last_dur = padded[-1]
    padded[-1] = (last_pitch, last_dur + missing)
    return padded


def melody_to_midi(
    melody: list[tuple[int, float]],
    path: str,
    tempo_bpm: int = 80,
) -> None:
    """
    Grava uma melodia como arquivo .mid.

    Parâmetros
    ----------
    melody : list[tuple[int, float]]
        Sequência de (altura_midi, duração em semínimas); altura 0 = pausa.
    path : str
        Caminho do arquivo de saída. Pastas inexistentes são criadas.
    tempo_bpm : int
        Andamento em semínimas por minuto (padrão: 80).

    Levanta ValueError se alguma altura estiver fora de 0–127 ou se alguma
    duração não for positiva.
    """
    s = stream.Stream()

    # Cabeçalho: instrumento, fórmula de compasso e marca de tempo,
    # todos inseridos no início (offset 0).
    s.insert(0, instrument.Piano())
    s.insert(0, meter.TimeSignature("4/4"))
    s.insert(0, tempo.MetronomeMark(number=tempo_bpm))

    # Notas e pausas em sequência: append() posiciona cada elemento
    # logo após o anterior.
    for i, (midi_pitch, dur) in enumerate(melody):
        if not 0 <= midi_pitch <= 127:
            raise ValueError(f"evento {i}: altura MIDI fora de 0–127: {midi_pitch}")
        if dur <= 0:
            raise ValueError(f"evento {i}: duração deve ser positiva: {dur}")

        if midi_pitch == REST:
            el = note.Rest(quarterLength=dur)
        else:
            el = note.Note(midi=midi_pitch, quarterLength=dur)
        s.append(el)

    # Garante que a pasta de destino existe e grava o arquivo.
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    s.write("midi", fp=str(out))
