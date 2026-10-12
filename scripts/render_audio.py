"""
render_audio.py — Converte os .mid de outputs/midi/ em .wav (e .mp3) em
outputs/audio/, usando a linha de comando do FluidSynth.

Uso:
    python scripts/render_audio.py --soundfont /caminho/FluidR3_GM.sf2
    python scripts/render_audio.py --soundfont FluidR3_GM.sf2 --mp3
    python scripts/render_audio.py --soundfont FluidR3_GM.sf2 --no-normalize

Etapas para cada arquivo:
  1. FluidSynth renderiza o MIDI num .wav bruto;
  2. se o ffmpeg estiver instalado, normaliza o volume (filtro loudnorm);
  3. com --mp3, o ffmpeg também gera um .mp3 a partir do .wav final.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Raiz do projeto: este arquivo fica em scripts/, então subimos um nível.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MIDI_DIR = PROJECT_ROOT / "outputs" / "midi"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "outputs" / "audio"

# Alvo de loudness: -16 LUFS integrado, pico real de -1.5 dBTP.
# É o padrão usual para áudio escutado em computador/celular.
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"


# ---------------------------------------------------------------------------
# Instruções de instalação (mostradas quando falta alguma ferramenta)
# ---------------------------------------------------------------------------

INSTALL_HELP = {
    "Linux": (
        "  sudo apt install fluidsynth fluid-soundfont-gm ffmpeg\n"
        "  # soundfont instalada em: /usr/share/sounds/sf2/FluidR3_GM.sf2"
    ),
    "Darwin": (
        "  brew install fluid-synth ffmpeg\n"
        "  # a soundfont FluidR3_GM.sf2 precisa ser baixada à parte"
    ),
    "Windows": (
        "  choco install fluidsynth ffmpeg\n"
        "  # ou baixe o .zip em https://github.com/FluidSynth/fluidsynth/releases\n"
        "  # e adicione a pasta bin/ ao PATH;\n"
        "  # a soundfont FluidR3_GM.sf2 precisa ser baixada à parte"
    ),
}


def install_hint() -> str:
    """Instruções de instalação para o sistema operacional atual."""
    system = platform.system()
    hint = INSTALL_HELP.get(system)
    if hint is None:
        return "  Consulte https://www.fluidsynth.org para instalar o FluidSynth."
    return hint


# ---------------------------------------------------------------------------
# Verificação de dependências
# ---------------------------------------------------------------------------

def find_fluidsynth() -> str:
    """Devolve o caminho do executável fluidsynth ou encerra com mensagem clara."""
    exe = shutil.which("fluidsynth")
    if exe is None:
        sys.exit(
            "ERRO: o FluidSynth não foi encontrado no PATH.\n"
            "Instale com:\n" + install_hint() + "\n"
            "Depois feche e reabra o terminal e rode o script de novo."
        )
    return exe


def find_ffmpeg(required: bool) -> str | None:
    """
    Devolve o caminho do ffmpeg, ou None se não estiver instalado.
    Se `required` for True (pedido de .mp3), a ausência é um erro.
    """
    exe = shutil.which("ffmpeg")
    if exe is None:
        msg = (
            "o ffmpeg não foi encontrado no PATH. "
            "Ele é usado para normalizar o volume e gerar .mp3.\n"
            "Instale junto com o FluidSynth:\n" + install_hint()
        )
        if required:
            sys.exit("ERRO: " + msg)
        print("AVISO: " + msg + "\nContinuando sem normalização.\n")
    return exe


# ---------------------------------------------------------------------------
# Conversões
# ---------------------------------------------------------------------------

def run(cmd: list[str]) -> None:
    """
    Executa um comando externo. Em caso de falha, levanta RuntimeError com
    a saída de erro, para que o arquivo seja reportado e os demais sigam.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{Path(cmd[0]).name} falhou:\n{detail}")


def render_wav(fluidsynth: str, soundfont: Path, midi: Path, wav: Path,
               sample_rate: int, gain: float) -> None:
    """
    Renderiza um MIDI em WAV com o FluidSynth, sem tocar nada no alto-falante.

    -n  não abre entrada MIDI      -i  não abre o shell interativo
    -g  ganho (volume do sintetizador)
    -r  taxa de amostragem         -F  grava o áudio neste arquivo
    A soundfont precisa vir ANTES do arquivo MIDI na linha de comando.
    """
    run([
        fluidsynth, "-n", "-i",
        "-g", str(gain),
        "-r", str(sample_rate),
        "-F", str(wav),
        str(soundfont), str(midi),
    ])
    # Algumas versões saem com código 0 mesmo sem gerar áudio.
    if not wav.exists() or wav.stat().st_size <= 44:  # 44 bytes = só o cabeçalho WAV
        raise RuntimeError(f"fluidsynth não gerou áudio para {midi.name}")


def normalize_wav(ffmpeg: str, src: Path, dst: Path, sample_rate: int) -> None:
    """
    Normaliza a loudness com o filtro loudnorm do ffmpeg.
    O loudnorm reamostra internamente para 192 kHz; -ar volta à taxa original.
    """
    run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-af", LOUDNORM,
        "-ar", str(sample_rate),
        str(dst),
    ])


def wav_to_mp3(ffmpeg: str, wav: Path, mp3: Path) -> None:
    """Codifica em MP3 com VBR de alta qualidade (-q:a 2, ~190 kbps)."""
    run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(wav),
        "-codec:a", "libmp3lame", "-q:a", "2",
        str(mp3),
    ])


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte os .mid de outputs/midi/ em áudio com o FluidSynth.",
    )
    parser.add_argument("--soundfont", "-s", required=True, type=Path,
                        help="caminho do arquivo FluidR3_GM.sf2")
    parser.add_argument("--midi-dir", type=Path, default=DEFAULT_MIDI_DIR,
                        help="pasta de entrada (padrão: outputs/midi)")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR,
                        help="pasta de saída (padrão: outputs/audio)")
    parser.add_argument("--mp3", action="store_true",
                        help="também gera .mp3 (exige ffmpeg)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="não normaliza o volume com ffmpeg")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="taxa de amostragem em Hz (padrão: 44100)")
    parser.add_argument("--gain", type=float, default=0.8,
                        help="ganho do FluidSynth, 0.0–10.0 (padrão: 0.8)")
    parser.add_argument("--overwrite", action="store_true",
                        help="reconverte arquivos que já têm .wav na saída")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Verificações antes de começar ---------------------------------
    fluidsynth = find_fluidsynth()

    want_ffmpeg = args.mp3 or not args.no_normalize
    ffmpeg = find_ffmpeg(required=args.mp3) if want_ffmpeg else None
    normalize = ffmpeg is not None and not args.no_normalize

    soundfont = args.soundfont.expanduser().resolve()
    if not soundfont.is_file():
        sys.exit(f"ERRO: soundfont não encontrada: {soundfont}")

    if not args.midi_dir.is_dir():
        sys.exit(f"ERRO: pasta de MIDI não encontrada: {args.midi_dir}")

    midis = sorted(args.midi_dir.glob("*.mid")) + sorted(args.midi_dir.glob("*.midi"))
    if not midis:
        sys.exit(f"Nenhum arquivo .mid em {args.midi_dir}")

    args.audio_dir.mkdir(parents=True, exist_ok=True)

    # --- Conversão arquivo a arquivo -------------------------------------
    ok = skipped = failed = 0
    for midi in midis:
        wav = args.audio_dir / (midi.stem + ".wav")
        mp3 = wav.with_suffix(".mp3")

        if wav.exists() and not args.overwrite:
            print(f"  = {midi.name} (já existe; use --overwrite)")
            skipped += 1
            continue

        # O áudio bruto vai para um arquivo temporário na pasta de saída.
        # Fica dentro de outputs/audio/, não na raiz, então o .gitignore
        # não o cobre; por isso ele é sempre apagado no bloco finally.
        raw = wav.with_name(wav.stem + ".raw.wav")
        try:
            render_wav(fluidsynth, soundfont, midi, raw,
                       args.sample_rate, args.gain)
            if normalize:
                normalize_wav(ffmpeg, raw, wav, args.sample_rate)
            else:
                raw.replace(wav)
            if args.mp3:
                wav_to_mp3(ffmpeg, wav, mp3)
        except RuntimeError as err:
            print(f"  ✗ {midi.name}\n{err}\n")
            failed += 1
            continue
        finally:
            raw.unlink(missing_ok=True)

        extras = " + .mp3" if args.mp3 else ""
        norm = " (normalizado)" if normalize else ""
        print(f"  ✓ {midi.name} → {wav.name}{extras}{norm}")
        ok += 1

    print(f"\n{ok} convertidos, {skipped} pulados, {failed} com erro "
          f"→ {args.audio_dir}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
