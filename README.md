# TP1 — Composição Musical Algorítmica com Cadeias de Markov

**Repositório:** https://github.com/Lucas-Junqueira/TP1_IA_Musica

Este projeto gera melodias em domínio simbólico (MIDI) com **cadeias de Markov de ordem n**. As probabilidades de transição são aprendidas de um corpus real.

**Restrição estilística:** melodias no estilo das **linhas de soprano dos corais de J. S. Bach**. O corpus é o de corais disponível no `music21` (`music21.corpus.chorales`).

---

## Sumário

- [Método](#método)
- [Instalação](#instalação)
- [Como gerar melodias](#como-gerar-melodias)
- [Como reproduzir as músicas entregues](#como-reproduzir-as-músicas-entregues)
- [Renderização de áudio](#renderização-de-áudio)
- [Experimentos e análise](#experimentos-e-análise)
- [Músicas geradas](#músicas-geradas)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Uso de IA](#uso-de-ia)

---

## Método

**Pré-processamento do corpus**

- Extração da voz de soprano de cada coral.
- Seleção dos corais em modo maior.
- Transposição de todas as melodias para Dó maior.

**Representação**

- Cada estado é uma tupla `(altura MIDI, duração em semínimas)`. Altura `0` representa pausa.
- Cada melodia recebe os tokens `START` no início e `END` no fim.

**Treino**

- Contagem das transições `contexto (n estados anteriores) → próximo estado`.
- As probabilidades são as frequências normalizadas (máxima verossimilhança).

**Geração**

- Amostragem estado a estado até encontrar `END` ou atingir o tamanho máximo.
- **Backoff** para ordens menores quando um contexto não foi visto no corpus.
- **Temperatura** opcional, que torna a amostragem mais ou menos conservadora.

**Hiperparâmetros**

| Parâmetro | Descrição | Valores usados |
|---|---|---|
| `--order` | Ordem da cadeia (memória de n estados) | 1, 2, 3, 4 |
| `--temperature` | Temperatura da amostragem | 1.0 (padrão) |
| `--seed` | Semente aleatória (reprodutibilidade) | 0–49 |
| `--max-len` | Número máximo de notas por melodia | TODO |

---

## Instalação

Requer **Python 3.10+**.

```bash
git clone https://github.com/Lucas-Junqueira/TP1_IA_Musica.git
cd TP1_IA_Musica

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Na primeira execução, o corpus é processado e salvo em cache. As execuções seguintes são mais rápidas.

---

## Como gerar melodias

```bash
# 1 melodia, ordem 2, seed 1
python src/generate.py --order 2 --seed 1

# 5 melodias diferentes com a mesma configuração
python src/generate.py --order 2 --seed 1 --n 5

# variando a ordem do modelo
python src/generate.py --order 1 --seed 1
python src/generate.py --order 3 --seed 1

# variando a temperatura
python src/generate.py --order 2 --seed 1 --temperature 1.5
```

Os arquivos são salvos em `outputs/midi/` com o nome `ord{ordem}_temp{temperatura}_seed{seed}.mid`.

---

## Como reproduzir as músicas entregues

```bash
python src/generate.py --order 1 --seed 1 --temperature 1.0
python src/generate.py --order 2 --seed 1 --temperature 1.0
python src/generate.py --order 3 --seed 1 --temperature 1.0
python src/generate.py --order 2 --seed 2 --temperature 1.0
python src/generate.py --order 2 --seed 3 --temperature 1.0

python scripts/render_audio.py --soundfont caminho/para/FluidR3_GM.sf2
```

---

## Renderização de áudio

O MIDI gerado é convertido em áudio **apenas para audição**, usando o [FluidSynth](https://www.fluidsynth.org/) com a soundfont livre **FluidR3_GM**.

Instalação do FluidSynth:

- **Ubuntu/Debian:** `sudo apt install fluidsynth fluid-soundfont-gm`, que instala a soundfont em `/usr/share/sounds/sf2/FluidR3_GM.sf2`.
- **macOS:** `brew install fluid-synth`, e depois baixar a soundfont FluidR3_GM.
- **Windows:** baixar o binário em https://github.com/FluidSynth/fluidsynth/releases e adicioná-lo ao `PATH`.

Para converter todos os MIDI de `outputs/midi/` em áudio em `outputs/audio/`:

```bash
python scripts/render_audio.py --soundfont caminho/para/FluidR3_GM.sf2
```

Como alternativa, os arquivos `.mid` podem ser abertos e exportados pelo [MuseScore](https://musescore.org/).

---

## Experimentos e análise

O script de experimentos gera 50 melodias para cada ordem (1 a 4):

```bash
python scripts/run_experiments.py
```

Em seguida, a análise mede o trade-off entre ordem baixa e ordem alta:

```bash
python src/analysis.py
```

Métricas calculadas:

- **% de n-gramas copiados:** a fração de n-gramas gerados que existem literalmente no corpus. Mede a memorização.
- **Maior trecho copiado:** o comprimento do maior trecho idêntico a algum coral do corpus.
- **Diversidade:** a fração de melodias distintas entre si.

Saídas:

- `outputs/metrics_by_order.pdf`: gráfico usado no resumo.
- `outputs/results_table.tex`: tabela usada no resumo.

---

## Músicas geradas

| Arquivo | Ordem | Temperatura | Seed | Observação |
|---|---|---|---|---|
| [ord1_temp1.0_seed1.wav](outputs/audio/ord1_temp1.0_seed1.wav) | 1 | 1.0 | 1 | TODO |
| [ord2_temp1.0_seed1.wav](outputs/audio/ord2_temp1.0_seed1.wav) | 2 | 1.0 | 1 | TODO |
| [ord3_temp1.0_seed1.wav](outputs/audio/ord3_temp1.0_seed1.wav) | 3 | 1.0 | 1 | TODO |
| [ord2_temp1.0_seed2.wav](outputs/audio/ord2_temp1.0_seed2.wav) | 2 | 1.0 | 2 | TODO |
| [ord2_temp1.0_seed3.wav](outputs/audio/ord2_temp1.0_seed3.wav) | 2 | 1.0 | 3 | TODO |

Os arquivos MIDI correspondentes estão em `outputs/midi/`.

---

## Estrutura do repositório

```
TP1_IA_Musica/
├── README.md
├── requirements.txt
├── src/
│   ├── corpus.py            # extração e normalização das melodias de soprano
│   ├── markov.py            # modelo de Markov: treino, backoff, temperatura, geração
│   ├── midi_io.py           # exportação das melodias para MIDI
│   ├── generate.py          # interface de linha de comando para gerar melodias
│   └── analysis.py          # métricas de memorização/diversidade e gráficos
├── scripts/
│   ├── run_experiments.py   # gera melodias para várias ordens e seeds
│   └── render_audio.py      # converte MIDI em áudio com FluidSynth
├── outputs/
│   ├── midi/                # melodias geradas (.mid)
│   └── audio/               # áudios renderizados (.wav/.mp3)
└── paper/                   # fonte LaTeX do resumo ISMIR
```

---

## Uso de IA

Conforme as regras do trabalho, ferramentas de IA foram usadas **apenas como apoio**: código auxiliar, planejamento e formatação. A implementação do método de geração musical (`src/markov.py`) e o texto do resumo **foram feitos integralmente pelo autor**.

| Ferramenta | O que foi gerado | Objetivo | Link da conversa |
|---|---|---|---|
| Claude (Anthropic) | Roteiro de implementação, lista de prompts para código auxiliar e este README | Planejamento e documentação | TODO: link da conversa do roteiro |
| Claude (Anthropic) | `requirements.txt` e `.gitignore` | Infraestrutura do projeto | [conversa](https://claude.ai/share/1614ab09-147f-4286-b95d-631b4c0ef959) |
| Claude (Anthropic) | `src/corpus.py` | Pré-processamento do corpus de Bach | [conversa](https://claude.ai/share/1614ab09-147f-4286-b95d-631b4c0ef959) |
| Claude (Anthropic) | `src/midi_io.py` | Exportação das melodias para MIDI | [conversa](https://claude.ai/share/1614ab09-147f-4286-b95d-631b4c0ef959) |
| Claude (Anthropic) | `scripts/render_audio.py` | Conversão de MIDI para áudio | [conversa](https://claude.ai/share/1614ab09-147f-4286-b95d-631b4c0ef959) |
| Claude (Anthropic) | `src/generate.py` e `scripts/run_experiments.py` | Interface de linha de comando e execução dos experimentos (chamando o modelo implementado pelo autor) | [conversa](https://claude.ai/share/1614ab09-147f-4286-b95d-631b4c0ef959) |
| Claude (Anthropic) | `src/analysis.py` | Métricas de memorização/diversidade e gráficos | [conversa](https://claude.ai/share/1614ab09-147f-4286-b95d-631b4c0ef959) |
| Claude (Anthropic) | Ajuda com formatação LaTeX do template ISMIR | Formatação do resumo (sem geração de conteúdo) | [conversa](https://claude.ai/share/1614ab09-147f-4286-b95d-631b4c0ef959) |

Todo código gerado por IA foi revisado e adaptado pelo autor antes de ser incluído no repositório.
