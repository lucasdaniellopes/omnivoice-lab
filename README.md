# OmniVoice Lab

Interface web local para testar clonagem de voz com o [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice): upload de vozes de referência, fila de geração com progresso em tempo real e monitoramento da GPU.

## Requisitos

- Windows com GPU NVIDIA (testado numa RTX 3060 Ti 8 GB)
- [uv](https://docs.astral.sh/uv/) (ou Python 3.11 + pip)
- [ffmpeg](https://ffmpeg.org/) no PATH

## Instalação

```
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Uso

```
run.bat
```

O navegador abre em http://127.0.0.1:8765 quando o servidor estiver no ar. Para desligar e liberar a GPU, feche a janela do terminal.

Na primeira execução o modelo `k2-fsa/OmniVoice` (~3 GB) é baixado para `hf_cache/`. Vozes sem transcrição são transcritas com o **Parakeet TDT 0.6B v3** (ONNX int8, CPU, ~650 MB, baixado no primeiro uso). Para usar o Whisper large-v3-turbo na GPU, rode `set OMNIVOICE_ASR=whisper` antes.

Vozes, áudios gerados e histórico ficam em `data/` (fora do git).

## Funcionalidades

- **Vozes**: upload de vários áudios (qualquer formato, convertido via ffmpeg para wav mono 24 kHz) ou gravação pelo microfone. Transcrição opcional; o prompt codificado fica em cache por voz. O ideal é uma referência limpa de 3 a 10 s.
- **Gerar**: clonar (uma ou mais vozes, um job por voz), voice design (`instruct`) ou automático, com todos os parâmetros de geração expostos.
- **Pausas na pontuação**:
  - *Natural*: o modelo decide as pausas dentro de uma duração estimada pelo ritmo da referência. Se ele atropelar a pontuação, use velocidade 0.85–0.9.
  - *Garantidas*: divide o texto na pontuação, gera os trechos em lote e junta com silêncio exato (padrão: 350 ms após `. ! ?`, 120 ms após vírgula, média entre os dois para `; :`, 1,75× para linha em branco).
- **Jobs**: fila com progresso por passo de difusão e por chunk, it/s, audio tok/s, ctx tok/s, RTF, pico de VRAM, player e download em WAV, MP3, OGG (Opus/Vorbis) ou FLAC.
- **GPU**: uso, VRAM, alocação do PyTorch, temperatura, energia e clock, com gráfico.

## Métricas

O OmniVoice não é autoregressivo: é um modelo de difusão mascarada que gera todos os frames de áudio em `num_step` passes (padrão 32).

- **it/s**: passes do transformer por segundo (cond + uncond em lote por causa do CFG).
- **audio tok/s**: tokens de áudio decididos por segundo (frames × 8 codebooks).
- **ctx tok/s**: posições de sequência processadas por segundo, o análogo ao throughput de um LLM.
- **RTF**: tempo de geração / duração do áudio (menor é melhor).

## Licenças

- Código do OmniVoice: Apache 2.0.
- **Pesos do modelo `k2-fsa/OmniVoice`: CC-BY-NC (uso não comercial)**, segundo a página do modelo no Hugging Face.
- Parakeet TDT 0.6B v3: CC-BY-4.0. Whisper: MIT.
- Não clone a voz de ninguém sem consentimento.
