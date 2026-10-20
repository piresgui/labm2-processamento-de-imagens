# Lab M2 — Segmentação, componentes conexos e morfologia

Entrega consolidada dos laboratórios **M2.1** (máscara binária e segmentação) e **M2.2** (componentes conexos e sequência morfológica) de Processamento de Imagens 2026-02.

- Aluno: Guilherme Pires
- Linguagem: Python 3.10+
- Um único projeto: o M2.2 amplia o mesmo código do M2.1 (`segment` continua funcionando, `analyze` foi acrescentada).

## Dependências

- `numpy`, `opencv-python` (leitura/escrita, limiarização, HSV, morfologia, rotulação)
- `pytest` (testes)

## Preparação do ambiente

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (cmd/PowerShell): .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Sem `pip install -e .`, basta definir `PYTHONPATH=src` antes de executar `python -m pdi_lab`.

```bash
python -m pdi_lab --help
```

## Interface (contrato técnico)

```bash
python -m pdi_lab --input <arquivo> --output <arquivo-ou-prefixo> --operation <segment|compare|analyze> [opções]
```

Sucesso retorna código `0`; erros (arquivo inexistente, imagem inválida, método/operação desconhecidos, parâmetro fora da faixa, saída JPEG) imprimem `erro: ...` no stderr e retornam `2`.

### `segment` (M2.1)

| Opção | Uso |
|---|---|
| `--method fixed\|otsu\|adaptive\|hsv` | método de segmentação |
| `--threshold N` | limiar fixo (0..255). Pixel **> N** vira 255 (`THRESH_BINARY`) |
| `--invert` | inverte a máscara (objetos escuros) |
| `--blur K` | pré-processamento opcional: Gaussiano K×K (0 = desligado, K ímpar ≥ 3) |
| `--block-size B`, `--c C` | limiar adaptativo gaussiano (B ímpar ≥ 3; padrão 31 e 5) |
| `--hsv-lower H,S,V`, `--hsv-upper H,S,V` | faixa HSV (H 0..179, S/V 0..255, escala OpenCV). Se `H_lower > H_upper`, a faixa cruza a descontinuidade do matiz e é tratada como união `[H_lower,179] ∪ [0,H_upper]` |
| `--reference mask.png` | opcional: máscara de referência para FP/FN/IoU e mapa de erros |

Saídas: máscara PNG em `--output` (mesmo tamanho, 1 canal, uint8, apenas 0/255), `<nome>_stats.json` com contagem e proporção de primeiro plano, e `<nome>_errors.png` quando há referência. A CLI imprime `foreground_pixels` e `foreground_ratio`.

```bash
python -m pdi_lab --input images/input/cena.png --output images/output/cena_mask.png --operation segment --method fixed --threshold 120
python -m pdi_lab --input images/input/cena.png --output images/output/cena_otsu.png --operation segment --method otsu --invert --blur 5
python -m pdi_lab --input images/input/cena.png --output images/output/cena_adapt.png --operation segment --method adaptive --block-size 51 --c 10 --invert
python -m pdi_lab --input images/input/cena.png --output images/output/cena_hsv.png --operation segment --method hsv --hsv-lower 0,80,50 --hsv-upper 30,255,255
```

### `compare` (M2.1)

Roda várias configurações definidas em um JSON (`--config`) sobre a mesma imagem e gera, com o prefixo `--output`:
`<prefixo>_<config>_mask.png`, `<prefixo>_<config>_overlay.png` (máscara em verde + contorno amarelo sobre a original), `<prefixo>_compare.csv` (proporção de primeiro plano por configuração) e `<prefixo>_compare_panel.png` (original | máscara | sobreposição [| erros]). Com referência, adiciona FP/FN/IoU e `<prefixo>_<config>_errors.png` (verde = acerto, **vermelho = FP**, **azul = FN**).

```bash
python -m pdi_lab --input images/input/cena.png --output results/cena/cena --operation compare --config configs/pipeline.json
```

Formato do JSON: `default` vale para todas as imagens e `images.<nome_do_arquivo>` sobrescreve por imagem (`configs`, `chosen`, `analyze`, `reference`). Ver `configs/pipeline.json` e `configs/synthetic.json`.

### `analyze` (M2.2)

```bash
python -m pdi_lab --input images/output/cena_mask.png --output results/cena/cena --operation analyze --connectivity 8 --min-area 200 --morph open:ellipse:5:1,close:ellipse:9:1 --original images/input/cena.png
```

| Opção | Uso |
|---|---|
| `--input` | máscara binária (0 fundo, 255 objeto); outras entradas geram erro |
| `--output` | **prefixo** dos arquivos |
| `--connectivity 4\|8` | conectividade da análise final (padrão **8**) |
| `--min-area N` | inteiro ≥ 0 (padrão 0): componente mantido se `area >= N` |
| `--morph` | sequência `op:forma:tamanho:iteracoes` separada por vírgula; `op` ∈ erode, dilate, open, close; `forma` ∈ rect, ellipse, cross; tamanho ímpar |
| `--original` | opcional: imagem original para o painel |

Arquivos gerados com o prefixo:

| Arquivo | Conteúdo |
|---|---|
| `_stage0_initial.png` | cópia da máscara inicial |
| `_stage<i>_<op>_<forma><tam>x<iter>.png` | saída de cada estágio morfológico |
| `_final_mask.png` | máscara após a sequência |
| `_initial.csv`, `_final.csv` | `label,area,x,y,width,height,cx,cy,kept` (sem o fundo) |
| `_initial_labels.png`, `_labels.png` | rótulos coloridos; caixa verde = mantido, vermelha/cinza = removido pelo filtro |
| `_stages_panel.png` | figura lado a lado: original, estágios, rótulos inicial/final |
| `_before_after.csv` | tabela antes/depois (componentes 4 e 8, brutos, filtrados, áreas) |
| `_summary.json` | conectividade, regra do filtro, sequência, contagens brutas/filtradas, áreas antes/depois |

## Processar todas as imagens

Coloque as fotos em `images/input/` e ajuste `configs/pipeline.json` (por imagem, se necessário). Depois:

```bash
python scripts/run_all.py
```

Para cada imagem: `compare` → `segment` com a configuração escolhida (`chosen`) → `analyze`. Gera `images/output/<nome>_mask.png`, `results/<nome>/...`, as tabelas `results/summary_compare.csv` e `results/summary_analyze.csv` e `results/commands.txt` com todos os comandos executados.

Demonstração com as imagens sintéticas:

```bash
python scripts/make_synthetic.py
python scripts/run_all.py --input-dir images/synthetic --config configs/synthetic.json --results results/synthetic --masks results/synthetic/masks
```

## Testes

```bash
python -m pytest
```

## Estrutura

```text
.
├── README.md, REPORT.md, AI_USAGE.md, lab.json
├── requirements.txt, pyproject.toml
├── configs/            configurações de segmentação e análise (JSON)
├── images/
│   ├── input/          fotos reais (entrada)
│   ├── output/         máscaras finais
│   └── synthetic/      imagens sintéticas e máscaras de referência (testes/demonstração)
├── results/            tabelas, figuras, estágios e resumos
├── scripts/            make_synthetic.py, run_all.py
├── src/pdi_lab/        cli, segment, visualize, analyze, imageio
└── tests/              pytest
```
