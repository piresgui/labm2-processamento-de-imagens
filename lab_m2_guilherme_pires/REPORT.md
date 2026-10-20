# Relatório — Lab M2 (M2.1 + M2.2)

Aluno: Guilherme Pires — Processamento de Imagens 2026-02 — Python + OpenCV

## M2.1 — Máscara binária e segmentação

No início do Lab M2.1, dei continuidade ao projeto dos Labs M1, agora com foco em segmentação: separar o objeto de interesse do fundo por meio de uma máscara binária. Por ser o primeiro marco de um projeto que será ampliado no M2.2, comecei pela base do programa. Estruturei o projeto em Python conforme o contrato técnico e implementei a interface pdi_lab com a operação segment, usando OpenCV para o limiar fixo e o método de Otsu, com opção de inversão para objetos escuros. A máscara é salva em PNG com o mesmo tamanho da entrada, um único canal e somente os valores 0 e 255. Entradas inválidas geram erro com código diferente de zero. Escrevi testes básicos e registrei uma execução, ainda com uma imagem sintética.

A execução registrada (`results/m2_1_synthetic_run.txt`) usou `images/synthetic/square_light.png` (quadrado 50×50 de intensidade 220 sobre fundo 30, imagem 100×100) com `--method fixed --threshold 120` e resultou em 2500 pixels de primeiro plano, proporção 0,250000 — exatamente o esperado. Nos testes, o limiar fixo foi verificado com pixels 119, 120 e 121 e T = 120: só o 121 vira 255, porque a regra adotada é `pixel > T` (`THRESH_BINARY` do OpenCV).

### Objeto e imagens reais

O objeto de interesse são **objetos pretos sobre a mesa branca**: um disco preto de borracha e uma armação de óculos preta. Conta como objeto toda a peça preta (armação, hastes e disco); a mesa, a lente transparente e as sombras são fundo. As duas fotos são minhas, tiradas com o celular (2576×1932 pixels), e estão sem alteração em `images/input/`:

- `disco.jpg` — **condição favorável**: um único disco, bem destacado sobre a mesa clara. Ainda assim há uma sombra suave à direita e a borda escura do monitor no topo da foto.
- `oculos_disco.jpg` — **dificuldade real**: dois objetos (óculos e disco) **sobrepostos**, lente transparente que deixa ver o disco através dela, reflexos na armação e faixas escuras do monitor no topo e da borda da mesa embaixo.

### Configurações e parâmetros

Todas estão em `configs/pipeline.json` e são reproduzíveis por `python scripts/run_all.py` (comandos exatos em `results/commands.txt`). Como o objeto é escuro, as configurações em cinza usam `--invert`, para que o objeto fique em 255.

| Nome | Método | Parâmetros |
|---|---|---|
| fixed_t120_inv | limiar fixo em cinza | T = 120, invertido, sem blur |
| fixed_t80_inv | limiar fixo em cinza | T = 80, invertido, sem blur |
| otsu_inv | Otsu | blur Gaussiano 5×5, invertido (limiar encontrado: 95 no disco e 99 nos óculos) |
| adaptive_b51_c10_inv | adaptativo gaussiano | bloco 51, C = 10, blur 5×5, invertido |
| hsv_dark | faixa HSV | H 0–179, S 0–255, V 0–70 (pixels escuros), blur 5×5 |

Pré-processamento: usei Gaussiano 5×5 no Otsu, no adaptativo e no HSV para reduzir a textura da mesa e o ruído da câmera. O limiar fixo foi rodado sem blur, como referência de comparação. Como o objeto é preto (sem matiz definido), a faixa HSV relevante é a de **V baixo**. Por isso a faixa cobre todo H e não há faixa cruzando a descontinuidade do matiz neste objeto (esse caso está implementado e testado em `tests/test_segment_m21_extra.py`).

### Tabela comparativa (proporção de primeiro plano)

Fonte: `results/summary_compare.csv`.

| Imagem | Configuração | Pixels 255 | Proporção |
|---|---|---|---|
| disco.jpg | fixed T=120 inv | 1 242 259 | 0,2496 |
| disco.jpg | fixed T=80 inv | 976 492 | 0,1962 |
| disco.jpg | Otsu inv (T=95) | 1 020 543 | 0,2051 |
| disco.jpg | adaptativo B=51 C=10 inv | 110 042 | 0,0221 |
| disco.jpg | HSV V ≤ 70 | 961 547 | 0,1932 |
| oculos_disco.jpg | fixed T=120 inv | 1 760 922 | 0,3538 |
| oculos_disco.jpg | fixed T=80 inv | 1 370 704 | 0,2754 |
| oculos_disco.jpg | Otsu inv (T=99) | 1 536 994 | 0,3088 |
| oculos_disco.jpg | adaptativo B=51 C=10 inv | 215 233 | 0,0432 |
| oculos_disco.jpg | HSV V ≤ 70 | 1 279 287 | 0,2570 |

Como validação do pipeline, rodei a mesma comparação nas imagens sintéticas, que têm máscara de referência (`results/synthetic/summary_compare.csv`):

| Imagem sintética | Configuração | Proporção | FP | FN | IoU |
|---|---|---|---|---|---|
| square_light | fixed T=120 | 0,2500 | — | — | — |
| square_light | Otsu (T=30) | 0,2500 | — | — | — |
| square_light | adaptativo B=51, C=10 | 0,7256 | — | — | — |
| square_light | fixed T=120 invertido | 0,7500 | — | — | — |
| coins_scene | fixed T=120 invertido | 0,0998 | 0 | 1857 | 0,6735 |
| coins_scene | Otsu invertido (T=176) | 0,3348 | 7167 | 0 | 0,4425 |
| coins_scene | adaptativo B=51, C=10 invertido | 0,1826 | 2690 | 1368 | 0,5156 |
| coins_scene | HSV H 5–25, S ≥ 120, V ≥ 80 | 0,1410 | 0 | 274 | 0,9518 |

Na cena sintética (círculos alaranjados sobre fundo com gradiente e ruído), o Otsu invertido incorporou a região mais escura do gradiente (FP em vermelho em `results/synthetic/coins_scene/coins_scene_compare_panel.png`). O fixo T=120 perdeu pixels dos círculos por causa do ruído (FN em azul), e o HSV foi o melhor, porque a cor separa objeto e fundo independentemente do brilho. No quadrado, a inversão produz o complemento (0,75), o que confirma que ela só deve ser usada para objetos escuros.

### Análise visual de FP/FN nas imagens reais

Figuras: `results/disco/disco_compare_panel.png` e `results/oculos_disco/oculos_disco_compare_panel.png` (original | máscara | sobreposição em verde com contorno amarelo), e as sobreposições individuais `results/<imagem>/<imagem>_<config>_overlay.png`.

Não tenho máscara de referência das fotos reais, então as observações abaixo são **visuais**; não declaro acurácia, precisão nem revocação.

- **FP visíveis (marcados como objeto, mas são fundo):**
  - Em todas as configurações em cinza e no HSV, a faixa escura do monitor no topo da foto vira primeiro plano. Ela é tão escura quanto o objeto, então nenhum limiar global a separa. Em `oculos_disco.jpg` a borda escura embaixo também aparece.
  - No disco, a **sombra** à direita entra na máscara com T=120 (mancha pontilhada larga na sobreposição) e, em menor grau, com Otsu.
  - Nos óculos, partes da lente sobre o disco e reflexos ficam marcados como objeto, porque através da lente se vê o disco preto.
- **FN visíveis (objeto não marcado):**
  - Com T=80, partes da lateral do disco iluminada e alguns trechos da armação com reflexo ficam de fora.
  - O adaptativo marca só as bordas (o interior uniforme do disco fica abaixo da média local − C), então quase todo o interior é FN. Ele também gera um pontilhado de FP na textura da mesa.

Respostas pedidas no enunciado:

- **(a) Onde o limiar fixo funcionou ou falhou.** Funcionou bem para o corpo do disco e da armação, que são muito mais escuros que a mesa. Falhou na sombra (T=120 inclui a sombra; proporção 0,2496 contra 0,1962 com T=80 no disco) e no monitor, que nenhum limiar global separa do objeto.
- **(b) Parâmetro mais sensível.** O limiar T. Passar de 80 para 120 aumentou o primeiro plano em 27% no disco (976 mil → 1,24 milhão de pixels) e em 28% nos óculos, quase tudo vindo da sombra e da borda das regiões escuras. O C do adaptativo também é sensível, mas aquele método não é adequado a objetos grandes e uniformes.
- **(c) Erros que podem atrapalhar a contagem no M2.2.**
  - Os óculos e o disco **sobrepostos** formam um único componente.
  - As faixas do monitor e da borda da mesa viram componentes grandes que não são objetos.
  - Reflexos dentro da lente geram fragmentos pequenos.
  - A sombra aumenta a área do disco.

### Configuração de trabalho escolhida para o M2.2

Escolhi **Otsu invertido com blur 5×5** (`otsu_inv`). Os motivos:
- O limiar é escolhido automaticamente por imagem (95 e 99), em vez de ajustado à mão.
- Fica entre o T=80 (que perde partes do objeto) e o T=120 (que inclui a sombra).
- Pega o disco e a armação inteiros, com pouca sombra.
- O blur reduz os pontos isolados.

Erros que continuam: o monitor no topo, a borda inferior em `oculos_disco.jpg` e a lente sobre o disco. As máscaras finais estão em `images/output/disco_mask.png` e `images/output/oculos_disco_mask.png`. Elas foram verificadas programaticamente: mesmo tamanho da entrada (2576×1932), 1 canal, uint8 e valores {0, 255}. A escrita recusa máscara não binária e JPEG.

## M2.2 — Componentes conexos e sequência morfológica

Ampliei o mesmo código com a operação `analyze`, que faz o seguinte:
1. Carrega a máscara binária produzida por `segment` (objeto 255, fundo 0; outras entradas são rejeitadas).
2. Salva uma cópia da máscara inicial (`_stage0_initial.png`).
3. Rotula os componentes com conectividade 4 e 8 na mesma máscara.
4. Aplica a sequência morfológica estágio a estágio.
5. Reconta na máscara final e grava CSVs, rótulos, painel e `_summary.json`.

O fundo (rótulo 0) nunca entra no CSV.

Comando usado para cada foto (ver `results/commands.txt`):

```bash
python -m pdi_lab --input images/output/oculos_disco_mask.png --output results/oculos_disco/oculos_disco --operation analyze --connectivity 8 --min-area 2000 --morph open:ellipse:5:1,close:ellipse:9:1 --original images/input/oculos_disco.jpg
```

### Sequência morfológica e parâmetros

1. **Abertura**, elipse 5×5, 1 iteração. Problema visado: os pontos isolados de ruído e de textura e os fragmentos finos dos reflexos, que geravam dezenas de componentes minúsculos (39 componentes brutos no disco e 51 nos óculos).
2. **Fechamento**, elipse 9×9, 1 iteração. Problema visado: pequenos buracos e rachaduras dentro do disco e da armação, causados por reflexos, que poderiam fragmentar um objeto.

A ordem é abrir primeiro, para que o fechamento não "cole" ruído aos objetos, e fechar depois, para recuperar o interior. Usei elementos diferentes porque o ruído é menor que 5 px, enquanto os buracos chegam a alguns pixels a mais. A elipse foi escolhida para não criar cantos retos nos contornos curvos.

Efeitos observados em `results/<imagem>/<imagem>_stages_panel.png`:
- A abertura removeu o ruído sem afetar o corpo dos objetos.
- O fechamento suavizou o contorno do disco e preencheu falhas na armação.
- Nos óculos, o fechamento **fundiu ainda mais** a armação com o disco na região de sobreposição, um efeito indesejado para a contagem.
- A área total mudou pouco: −384 px no disco e −3 094 px nos óculos.

**Critério de filtragem:** o componente é mantido se `area >= 2000` pixels (coluna `kept`). Esse valor é cerca de 0,04% da imagem, bem abaixo do menor objeto real, que tem centenas de milhares de pixels. O mesmo critério foi aplicado antes e depois da morfologia.

### Conectividade 4 × 8

- **Conectividade 4:** dois pixels só pertencem ao mesmo componente se compartilham uma aresta.
- **Conectividade 8:** basta compartilhar um vértice (diagonal).

Por isso a contagem com 4 é sempre maior ou igual à com 8. O teste com dois pixels apenas na diagonal dá 2 componentes em conectividade 4 e 1 em 8.

Nas fotos, a máscara inicial tem:
- **disco.jpg:** 48 componentes (4) contra 39 (8);
- **oculos_disco.jpg:** 58 contra 51.

A diferença está nos fragmentos de ruído ligados só pela diagonal. Após abertura e fechamento, as duas conectividades dão 4 componentes em ambas as fotos, porque os contornos ficaram suaves.

Escolhi **8** para a análise final porque a armação e as hastes têm trechos finos e diagonais que a conectividade 4 poderia partir em pedaços.

### Tabela antes/depois

Fonte: `results/summary_analyze.csv` e `results/<imagem>/<imagem>_before_after.csv`.

| Imagem | Estágio | Comp. 4 | Comp. 8 | Brutos (8) | Filtrados (≥ 2000) | Área primeiro plano |
|---|---|---|---|---|---|---|
| disco.jpg | inicial | 48 | 39 | 39 | 2 | 1 020 543 |
| disco.jpg | final | 4 | 4 | 4 | 2 | 1 020 159 |
| oculos_disco.jpg | inicial | 58 | 51 | 51 | 6 | 1 536 994 |
| oculos_disco.jpg | final | 4 | 4 | 4 | 4 | 1 533 900 |
| coins_scene (sintética) | inicial | 6 | 4 | 4 | 4 | 5 414 |
| coins_scene (sintética) | final | 4 | 4 | 4 | 4 | 5 635 |
| square_light (sintética) | inicial | 1 | 1 | 1 | 1 | 2 500 |
| square_light (sintética) | final | 1 | 1 | 1 | 1 | 2 496 |

Observações:

1. **disco.jpg.** A sequência reduziu os componentes brutos de 39 para 4, e as 37 manchas pequenas de ruído e sombra sumiram. Após o filtro sobram 2 componentes: o disco (486 531 px, caixa x=910, y=697, 778×776) e a faixa do monitor no topo (533 574 px, caixa tocando a borda superior). Os 2 componentes restantes (17 e 37 px) foram removidos pelo filtro.
2. **oculos_disco.jpg.** A sequência reduziu de 51 para 4 componentes brutos, e os 4 passam no filtro:
   - óculos e disco **fundidos em um único componente** (591 286 px, caixa 1958×706);
   - faixa do monitor no topo (770 581 px);
   - borda escura inferior (169 669 px);
   - um fragmento de reflexo dentro da lente esquerda (2 364 px, caixa x=521, y=982).

   Antes da morfologia, 6 passavam no filtro: a abertura eliminou dois pedaços da borda inferior (3 932 e 2 512 px), que se juntaram ou foram suavizados.
3. **Imagens sintéticas.** Na cena dos círculos, o fechamento preencheu os furos do ruído e a área subiu de 5 414 para 5 635 (a referência tem 5 688). No quadrado, a abertura com elipse 3×3 removeu os 4 pixels dos cantos (2 500 → 2 496), um exemplo pequeno de mudança de contorno.

**Verificação:** a soma das áreas dos componentes brutos é igual aos pixels de primeiro plano da máscara antes de qualquer filtro (`raw_area_total` = `foreground_pixels` em `_summary.json`: 1 020 543 e 1 536 994). Isso também é conferido em teste automático.

### Figuras

- `results/<imagem>/<imagem>_stages_panel.png`: original, cada estágio e rótulos inicial e final lado a lado.
- `results/<imagem>/<imagem>_labels.png` e `_initial_labels.png`: rótulos coloridos com caixas (verde = mantido, vermelho = removido pelo filtro).

### Avaliação das limitações

- **Caso em que a sequência ajuda** (`disco.jpg`): a abertura limpou as dezenas de manchas de ruído e sombra (39 → 4 componentes), e o disco virou um único componente compacto com contorno suave.
- **Caso em que não resolve ou prejudica** (`oculos_disco.jpg`): os óculos e o disco estão sobrepostos na foto, então nenhuma abertura pequena os separa, e o fechamento 9×9 reforça a união. Uma abertura grande o suficiente para cortar a ligação apagaria também as hastes finas da armação.
- **Contagem × interpretação humana:**

  | Foto | Objetos que eu vejo | Componentes após o filtro | Diferença |
  |---|---|---|---|
  | disco.jpg | 1 | 2 | +1: o monitor |
  | oculos_disco.jpg | 2 | 4 | 2 erros para mais (monitor e borda inferior) e 1 para menos (óculos e disco contados como um), mais o fragmento da lente |

  O número de componentes não equivale ao número real de objetos:
  - objetos que se tocam viram um componente;
  - reflexos criam componentes extras;
  - regiões escuras do cenário passam no limiar.

  Essa contagem manual é uma observação minha, não uma métrica.
- Uma melhoria possível seria descartar componentes que tocam a borda da imagem ou definir uma região de interesse. Isso eliminaria o monitor e a borda da mesa, mas não fez parte da sequência obrigatória; fica registrado como extensão.

## Testes

`python -m pytest` cobre, em 38 testes:
- **Máscara:** mesmo shape, 1 canal, uint8 e valores {0,255}.
- **Segmentação:**
  - limiar fixo nos pixels T−1, T e T+1;
  - proporção 0,25 com fixo e Otsu no quadrado;
  - inversão;
  - adaptativo sobre gradiente;
  - HSV, incluindo faixa que cruza o matiz 179→0.
- **Validação e erros:**
  - parâmetros inválidos;
  - métricas FP/FN/IoU e cores do mapa de erros;
  - erros da CLI com código ≠ 0: arquivo inexistente, imagem inválida, método e operação desconhecidos, limiar fora da faixa, saída JPEG, conectividade inválida, `--min-area` negativo e máscara não binária.
- **Análise (M2.2):**
  - diagonal com conectividade 4 × 8;
  - abertura removendo ruído isolado;
  - fechamento preenchendo um furo;
  - máscara vazia;
  - objeto tocando a borda;
  - filtro por área;
  - arquivos e colunas do `analyze`;
  - soma das áreas brutas igual aos pixels de primeiro plano.

Todos passam.

## Limitações

- Os parâmetros foram ajustados para estas fotos; outra iluminação ou outra mesa exige novo ajuste.
- Sem máscara de referência das fotos reais, a avaliação delas é qualitativa.
- Regiões escuras do cenário (monitor, borda da mesa) não se distinguem do objeto por intensidade nem por HSV.
- A rotulação não separa objetos encostados ou sobrepostos; watershed ficou fora do escopo, como extensão opcional.
- Lentes transparentes deixam ver o que está atrás e geram erros que nenhum limiar global corrige.

## Conclusão

Para objetos pretos sobre mesa clara, o Otsu invertido com blur 5×5 deu a máscara mais equilibrada: evitou a maior parte da sombra que o limiar T=120 incluía e as perdas do T=80. O adaptativo não serve para objetos grandes e uniformes, e o HSV só ajuda pelo canal V, já que o objeto não tem cor. A sequência abertura 5×5 seguida de fechamento 9×9 reduziu os componentes de 39 e 51 para 4 em cada foto, removendo ruído e mantendo a forma dos objetos. Mesmo assim, a contagem final só se aproxima da humana na foto favorável (disco), e só se descontar o monitor. Na foto difícil, a sobreposição dos óculos com o disco e as regiões escuras do cenário mostram que componente conexo não é sinônimo de objeto.
