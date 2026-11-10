# Sailor — Guardiã Estelar

Arte generativa em Python: uma guardiã mágica (coques, marias-chiquinhas, gola de marinheira, saia e varinha) que se forma a partir de **poeira de estrelas** e termina desenhada **só com traçados**, sobre um céu estrelado animado.

Tudo é feito com `numpy` (cálculo dos pontos) e `Pillow` (imagens), sem nenhuma biblioteca de desenho vetorial.

---

## 1. Preparando o ambiente

Você precisa do Python 3.9 ou mais novo.

```bash
pip install numpy pillow
```

A janela animada usa o `tkinter`, que já vem com o Python no Windows e no macOS. No Linux, pode ser preciso instalar o pacote `python3-tk`.

## 2. Rodando

```bash
python Sailor.py          # janela animada
python Sailor.py --png    # salva imagens de cada etapa + guardia_etapas.png
python Sailor.py --gif    # salva guardia.gif
```

Atalhos na janela:

| Tecla   | Ação                              |
|---------|-----------------------------------|
| `Espaço`| pausa / continua                  |
| `r`     | recomeça a animação               |
| `s`     | salva o quadro atual em PNG       |
| `Esc`   | fecha                             |

---

## 3. Como funciona, passo a passo

O desenho inteiro é uma **nuvem de pontos**. Cada parte do corpo vira milhares de pontinhos coloridos, e o estilo final decide como esses pontos aparecem na tela.

### Passo 1 — Um sistema de coordenadas próprio

A figura é desenhada em "unidades da figura": a altura total vale `1`, os pés ficam em `y = 0` e o centro do corpo em `x = 0`. Só na hora de desenhar convertemos para pixels:

```python
S  = H * 0.62      # escala: quantos pixels vale 1 unidade
ox = W / 2         # centro horizontal da tela
oy = H * 0.90      # linha dos pés
px, py = ox + x * S, oy - y * S   # o y é invertido (na tela ele cresce para baixo)
```

Assim, a figura funciona em qualquer tamanho de janela.

### Passo 2 — Preenchendo formas com pontos

Cada forma (rosto, blusa, gola, botas...) é montada com algumas funções de geometria:

- `fill_ellipse` sorteia pontos dentro de uma elipse (cabeça, coques, mangas);
- `fill_tris` sorteia pontos dentro de triângulos (blusa, gola, laço). Com `quad` e `fan` qualquer polígono vira triângulos;
- `along` espalha pontos ao longo de uma linha (listras da gola, tiara);
- `bez` calcula uma curva de Bézier (cabelo, braços e saia se dobram por ela).

```python
# a blusa é um trapézio dividido em dois triângulos
add(fill_tris(rng, quad((-0.075, 0.738), (0.075, 0.738),
                        (0.050, 0.568), (-0.050, 0.568))),
    BLOUSE_PAL, lambda x, y: 0.10 + 0.5 * rnd(x), -400)
```

O último número (`-400`) é o **z**, a profundidade: quanto maior, mais na frente a parte fica. É o que faz a gola ficar por cima da blusa e o rosto por cima do cabelo de trás.

### Passo 3 — Cores com paletas

Cada parte tem uma paleta (`HAIR_PAL`, `COLLAR_PAL`, `SKIRT_PAL`...). A função `palette_pick` escolhe uma cor da paleta para cada ponto a partir de um valor entre 0 e 1, com um pouco de ruído, o que dá a textura de pontilhismo. Para mudar as cores da roupa, basta editar as paletas no começo do arquivo.

### Passo 4 — Partes fixas e partes que se mexem

- `build_figure` cria as partes **fixas** (cabeça, gola, laço, botas, tiara).
- `build_dynamic` cria as partes que **se mexem** (marias-chiquinhas, saia, braços e varinha). Esses pontos guardam coordenadas "locais" (por exemplo, a posição ao longo do cabelo), e a cada quadro `draw_figure` recalcula onde eles ficam usando senos do tempo `t`. É isso que faz o cabelo balançar e a saia ondular.

### Passo 5 — A montagem com poeira de estrelas

Cada ponto nasce com uma posição de "poeira" espalhada pela tela e um atraso `d`. A função `assemble` faz o ponto voar em curva até o lugar dele conforme a animação avança:

```python
e = np.clip(build * 1.7 - p['d'] * 0.7, 0.0, 1.0)   # progresso deste ponto (0 → 1)
x = sx + vx * e - vy * k                            # caminho levemente curvo
```

O "relógio" da animação está em `state(t)`: primeiro aparece o céu, depois a figura se monta e, por fim, acende o brilho da varinha.

### Passo 6 — Só os traçados (estilo `"linhas"`)

No estilo `"linhas"`, os pontos que já chegaram **não são pintados**. Em vez disso:

1. `make_group` desenha cada parte (cada valor de z) numa máscara, suaviza com um *blur* e fica só com a **borda**. Partes finas, como listras e a haste da varinha, viram a própria linha.
2. `draw_lines` empilha as bordas de trás para a frente. A região de uma parte que fica na frente **apaga** as linhas que estão atrás dela, como num desenho de verdade.
3. Um "granulado" fixo (`grain`) deixa falhas no traço, com cara de lápis.

Os contornos das partes fixas são calculados uma vez só (`build_static_lines`). Os das partes que se mexem são recalculados a cada quadro.

### Passo 7 — O céu

`build_sky` e `draw_sky` cuidam da nebulosa, das estrelinhas que piscam, das estrelas de 4 e 5 pontas, da lua crescente, das estrelas cadentes, do banco de nuvens e do anel de luz sob os pés. Cada um é só mais uma nuvem de pontos com brilho variando no tempo.

### Passo 8 — Juntando tudo

`render(t)` monta um quadro:

```
céu  →  poeira em voo  →  traçados da figura  →  brilhos da varinha
```

Os pontos são agrupados em `Batch`, ordenados por z e "carimbados" no canvas pela função `splat`, que desenha cada ponto como um disquinho de 1 a 3 pixels.

---

## 4. Personalizando

No começo de `Sailor.py`:

| Variável     | O que faz                                                                  |
|--------------|----------------------------------------------------------------------------|
| `ESTILO`     | `"linhas"` (só traçados), `"esboco"` (esboço colorido) ou `"pontos"` (pontilhismo) |
| `SEED`       | muda o sorteio dos pontos e das estrelas                                   |
| `TIME_SCALE` | maior = animação mais lenta                                                |
| `LOOP`       | `True` para a animação recomeçar sozinha                                   |
| `DENS`       | densidade de pontos (mais pontos = mais detalhe e mais lento)              |
| `EDGE_INNER` | sensibilidade das linhas internas no estilo `"esboco"`                     |
| `*_PAL`      | paletas de cores de cada parte                                             |

### Ideias para praticar

- Trocar as cores da gola e da saia para outra guardiã (ex.: vermelho, verde, laranja).
- Mudar o objeto na ponta da varinha: troque a lua por uma estrela usando `star_tris`.
- Acrescentar um acessório novo em `build_figure` (brincos, uma pulseira) com `fill_ellipse` e um z adequado.
- Fazer o cabelo balançar mais, aumentando os valores que multiplicam `sw1` e `sw2` em `draw_figure`.
