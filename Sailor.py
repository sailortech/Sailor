"""Guardiã Estelar: silhueta de guardiã mágica (coques, tranças e uniforme de marinheira), sem rosto, em pontilhismo.

Arquivo independente: não precisa do jardim.py.

    python Sailor.py          # janela animada
    python Sailor.py --png    # imagens das etapas
    python Sailor.py --gif    # GIF animado
"""
import math
import sys
import time

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------- ferramentas pontilhistas (do jardim.py)
_OFF = sorted(((dx, dy) for dy in range(-3, 4) for dx in range(-3, 4) if dx * dx + dy * dy <= 9),
              key=lambda p: (p[0] ** 2 + p[1] ** 2, p))
OFF = np.array(_OFF, np.int32)
D2 = (OFF ** 2).sum(1)


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)



def palette_pick(rng, pal, g, spread=0.55, jitter=0.07):
    pal = np.asarray(pal, np.float32)
    g = np.asarray(g, np.float32)
    x = np.clip(g, 0, 1) * (len(pal) - 1) + rng.normal(0, spread, g.shape)
    idx = np.clip(np.rint(x), 0, len(pal) - 1).astype(int)
    return (pal[idx] * (1 + rng.normal(0, jitter, (len(idx), 1)))).astype(np.float32)



def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}



def splat(canvas, x, y, col, rad, nmax=13):
    H, W = canvas.shape[:2]
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    off = OFF[:nmax]
    xs = xi[:, None] + off[None, :, 0]
    ys = yi[:, None] + off[None, :, 1]
    m = (D2[None, :nmax] <= (rad * rad)[:, None] + 0.01) & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    rows = np.nonzero(m)[0]
    canvas[ys[m], xs[m]] = col[rows]


class Batch:
    def __init__(self):
        self.p = []

    def add(self, x, y, col, rad, z):
        if len(x):
            self.p.append((x, y, col, rad, z))

    def draw(self, canvas):
        if not self.p:
            return
        x, y, col, rad, z = [np.concatenate([q[i] for q in self.p]) for i in range(5)]
        o = np.argsort(z, kind="stable")
        c = np.clip(col[o], 0, 255).astype(np.uint8)
        splat(canvas, x[o], y[o], c, rad[o])


def bez(P, s):
    P0, P1, P2, P3 = P
    q = 1 - s
    x = q ** 3 * P0[0] + 3 * q * q * s * P1[0] + 3 * q * s * s * P2[0] + s ** 3 * P3[0]
    y = q ** 3 * P0[1] + 3 * q * q * s * P1[1] + 3 * q * s * s * P2[1] + s ** 3 * P3[1]
    dx = 3 * q * q * (P1[0] - P0[0]) + 6 * q * s * (P2[0] - P1[0]) + 3 * s * s * (P3[0] - P2[0])
    dy = 3 * q * q * (P1[1] - P0[1]) + 6 * q * s * (P2[1] - P1[1]) + 3 * s * s * (P3[1] - P2[1])
    return x, y, dx, dy


SEED = 7
TIME_SCALE = 1.0
TICK_MS = 10
WIN_FRAC = 0.92
LOOP = False
EDGE_INNER = 7.0      # sensibilidade dos traços internos (menor = mais linhas)
ESTILO = "linhas"    # "linhas" = só contornos | "esboco" = esboço preenchido | "pontos" = pontilhismo colorido
T_END = 24.0          # depois disso ela continua "respirando" (só o cabelo e as estrelas se mexem)
DENS = 70000.0        # pontos por unidade² (a figura tem 1 unidade de altura)

# ---------------------------------------------------------------- paletas
SKIN_PAL = [(196, 140, 100), (168, 114, 80), (134, 88, 62)]
HAIR_PAL = [(108, 96, 146), (68, 57, 104), (42, 33, 72), (20, 16, 42)]
HAIR_HI = [(190, 180, 225), (225, 215, 250)]
EYE_PAL = [(14, 8, 8), (40, 22, 16), (78, 48, 30), (122, 82, 52)]
BROW_PAL = [(58, 40, 40), (38, 26, 30)]
LIP_PAL = [(226, 126, 130), (198, 94, 106), (170, 72, 88)]
BLOUSE_PAL = [(255, 252, 255), (238, 230, 255), (214, 202, 250)]
COLLAR_PAL = [(22, 58, 168), (34, 82, 204), (58, 112, 232)]
STRIPE_PAL = [(255, 246, 235), (255, 232, 190)]
ROSE_PAL = [(255, 170, 215), (255, 112, 178), (238, 66, 145), (196, 36, 112)]
BOW_PAL = [(255, 232, 240), (255, 202, 222), (255, 170, 204), (240, 132, 176)]
SKIRT_PAL = [(18, 48, 150), (28, 72, 190), (50, 104, 226), (110, 158, 246)]
SKIRT_UNDER = [(222, 232, 255), (240, 246, 255), (252, 253, 255)]
TRIM_PAL = [(255, 255, 255), (236, 242, 255)]
BOOT_PAL = [(255, 224, 235), (255, 194, 216), (246, 160, 194), (218, 124, 164)]
CUFF_PAL = [(255, 255, 255), (238, 242, 255)]
GOLD_PAL = [(255, 232, 150), (255, 210, 100), (240, 176, 70)]
GLOVE_PAL = [(255, 253, 255), (238, 230, 255), (216, 206, 250)]
BLUSH_PAL = [(255, 176, 190), (255, 150, 172)]
STAR_COLS = np.array([(255, 255, 255), (255, 246, 205), (255, 222, 238), (205, 226, 255)], np.float32)
NEB_PAL = [(40, 20, 92), (70, 36, 142), (112, 52, 172), (40, 62, 152), (150, 72, 192)]
CLOUD_PAL = [(176, 118, 226), (126, 84, 196), (88, 60, 170), (60, 42, 130), (38, 28, 96)]
MOON_PAL = [(255, 248, 210), (255, 234, 165), (255, 214, 122)]
RING_PAL = [(212, 164, 255), (255, 192, 250), (255, 226, 164), (172, 202, 255)]
SPARK_COLS = np.array([(255, 255, 255), (255, 222, 245), (255, 236, 170), (214, 214, 255)], np.float32)
SPARK_W = np.array([255, 240, 255], np.float32)


# ---------------------------------------------------------------- geometria
def fill_tris(rng, tris, dens=DENS):
    """Pontos uniformes dentro de uma lista de triângulos."""
    t = np.asarray(tris, float)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    ar = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    n = max(int(ar.sum() * dens), 6)
    idx = rng.choice(len(t), n, p=ar / ar.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    x = w0 * a[idx, 0] + w1 * b[idx, 0] + w2 * c[idx, 0]
    y = w0 * a[idx, 1] + w1 * b[idx, 1] + w2 * c[idx, 1]
    return x, y


def quad(a, b, c, d):
    return [(a, b, c), (a, c, d)]


def fan(pts):
    return [(pts[0], pts[i], pts[i + 1]) for i in range(1, len(pts) - 1)]


def star_tris(cx, cy, R, r=0.42, rot=0.0, k=5):
    v = []
    for i in range(2 * k):
        rr = R if i % 2 == 0 else R * r
        a = rot + math.pi / 2 + i * math.pi / k
        v.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return [((cx, cy), v[i], v[(i + 1) % (2 * k)]) for i in range(2 * k)]


def fill_ellipse(rng, cx, cy, a, b, dens=DENS):
    n = max(int(math.pi * a * b * dens), 6)
    r, th = np.sqrt(rng.random(n)), rng.random(n) * 2 * np.pi
    return cx + a * r * np.cos(th), cy + b * r * np.sin(th)


def along(rng, pts, n, w=0.0):
    """Pontos ao longo de uma polilinha (com espessura w)."""
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1]) + 1e-9
    cum = np.concatenate([[0], np.cumsum(ln)])
    s = rng.random(n) * cum[-1]
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    f = (s - cum[i]) / ln[i]
    x = pts[i, 0] + seg[i, 0] * f + rng.normal(0, w, n)
    y = pts[i, 1] + seg[i, 1] * f + rng.normal(0, w, n)
    return x, y


def weighted(rng, n, f):
    """Sorteia a em [0,1] com densidade proporcional a f(a)."""
    out, got = [], 0
    while got < n:
        a = rng.random(n * 2)
        k = a[rng.random(n * 2) < f(a)]
        out.append(k)
        got += len(k)
    return np.concatenate(out)[:n]


def blink(t):
    if t < 13:
        return 0.0
    ph = (t + 0.4) % 4.3
    return float(np.clip(1 - abs(ph - 0.1) / 0.1, 0, 1))


def state(t):
    sky = float(smoothstep(t, 0.0, 3.0))
    build = float(smoothstep(t, 2.0, 11.0))
    face = float(smoothstep(t, 10.0, 12.5))
    glow = float(smoothstep(t, 11.0, 15.0))
    return sky, build, face, glow


# ---------------------------------------------------------------- a guardiã
class Guardian:
    def __init__(self, W, H, seed=SEED):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        self.build_figure()
        self.build_dynamic()
        self.build_sky()
        # granulado fixo do "papel" (faz o esboço ter falhas, como grafite)
        gr = np.random.default_rng(seed + 1)
        blk = np.kron(gr.random((H // 3 + 1, W // 3 + 1)), np.ones((3, 3)))[:H, :W]
        self.grain = (0.55 * blk + 0.45 * gr.random((H, W))).astype(np.float32)
        self.grain2 = gr.random((H, W)).astype(np.float32)
        self.grain3 = gr.random((H, W)).astype(np.float32)
        self._sg = None          # contornos das partes fixas (calculados uma vez)
        self._dynraw = []
        self._want_lines = False
        self._bob = 0.0

    # ---- util
    def pk(self, pal, g, **k):
        return palette_pick(self.rng, pal, np.clip(g, 0, 1), **k)

    def mk(self, x, y, col, z, rad=None, extra=None):
        """Empacota uma nuvem de pontos (com atraso e origem de 'poeira' para a montagem)."""
        rng = self.rng
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        n = len(x)
        if rad is None:
            rad = rng.choice([1.0, 1.5, 2.0], n, p=[0.3, 0.45, 0.25])
        elif np.isscalar(rad):
            rad = np.full(n, rad)
        d = np.clip(rng.random(n) * 0.55 + 0.45 * (1 - np.clip(y, 0, 1.1) / 1.1), 0, 1)
        a = rng.random(n) * 2 * np.pi
        r = 0.4 + rng.random(n) ** 0.8 * 1.1
        out = dict(x=x, y=y, col=np.asarray(col, np.float32),
                   z=np.full(n, z, np.float32) if np.isscalar(z) else np.asarray(z, np.float32),
                   rad=np.asarray(rad, np.float32), d=d.astype(np.float32),
                   sx=(r * np.cos(a)).astype(np.float32), sy=(0.55 + r * np.sin(a)).astype(np.float32))
        if extra:
            out.update(extra)
        return out

    # ---- partes fixas do corpo (em "unidades da figura": altura total = 1, pés em y=0)
    def build_figure(self):
        rng = self.rng
        parts = []
        rnd = lambda x: rng.random(len(x))
        ones = lambda n: np.ones(n, np.float32)

        def add(xy, pal, g, z, rad=None, **k):
            x, y = xy
            gg = g(x, y) if callable(g) else g
            parts.append(self.mk(x, y, self.pk(pal, gg, **k), z, rad))

        def mirror(tris):
            return [tuple((-p[0], p[1]) for p in t) for t in tris]

        # cabelo de trás + topo da cabeça
        add(fill_ellipse(rng, 0, 0.855, 0.094, 0.100, dens=120000), HAIR_PAL,
            lambda x, y: 0.15 + 0.45 * rnd(x) + 0.25 * (0.90 - y) / 0.10, -1100)
        x, y = fill_ellipse(rng, 0, 0.905, 0.082, 0.048, dens=170000)
        m = y >= 0.895
        add((x[m], y[m]), HAIR_PAL, lambda x, y: 0.05 + 0.40 * rnd(x), -40)
        x, y = along(rng, [(-0.058, 0.925), (-0.030, 0.945), (0, 0.951), (0.030, 0.945), (0.058, 0.925)], 240, 0.0035)
        add((x, y), HAIR_HI, lambda x, y: rnd(x), -35, rad=1.0)

        # franja e mechas laterais
        locks = [(-0.058, 0.024, -0.068, 0.846), (-0.034, 0.024, -0.041, 0.885), (-0.009, 0.024, -0.014, 0.893),
                 (0.017, 0.024, 0.014, 0.893), (0.041, 0.024, 0.041, 0.885), (0.063, 0.024, 0.068, 0.846)]
        tris = [((xc - hw, 0.94), (xc + hw, 0.94), (ax, ay)) for xc, hw, ax, ay in locks]
        add(fill_tris(rng, tris, dens=170000), HAIR_PAL, lambda x, y: 0.20 + 0.40 * rnd(x) + 0.35 * (0.94 - y) / 0.09, -30)
        side = fan([(-0.062, 0.93), (-0.098, 0.915), (-0.094, 0.76), (-0.076, 0.80)])
        add(fill_tris(rng, side + mirror(side), dens=170000), HAIR_PAL, lambda x, y: 0.25 + 0.5 * rnd(x), -30)

        # coques: dois "odangos" no alto da cabeça, com espiral de fios e brilho
        for m_ in (-1, 1):
            bx, by, bR = m_ * 0.074, 0.996, 0.048
            n = int(math.pi * bR * bR * 210000)
            r, th = bR * np.sqrt(rng.random(n)), rng.random(n) * 2 * np.pi
            x, y = bx + r * np.cos(th), by + r * np.sin(th)
            spiral = 0.5 + 0.5 * np.sin(5 * th + 75 * r)
            g = 0.08 + 0.30 * spiral + 0.35 * (r / bR) ** 2 + 0.18 * (by - y) / bR + rng.normal(0, 0.06, n)
            col = self.pk(HAIR_PAL, g)
            hl = (np.hypot((x - bx) / bR, (y - by) / bR) > 0.5) & ((y - by) / bR > 0.30) & (rng.random(n) < 0.6)
            col[hl] = self.pk(HAIR_HI, rng.random(int(hl.sum())))
            parts.append(self.mk(x, y, col, -45))

        # rosto
        x1, y1 = fill_ellipse(rng, 0, 0.852, 0.074, 0.082, dens=85000)
        x2, y2 = fill_tris(rng, [((-0.05, 0.80), (0.05, 0.80), (0, 0.766))], dens=85000)
        add((np.concatenate([x1, x2]), np.concatenate([y1, y2])), SKIN_PAL,
            lambda x, y: 0.10 + 0.30 * rnd(x) + 0.30 * np.clip((0.82 - y) / 0.06, 0, 1) + 0.45 * (np.abs(x) / 0.072) ** 2, -100)

        # pescoço, blusa
        add(fill_tris(rng, quad((-0.017, 0.785), (0.017, 0.785), (0.020, 0.738), (-0.020, 0.738)), dens=85000), SKIN_PAL,
            lambda x, y: 0.45 + 0.4 * rnd(x), -350)
        add(fill_tris(rng, quad((-0.075, 0.738), (0.075, 0.738), (0.050, 0.568), (-0.050, 0.568))), BLOUSE_PAL,
            lambda x, y: 0.10 + 0.5 * rnd(x) + 0.3 * (0.74 - y) / 0.17, -400)

        # gola de marinheira (azul-marinho com listras)
        A, B, C, D = (-0.026, 0.764), (-0.142, 0.742), (-0.128, 0.690), (0.0, 0.652)
        coll = quad(A, B, C, D)
        coll = coll + mirror(coll)
        add(fill_tris(rng, coll), COLLAR_PAL, lambda x, y: 0.1 + 0.6 * rnd(x), -250)
        for dy in (0.010, 0.020):
            for sg in (-1, 1):
                pl = [(sg * -0.136, 0.738 + dy), (sg * -0.124, 0.694 + dy), (sg * -0.003, 0.656 + dy)]
                x, y = along(rng, pl, 260, 0.0013)
                add((x, y), STRIPE_PAL, lambda x, y: rnd(x), -240, rad=ones(len(x)))

        # laço do peito
        wing = fan([(0, 0.652), (-0.030, 0.682), (-0.070, 0.690), (-0.078, 0.652), (-0.070, 0.614), (-0.030, 0.622)])
        add(fill_tris(rng, wing + mirror(wing)), BOW_PAL,
            lambda x, y: 0.1 + 0.7 * np.clip(np.abs(x) / 0.075, 0, 1) * rnd(x) + 0.2 * rnd(x), -200)
        tails = [((-0.006, 0.640), (-0.040, 0.575), (-0.014, 0.588))]
        add(fill_tris(rng, tails + mirror(tails)), BOW_PAL, lambda x, y: 0.3 + 0.5 * rnd(x), -205)
        add(fill_ellipse(rng, 0, 0.652, 0.013, 0.012, dens=200000), BOW_PAL, lambda x, y: 0.7 + 0.3 * rnd(x), -190)
        add(fill_ellipse(rng, -0.004, 0.656, 0.004, 0.004, dens=400000), [(255, 255, 255)], lambda x, y: rnd(x), -180,
            rad=1.0)

        # cinto e mangas
        add(fill_tris(rng, quad((-0.054, 0.572), (0.054, 0.572), (0.057, 0.552), (-0.057, 0.552))), COLLAR_PAL,
            lambda x, y: 0.2 + 0.6 * rnd(x), -390)
        for m_ in (-1, 1):
            add(fill_ellipse(rng, m_ * 0.088, 0.722, 0.030, 0.026), BLOUSE_PAL, lambda x, y: 0.2 + 0.6 * rnd(x), -280)

        # coxas e botas
        for m_ in (-1, 1):
            cx = m_ * 0.045
            add(fill_tris(rng, quad((cx - 0.027, 0.395), (cx + 0.027, 0.395), (cx + 0.024, 0.300), (cx - 0.024, 0.300)), dens=62000),
                SKIN_PAL, lambda x, y: 0.35 + 0.4 * rnd(x), -600)
            yb = weighted(rng, 2600, lambda a: np.ones_like(a)) * 0.265 + 0.035
            hw = 0.021 + 0.007 * (yb - 0.035) / 0.265
            xb = cx + rng.uniform(-1, 1, len(yb)) * hw
            col = self.pk(BOOT_PAL, 0.15 + 0.5 * rng.random(len(yb)) + 0.25 * np.abs(xb - cx) / hw, jitter=0.05)
            cuff = yb > 0.275
            col[cuff] = self.pk(CUFF_PAL, rng.random(int(cuff.sum())))
            parts.append(self.mk(xb, yb, col, -590))
            add(fill_ellipse(rng, cx + m_ * 0.012, 0.024, 0.034, 0.017), BOOT_PAL,
                lambda x, y: 0.2 + 0.6 * rnd(x), -590)

        # tiara dourada com gema
        x, y = along(rng, [(-0.064, 0.915), (-0.040, 0.925), (-0.018, 0.934), (0, 0.948), (0.018, 0.934),
                           (0.040, 0.925), (0.064, 0.915)], 420, 0.0020)
        add((x, y), GOLD_PAL, lambda x, y: rnd(x), -20, rad=ones(len(x)))
        add(fill_tris(rng, [((-0.013, 0.944), (0.013, 0.944), (0, 0.975))], dens=160000), GOLD_PAL,
            lambda x, y: rnd(x), -19, rad=1.0)
        add(fill_ellipse(rng, 0, 0.947, 0.0105, 0.0105, dens=300000), BOW_PAL, lambda x, y: 0.5 * rnd(x), -15)
        add(fill_ellipse(rng, -0.003, 0.951, 0.003, 0.003, dens=500000), [(255, 255, 255)], lambda x, y: rnd(x), -10,
            rad=1.0)

        self.static = cat(parts)

        self.eyes = None  # sem rosto: o desenho é só a silhueta

    # ---- partes que se mexem (cabelo, saia, braços, varinha)
    def build_dynamic(self):
        rng = self.rng
        # marias-chiquinhas
        self.tails = []
        for m_, ph in ((-1, 0.0), (1, 1.7)):
            n = 4800
            s = rng.random(n) ** 0.9
            lat = rng.uniform(-1, 1, n)
            col = self.pk(HAIR_PAL, 0.35 + 0.40 * s + 0.2 * np.sin(lat * 7 + s * 5) + rng.normal(0, 0.1, n))
            hl = (rng.random(n) < 0.06) | ((np.abs(lat) > 0.82) & (rng.random(n) < 0.5))
            col[hl] = self.pk(HAIR_HI, rng.random(int(hl.sum())))
            p = self.mk(np.zeros(n), 0.9 - 0.7 * s, col, -950)
            p.update(s=s, lat=lat, m=m_, ph=ph)
            self.tails.append(p)

        # saia (duas camadas)
        self.skirts = []
        for L, hw1, pal, z, ph, n in ((0.215, 0.188, SKIRT_UNDER, -520, 0.7, 2600),
                                      (0.185, 0.166, SKIRT_PAL, -500, 0.0, 2800)):
            a = weighted(rng, n, lambda u: (0.058 + (hw1 - 0.058) * u ** 0.9) / hw1)
            b = rng.uniform(-1, 1, n)
            pl = 0.5 + 0.5 * np.cos(b * 5 * math.pi)
            g = 0.08 + 0.50 * a + 0.32 * pl + rng.normal(0, 0.05, n)
            col = self.pk(pal, g)
            if pal is SKIRT_PAL:
                trim = a > 0.94
                col[trim] = self.pk(TRIM_PAL, rng.random(int(trim.sum())))
            p = self.mk(np.zeros(n), 0.565 - L * a, col, z)
            p.update(a=a, b=b, L=L, hw1=hw1, ph=ph)
            self.skirts.append(p)

        # braços com luvas
        self.arms = []
        for m_ in (-1, 1):
            n = 1300
            s, lat = rng.random(n), rng.uniform(-1, 1, n)
            col = self.pk(SKIN_PAL, 0.15 + 0.4 * rng.random(n))
            gl = s > 0.44
            col[gl] = self.pk(GLOVE_PAL, 0.1 + 0.5 * rng.random(int(gl.sum())) + 0.2 * np.abs(lat[gl]))
            cu = (s > 0.40) & (s < 0.46)
            col[cu] = self.pk(BOW_PAL, 0.3 + 0.5 * rng.random(int(cu.sum())))
            keep = gl | cu | (rng.random(n) > 0.35)   # a pele do braço fica mais rala
            s, lat, col = s[keep], lat[keep], col[keep]
            n = len(s)
            p = self.mk(np.zeros(n), 0.72 - 0.2 * s, col, -300)
            p.update(s=s, lat=lat, m=m_)
            self.arms.append(p)

        # varinha com lua crescente e estrela
        loc_u, loc_v, cols, zs = [], [], [], []
        n = 520
        u, v = rng.uniform(-0.07, 0.225, n), rng.uniform(-0.006, 0.006, n)
        loc_u.append(u); loc_v.append(v)
        cols.append(self.pk(GOLD_PAL, 0.2 + 0.6 * rng.random(n))); zs.append(np.full(n, 20.0))
        n = 90
        u, v = 0.10 + rng.uniform(-0.005, 0.005, n), rng.uniform(-0.011, 0.011, n)
        loc_u.append(u); loc_v.append(v)
        cols.append(self.pk(BOW_PAL, 0.4 * rng.random(n))); zs.append(np.full(n, 21.0))
        R, u0 = 0.060, 0.295
        n0 = 4200
        r, th = R * np.sqrt(rng.random(n0)), rng.random(n0) * 2 * np.pi
        lx, ly = r * np.cos(th), r * np.sin(th)
        keep = (lx - 0.42 * R) ** 2 + (ly - 0.12 * R) ** 2 >= (0.82 * R) ** 2
        lx, ly = lx[keep], ly[keep]
        loc_u.append(u0 + ly); loc_v.append(lx)
        cols.append(self.pk(MOON_PAL, 0.55 * rng.random(len(lx)) + 0.35 * (R - lx) / (2 * R))); zs.append(np.full(len(lx), 25.0))
        gx, gy = fill_tris(rng, star_tris(0.42 * R, 0.12 * R, 0.022), dens=500000)
        loc_u.append(u0 + gy); loc_v.append(gx)
        cols.append(self.pk(BOW_PAL, 0.45 * rng.random(len(gx)))); zs.append(np.full(len(gx), 26.0))
        U, V = np.concatenate(loc_u), np.concatenate(loc_v)
        self.wand = self.mk(np.zeros(len(U)), 0.9 + 0.2 * (U + 0.1), np.concatenate(cols), np.concatenate(zs),
                            np.full(len(U), 1.0))
        self.wand.update(u=U, v=V)
        self.wand_u0 = u0

        # brilho em volta da varinha e brilhos soltos
        m = 300
        self.wglow = dict(a=rng.random(m) * 2 * np.pi, r=np.minimum(np.abs(rng.normal(0, 0.45, m)), 1.4),
                          fl=rng.random(m))
        m = 300
        self.wsp = dict(a=rng.random(m) * 2 * np.pi, rho=rng.random(m), ph=rng.random(m),
                        sp=rng.uniform(0.6, 1.4, m), rad=rng.choice([1.0, 2.0], m, p=[0.7, 0.3]),
                        c=rng.integers(0, len(SPARK_COLS), m))
        m = 260
        self.bsp = dict(x0=rng.uniform(-0.55, 0.55, m), ph=rng.random(m), sp=rng.uniform(0.6, 1.4, m),
                        rad=rng.choice([1.0, 2.0], m, p=[0.7, 0.3]), c=rng.integers(0, len(SPARK_COLS), m))

    # ---- céu estrelado, lua, círculo mágico, nuvens
    def build_sky(self):
        rng = self.rng
        # nebulosa
        n = 4200
        fx, fy = rng.random(n), rng.random(n) * 0.95
        clump = 0.5 + 0.5 * np.sin(fx * 6.5 + 1.0) * np.sin(fy * 8.0 - fx * 2.5)
        br = np.clip(0.10 + 0.55 * clump ** 2 - 0.25 * fy, 0.05, 0.8)
        self.neb = dict(fx=fx, fy=fy, col=self.pk(NEB_PAL, rng.random(n) * 0.9 + 0.1 * clump) * br[:, None])

        # estrelinhas pequenas
        n = 1100
        self.tiny = dict(fx=rng.random(n), fy=rng.random(n) * 0.92, ph=rng.random(n) * 6.28,
                         sp=rng.uniform(0.8, 2.6, n), ap=rng.random(n),
                         col=STAR_COLS[rng.integers(0, 4, n)], rad=rng.choice([1.0, 1.5], n, p=[0.75, 0.25]))

        # estrelas de 4 pontas (brilhinhos)
        ns, per = 56, 44
        pool = rng.uniform(-1, 1, (ns * per * 8, 2))
        pool = pool[np.sqrt(np.abs(pool[:, 0])) + np.sqrt(np.abs(pool[:, 1])) <= 1][:ns * per]
        self.spk = dict(si=np.repeat(np.arange(ns), per), lx=pool[:, 0], ly=pool[:, 1],
                        fx=rng.uniform(0.03, 0.97, ns), fy=rng.uniform(0.03, 0.80, ns),
                        R=rng.uniform(0.006, 0.024, ns), ph=rng.random(ns) * 6.28, sp=rng.uniform(0.7, 2.0, ns),
                        ap=rng.random(ns), col=STAR_COLS[rng.integers(0, 4, ns)])

        # estrelas de 5 pontas girando
        ns, per = 16, 52
        lx, ly, si = [], [], []
        for i in range(ns):
            x, y = fill_tris(rng, star_tris(0, 0, 1.0), dens=1)
            k = rng.integers(0, len(x), per)
            lx.append(x[k]); ly.append(y[k]); si.append(np.full(per, i))
        lx, ly, si = np.concatenate(lx), np.concatenate(ly), np.concatenate(si)
        cols = np.array([(255, 236, 160), (255, 205, 235), (255, 255, 255), (200, 220, 255)], np.float32)
        self.st5 = dict(si=si, lx=lx, ly=ly, fx=rng.uniform(0.04, 0.96, ns), fy=rng.uniform(0.04, 0.78, ns),
                        R=rng.uniform(0.010, 0.026, ns), rot=rng.random(ns) * 6.28, om=rng.uniform(-0.5, 0.5, ns),
                        ph=rng.random(ns) * 6.28, ap=rng.random(ns), col=cols[rng.integers(0, 4, ns)])

        # lua crescente
        Rm = 0.085
        n0 = 6500
        r, th = Rm * np.sqrt(rng.random(n0)), rng.random(n0) * 2 * np.pi
        lx, ly = r * np.cos(th), r * np.sin(th)
        keep = (lx - 0.42 * Rm) ** 2 + (ly - 0.12 * Rm) ** 2 >= (0.82 * Rm) ** 2
        lx, ly = lx[keep], ly[keep]
        self.moon = dict(lx=lx, ly=ly, cx=0.17, cy=0.20,
                         col=self.pk(MOON_PAL, 0.55 * rng.random(len(lx)) + 0.4 * (lx + Rm) / (2 * Rm)))
        m = 1700
        a, rr = rng.random(m) * 2 * np.pi, Rm * (1.0 + np.minimum(np.abs(rng.normal(0, 0.45, m)), 1.5))
        self.mhalo = dict(lx=rr * np.cos(a), ly=rr * np.sin(a),
                          br=np.clip(1 - (rr / Rm - 1) / 1.6, 0, 1) ** 1.6 * 0.5)

        # estrelas cadentes
        self.shoot = [dict(off=o, per=8.0, x0=x0, y0=y0, k=rng.random(70)) for o, x0, y0 in
                      ((2.5, 0.95, 0.10), (5.5, 0.80, 0.28), (7.2, 1.0, 0.45))]

        # aura (halo suave)
        m = 1800
        r = np.minimum(np.abs(rng.normal(0, 0.45, m)), 0.95)
        self.halo = dict(a=rng.random(m) * 2 * np.pi, r=r, b=(1 - r / 0.95) ** 1.5 * 0.85, fl=rng.random(m))

        # anel de luz sob os pés
        m = 1900
        a = rng.random(m) * 2 * np.pi
        rr = np.where(rng.random(m) < 0.45, 1.0 + rng.normal(0, 0.012, m), np.sqrt(rng.random(m)))
        self.floor = dict(a=a, r=rr, b=np.where(rr > 0.9, 1.6, 0.6 * (1 - rr) + 0.15))

        # banco de nuvens no chão
        n = 6500
        fx = rng.random(n)
        top = 0.935 + 0.012 * np.sin(fx * 9) + 0.008 * np.sin(fx * 23 + 1)
        dep = rng.random(n) ** 1.3
        fy = top + dep * (0.995 - 0.935)
        col = self.pk(CLOUD_PAL, dep * 0.9 + rng.normal(0, 0.1, n)) * (1 - 0.55 * dep)[:, None]
        n2 = 420
        fx2 = rng.random(n2)
        top2 = 0.935 + 0.012 * np.sin(fx2 * 9) + 0.008 * np.sin(fx2 * 23 + 1)
        self.cloud = dict(fx=np.concatenate([fx, fx2]), fy=np.concatenate([fy, top2 + rng.normal(0, 0.003, n2)]),
                          col=np.concatenate([col, self.pk([(255, 190, 240), (230, 170, 255)], rng.random(n2))]),
                          rad=rng.choice([1.0, 1.5, 2.0], n + n2, p=[0.4, 0.4, 0.2]))

    # ------------------------------------------------------------ desenho
    def assemble(self, p, tx, ty, build, only_flying=False):
        """Faz os pontos virem "poeira de estrelas" até a posição final.

        only_flying=True: os pontos que já chegaram somem (no estilo "linhas" a figura
        é só o traçado, então a poeira não deve "pintar" o interior)."""
        if build >= 0.9999:
            if only_flying:
                z = np.zeros(0, np.float32)
                return z, z, np.zeros((0, 3), np.float32), z, z
            return tx, ty, p['col'], p['rad'], p['z']
        e = np.clip(build * 1.7 - p['d'] * 0.7, 0.0, 1.0)
        e = e * e * (3 - 2 * e)
        m = (e > 0.0) & (e < 0.985) if only_flying else e > 0.0
        em = e[m]
        sx, sy = p['sx'][m], p['sy'][m]
        vx, vy = tx[m] - sx, ty[m] - sy
        k = np.sin(np.pi * em) * 0.35
        x = sx + vx * em - vy * k
        y = sy + vy * em + vx * k
        vis = np.clip(em * 8, 0, 1)[:, None]
        col = (p['col'][m] + (SPARK_W - p['col'][m]) * ((1 - em) * 0.75)[:, None]) * vis
        return x, y, col, p['rad'][m], p['z'][m] + (1 - em) * 6000

    def render(self, t):
        H, W = self.H, self.W
        sky, build, face, glow = state(t)
        canvas = np.zeros((H, W, 3), np.uint8)
        Bsky, Bfig, Bfx = Batch(), Batch(), Batch()
        self._dynraw = []
        line_frac = float(smoothstep(build, 0.25, 1.0))
        self._want_lines = ESTILO == "linhas" and line_frac > 0.0
        self.draw_sky(Bsky, t, sky, glow)
        self.draw_figure(Bfig, Bfx, t, build, face, glow)
        Bsky.draw(canvas)
        if ESTILO == "pontos":
            Bfig.draw(canvas)
        elif ESTILO == "esboco":
            fig = np.zeros((H, W, 3), np.uint8)
            Bfig.draw(fig)
            self.sketch(canvas, fig)
        else:
            Bfig.draw(canvas)                           # só a poeira ainda em voo (sem preenchimento)
            if line_frac > 0.0:
                self.draw_lines(canvas, line_frac)      # os contornos vão aparecendo onde a poeira chega
        Bfx.draw(canvas)
        return canvas

    # ---------------- estilo "linhas": a figura é formada só pelas bordas das suas partes
    def wash(self, canvas, fig, frac):
        mask = fig.any(axis=2)
        if not mask.any():
            return
        paper = np.array([236, 230, 246], np.float32)
        soft = fig.astype(np.float32) * 0.55 + paper * 0.45
        keep = mask & (self.grain > 0.10) & (self.grain3 < frac)
        canvas[keep] = np.clip(soft[keep], 0, 255).astype(np.uint8)

    def make_group(self, X, Y, rad):
        """Contorno de uma parte: devolve (índices da região, índices do traço) ou None."""
        H, W = self.H, self.W
        if len(X) < 4:
            return None
        x0, x1 = max(int(X.min()) - 12, 0), min(int(X.max()) + 13, W)
        y0, y1 = max(int(Y.min()) - 12, 0), min(int(Y.max()) + 13, H)
        h, w = y1 - y0, x1 - x0
        if h < 8 or w < 8:
            return None
        m = np.zeros((h, w), np.uint8)
        splat(m, X - x0, Y - y0, np.full(len(X), 255, np.uint8), rad)

        def erode(a, r):
            o = a.copy()
            for dy, dx in ((r, 0), (-r, 0), (0, r), (0, -r)):
                o &= np.roll(np.roll(a, dy, 0), dx, 1)
            return o

        def thick(a):
            return a | np.roll(a, 1, 0) | np.roll(a, 1, 1) | np.roll(np.roll(a, 1, 0), 1, 1)

        solid = np.asarray(Image.fromarray(m).filter(ImageFilter.GaussianBlur(3.0))) > 97
        if solid.sum() < 20:
            solid = thick(m > 0)
        if erode(solid, 2).sum() < 0.12 * solid.sum():
            outline = solid.copy()          # parte fina (listra, tiara, haste): o próprio traço é a linha
        else:
            outline = solid & ~erode(solid, 1)
        line = thick(outline)
        ys, xs = np.nonzero(solid)
        ly, lx = np.nonzero(line)
        return ((ys + y0) * W + xs + x0).astype(np.int64), ((ly + y0) * W + lx + x0).astype(np.int64)

    @staticmethod
    def line_color(mean):
        return np.clip(np.asarray(mean, np.float32) * 0.7 + 112, 0, 255).astype(np.uint8)

    def build_static_lines(self):
        H, W = self.H, self.W
        S, ox, oy0 = H * 0.62, W / 2, H * 0.90
        st = self.static
        groups = []
        for zv in np.unique(st['z']):
            k = st['z'] == zv
            g = self.make_group(ox + st['x'][k] * S, oy0 - st['y'][k] * S, st['rad'][k])
            if g is not None:
                groups.append((float(zv), g[0], g[1], self.line_color(st['col'][k].mean(axis=0))))
        self._sg = groups

    def draw_lines(self, canvas, frac):
        H, W = self.H, self.W
        if self._sg is None:
            self.build_static_lines()
        dy = int(round(-self._bob)) * W      # a flutuação vertical é só um deslocamento
        groups = [(z, np.clip(si + dy, 0, H * W - 1), np.clip(li + dy, 0, H * W - 1), c)
                  for z, si, li, c in self._sg]
        for z, X, Y, rad, c in self._dynraw:
            g = self.make_group(X, Y, rad)
            if g is not None:
                groups.append((z, g[0], g[1], self.line_color(c)))
        A = np.zeros(H * W, bool)
        Lc = np.zeros((H * W, 3), np.uint8)
        for z, si, li, c in sorted(groups, key=lambda g: g[0]):
            A[si] = False        # o que está na frente apaga as linhas de trás
            A[li] = True
            Lc[li] = c
        A, Lc = A.reshape(H, W), Lc.reshape(H, W, 3)
        sel = A & (self.grain2 > 0.12) & (self.grain3 < frac)
        mod = (0.82 + 0.18 * self.grain[sel])[:, None]
        canvas[sel] = np.clip(Lc[sel] * mod, 0, 255).astype(np.uint8)

    def sketch(self, canvas, fig):
        """Transforma a figura num esboço: cores lavadas, falhas de grafite e contornos a lápis."""
        mask = fig.any(axis=2)
        if not mask.any():
            return
        paper = np.array([236, 230, 246], np.float32)
        soft = fig.astype(np.float32) * 0.55 + paper * 0.45
        keep = mask & (self.grain > 0.10)
        canvas[keep] = np.clip(soft[keep], 0, 255).astype(np.uint8)

        # contornos a lápis (finos): silhueta + mudanças de cor importantes por dentro
        def blur(a, r):
            return np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(r)), np.float32)

        def shifts(m, r=1):
            o = m.copy()
            for dy, dx in ((r, 0), (-r, 0), (0, r), (0, -r)):
                o &= np.roll(np.roll(m, dy, 0), dx, 1)
            return o        # erosão

        solid = blur((mask * 255).astype(np.uint8), 3.0) / 255.0 > 0.38
        outline = solid & ~shifts(solid, 1)
        lum = blur(fig, 2.6).mean(axis=2)
        gy, gx = np.gradient(lum)
        g = np.hypot(gx, gy)
        ridge = ((g >= np.roll(g, 1, 0)) & (g >= np.roll(g, -1, 0)) &
                 (g >= np.roll(g, 1, 1)) & (g >= np.roll(g, -1, 1)))
        inner = ridge & (g > EDGE_INNER) & shifts(solid, 4)
        thick = lambda m: m | np.roll(m, 1, 0) | np.roll(m, 1, 1) | np.roll(np.roll(m, 1, 0), 1, 1)
        out_l = thick(outline) & (self.grain2 > 0.12)       # contorno claro (aparece contra o céu)
        in_l = thick(inner) & (self.grain2 > 0.20) & ~out_l  # traços internos em grafite (escuros)
        ink = np.array([250, 245, 255], np.float32)
        pencil = np.array([84, 72, 118], np.float32)
        canvas[in_l] = np.clip(pencil * (0.75 + 0.25 * self.grain[in_l])[:, None], 0, 255).astype(np.uint8)
        canvas[out_l] = np.clip(ink * (0.78 + 0.22 * self.grain[out_l])[:, None], 0, 255).astype(np.uint8)

    def draw_sky(self, B, t, sky, glow):
        H, W = self.H, self.W
        S, ox, oy = H * 0.62, W / 2, H * 0.90
        one = lambda n: np.ones(n, np.float32)
        if sky <= 0.001:
            return
        N = self.neb
        B.add(N['fx'] * W, N['fy'] * H, N['col'] * sky, one(len(N['fx'])), np.full(len(N['fx']), -5000.0))

        T = self.tiny
        tw = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(t * T['sp'] + T['ph']))
        m = T['ap'] < sky
        B.add(T['fx'][m] * W, T['fy'][m] * H, T['col'][m] * (tw[m] * sky)[:, None], T['rad'][m], np.full(int(m.sum()), -4500.0))

        K = self.spk
        si = K['si']
        tws = 0.5 + 0.5 * np.sin(t * K['sp'] + K['ph'])
        vis = (K['ap'] < sky)[si]
        R = (K['R'] * H * (0.35 + 0.65 * tws))[si]
        X = K['fx'][si] * W + K['lx'] * R
        Y = K['fy'][si] * H + K['ly'] * R
        core = (np.hypot(K['lx'], K['ly']) < 0.28)[:, None]
        col = K['col'][si] * (0.45 + 0.55 * tws[si])[:, None] * sky
        col = np.where(core, np.minimum(col * 1.3 + 60, 255), col)
        B.add(X[vis], Y[vis], col[vis], one(int(vis.sum())), np.full(int(vis.sum()), -4000.0))

        F = self.st5
        si = F['si']
        ang = F['rot'][si] + F['om'][si] * t
        c, s = np.cos(ang), np.sin(ang)
        R = F['R'][si] * H * (0.75 + 0.25 * np.sin(t * 1.1 + F['ph'][si]))
        X = F['fx'][si] * W + R * (F['lx'] * c - F['ly'] * s)
        Y = F['fy'][si] * H + R * (F['lx'] * s + F['ly'] * c)
        vis = (F['ap'] < sky)[si]
        col = F['col'][si] * (0.6 + 0.4 * np.sin(t * 1.3 + F['ph'][si]))[:, None] * sky
        B.add(X[vis], Y[vis], col[vis], one(int(vis.sum())), np.full(int(vis.sum()), -3900.0))

        M = self.moon
        Rm = H
        pulse = 0.9 + 0.1 * math.sin(t * 0.7)
        B.add(M['cx'] * W + M['lx'] * Rm, M['cy'] * H + M['ly'] * Rm, M['col'] * (sky * pulse),
              np.full(len(M['lx']), 1.5, np.float32), np.full(len(M['lx']), -3800.0))
        Mh = self.mhalo
        hc = np.array([255, 240, 190], np.float32)[None, :] * (Mh['br'] * sky * (0.9 + 0.1 * math.sin(t * 0.9)))[:, None]
        B.add(M['cx'] * W + Mh['lx'] * Rm, M['cy'] * H + Mh['ly'] * Rm, hc, one(len(Mh['lx'])),
              np.full(len(Mh['lx']), -3850.0))

        for sh in self.shoot:
            u = ((t - sh['off']) % sh['per']) / 1.2
            if 0 <= u <= 1:
                dirv = np.array([-0.50, 0.24])
                head = np.array([sh['x0'], sh['y0']]) + dirv * u
                unit = dirv / np.hypot(*dirv)
                k = sh['k']
                px = (head[0] - unit[0] * k * 0.16) * W
                py = (head[1] - unit[1] * k * 0.16) * H
                br = ((1 - k) ** 1.5 * math.sin(math.pi * u) ** 0.5 * sky)[:, None]
                B.add(px, py, np.array([235, 235, 255], np.float32)[None, :] * br, np.where(k < 0.06, 2.0, 1.0).astype(np.float32),
                      np.full(len(k), -3700.0))

        C = self.cloud
        B.add(C['fx'] * W, C['fy'] * H, C['col'] * sky, C['rad'], np.full(len(C['fx']), -3000.0))

        if glow > 0.02 and ESTILO != "linhas":       # a aura preencheria o interior da figura
            h = self.halo
            rh = h['r'] * S * 1.2
            br = (h['b'] * glow * (0.85 + 0.3 * np.sin(t * 1.3 + h['fl'] * 6.28)))[:, None]
            B.add(ox + np.cos(h['a']) * rh, oy - 0.56 * S + np.sin(h['a']) * rh * 1.1,
                  np.array([150, 70, 240], np.float32)[None, :] * br, one(len(rh)), np.full(len(rh), -3100.0))

        if glow > 0.02:
            fl = self.floor
            pulse = 1 + 0.03 * math.sin(t * 1.5)
            fa, fb = 0.26 * pulse, 0.055 * pulse
            br = (fl['b'] * glow * (0.75 + 0.25 * math.sin(t * 1.5)))[:, None]
            B.add(ox + np.cos(fl['a']) * fl['r'] * fa * S, oy + 0.02 * S + np.sin(fl['a']) * fl['r'] * fb * S,
                  np.array([255, 170, 240], np.float32)[None, :] * br, one(len(br)), np.full(len(br), -2000.0))

    def draw_figure(self, Bf, B, t, build, face, glow):
        H, W = self.H, self.W
        S = H * 0.62
        ox = W / 2
        oy0 = H * 0.90
        bob = 0.010 * H * math.sin(t * 1.1) * float(smoothstep(build, 0.6, 1.0))
        oy = oy0 - bob   # sobe e desce de leve (ela flutua)
        one = lambda n: np.ones(n, np.float32)

        self._bob = bob

        def put(p, tx, ty, grp=False):
            x, y, col, rad, z = self.assemble(p, tx, ty, build, only_flying=ESTILO == "linhas")
            Bf.add(ox + x * S, oy - y * S, col, rad, z)
            if grp and self._want_lines:
                self._dynraw.append((float(p['z'].mean()), (ox + tx * S).astype(np.float32),
                                     (oy - ty * S).astype(np.float32), p['rad'], p['col'].mean(axis=0)))

        put(self.static, self.static['x'], self.static['y'])

        # cabelo
        for p in self.tails:
            m_, ph, s, lat = p['m'], p['ph'], p['s'], p['lat']
            sw1 = math.sin(t * 1.15 + ph)
            sw2 = math.sin(t * 1.15 + ph + 0.9)
            sw3 = math.sin(t * 0.8 + ph)
            P = ((m_ * 0.080, 0.965), (m_ * 0.245, 0.990 + 0.01 * sw3),
                 (m_ * (0.225 + 0.035 * sw1), 0.66), (m_ * (0.175 + 0.06 * sw2), 0.30 + 0.02 * sw1))
            x, y, dx, dy = bez(P, s)
            nn = np.hypot(dx, dy) + 1e-6
            w = 0.036 * (1 - 0.82 * s ** 1.3) * (0.9 + 0.1 * np.sin(s * 8 + ph))
            off = lat * w + 0.010 * np.sin(s * 9 - t * 2.2 + ph) * s
            put(p, x - dy / nn * off, y + dx / nn * off, True)

        # saia
        for p in self.skirts:
            a, b = p['a'], p['b']
            hw = 0.058 + (p['hw1'] - 0.058) * a ** 0.9
            sway = 0.010 * a ** 1.4 * np.sin(t * 1.5 + p['ph'] + a * 1.5)
            x = b * hw + sway
            y = (0.565 - p['L'] * a + 0.010 * a * np.sin(b * 6.0 + t * 1.8 + p['ph']) - 0.012 * a * (1 - b * b))
            put(p, x, y, True)

        # braços e varinha
        hand = None
        for p in self.arms:
            m_ = p['m']
            sw = math.sin(t * 1.1 + (0.4 if m_ < 0 else 1.3))
            sw2 = math.sin(t * 1.1 + (1.2 if m_ < 0 else 2.1))
            if m_ < 0:
                P = ((-0.082, 0.722), (-0.120, 0.675), (-0.150 + 0.012 * sw, 0.600), (-0.143 + 0.014 * sw2, 0.512 + 0.008 * sw))
            else:
                P = ((0.082, 0.722), (0.138, 0.742), (0.178 + 0.006 * sw, 0.795), (0.190 + 0.006 * sw2, 0.862 + 0.006 * sw))
                hand = P[3]
            s = p['s']
            x, y, dx, dy = bez(P, s)
            nn = np.hypot(dx, dy) + 1e-6
            w = 0.019 - 0.005 * s + 0.007 * smoothstep(s, 0.86, 0.95) * (1 - smoothstep(s, 0.97, 1.0))
            put(p, x - dy / nn * p['lat'] * w, y + dx / nn * p['lat'] * w, True)

        tilt = 0.10 + 0.02 * math.sin(t * 1.1)
        d = (math.sin(tilt), math.cos(tilt))
        nv = (math.cos(tilt), -math.sin(tilt))
        W_ = self.wand
        put(W_, hand[0] + W_['u'] * d[0] + W_['v'] * nv[0], hand[1] + W_['u'] * d[1] + W_['v'] * nv[1], True)

        # brilhos da varinha
        ocx = hand[0] + (self.wand_u0 + 0.007) * d[0] + 0.025 * nv[0]
        ocy = hand[1] + (self.wand_u0 + 0.007) * d[1] + 0.025 * nv[1]
        cx, cy = ox + ocx * S, oy - ocy * S
        if glow > 0.02:
            g = self.wglow
            rh = g['r'] * S * 0.12
            br = (np.clip(1 - g['r'] / 1.4, 0, 1) ** 1.5 * glow * 0.8 * (0.8 + 0.2 * np.sin(t * 2.0 + g['fl'] * 6.28)))[:, None]
            B.add(cx + np.cos(g['a']) * rh, cy + np.sin(g['a']) * rh,
                  np.array([255, 170, 245], np.float32)[None, :] * br, one(len(rh)), np.full(len(rh), 15.0))

            sp = self.wsp
            life = (t * 0.25 * sp['sp'] + sp['ph']) % 1.0
            rho = (0.02 + 0.20 * sp['rho']) * S * (0.5 + life)
            ang = sp['a'] + life * 1.5
            bs = (np.sin(np.pi * life) ** 0.8 * glow)[:, None]
            B.add(cx + np.cos(ang) * rho, cy + np.sin(ang) * rho - life * 0.10 * S, SPARK_COLS[sp['c']] * bs,
                  sp['rad'], np.full(len(rho), 2000.0))

            b = self.bsp
            life = (t * 0.06 * b['sp'] + b['ph']) % 1.0
            x = b['x0'] + 0.03 * np.sin(t * 0.8 * b['sp'] + b['ph'] * 6)
            y = -0.05 + life * 1.35
            bs = (np.sin(np.pi * life) ** 0.8 * glow)[:, None]
            B.add(ox + x * S, oy0 - y * S, SPARK_COLS[b['c']] * bs, b['rad'], np.full(len(x), 1900.0))


# ---------------------------------------------------------------- saídas
def save_png():
    H = 1000
    g = Guardian(H, H)
    times = [1.5, 4.5, 7.0, 9.5, 13.0, 20.0]
    frames = []
    for i, tt in enumerate(times):
        a = g.render(tt)
        Image.fromarray(a).save(f"guardia_{i + 1}.png")
        frames.append(Image.fromarray(a).resize((500, 500), Image.LANCZOS))
    strip = Image.new("RGB", (500 * len(frames), 500))
    for i, f in enumerate(frames):
        strip.paste(f, (500 * i, 0))
    strip.save("guardia_etapas.png")
    print("Saved: guardia_etapas.png and stage frames")


def save_gif(fps=10, size=480):
    g = Guardian(size, size)
    frames = [Image.fromarray(g.render(i / fps)).convert("P", palette=Image.ADAPTIVE)
              for i in range(int(T_END * fps) + 1)]
    frames[0].save("guardia.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=False)
    print("Saved: guardia.gif")


class App:
    def __init__(self):
        import tkinter as tk
        from PIL import ImageTk

        self.root = tk.Tk()
        self.root.title("Sailor")
        self.root.configure(bg="black")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        H = int(min(sh * WIN_FRAC, sw * 0.96))
        W = H
        self.root.geometry(f"{W}x{H}+{(sw - W) // 2}+{max(0, (sh - H) // 2 - 20)}")
        self.cv = tk.Canvas(self.root, width=W, height=H, bg="black", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.cv.create_text(W // 2, H // 2, text="preparing...", fill="#777777")
        self.root.update()

        self.g = Guardian(W, H)
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.g.render(0)))
        self.cv.delete("all")
        self.cv.create_image(0, 0, anchor="nw", image=self.photo)
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<space>", lambda e: self.toggle())
        self.root.bind("<r>", lambda e: self.restart())
        self.root.bind("<s>", lambda e: Image.fromarray(self.g.render(self.clock / TIME_SCALE)).save("guardia_frame.png"))
        self.clock, self.last, self.paused = 0.0, time.perf_counter(), False
        self.tick()

    def toggle(self):
        self.paused = not self.paused

    def restart(self):
        self.clock = 0.0

    def tick(self):
        now = time.perf_counter()
        if not self.paused:
            self.clock += now - self.last
        self.last = now
        t = self.clock / TIME_SCALE
        if LOOP and t > T_END + 4:
            self.clock, t = 0.0, 0.0
        self.photo.paste(Image.fromarray(self.g.render(t)))
        self.root.after(TICK_MS, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if "--png" in sys.argv:
        save_png()
    elif "--gif" in sys.argv:
        save_gif()
    else:
        App().run()