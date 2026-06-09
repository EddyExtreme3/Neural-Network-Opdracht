# Dodge Blocks — AI Practicum

## Installatie

```bash
pip install pygame tensorflow scikit-learn pandas matplotlib numpy
```

---

## Stap 1 — Speel het spel & verzamel data

```bash
python game.py
```

- **Pijltje links/rechts** om te bewegen
- Ontwijkt de rode vallende blokken
- Elke frame wordt automatisch opgeslagen in `training_data.csv`
- Speel **minimaal 3–5 potjes** voor genoeg data (~1000+ rijen)
- **ESC** om te stoppen

---

## Stap 2 — Train het neuraal netwerk

```bash
python AI-Pygame/train.py
```

- Leest `training_data.csv`
- Traint een neuraal netwerk met 2 hidden layers (32 → 16 → 3)
- Voert automatisch experimenten uit (dataset grootte, epochs)
- Slaat grafieken op als `.png` bestanden
- Slaat het model op als `model.keras`

---

## Stap 3 — Laat de AI spelen

```bash
python AI-Pygame/ai_play.py
```

- Het getrainde model bestuurt de speler automatisch
- Rechtsonder zie je een **AI-paneel** met:
  - Kans per actie (Softmax output)
  - Welke actie gekozen wordt
  - Beste score van deze sessie
- **R** om opnieuw te starten, **ESC** om te stoppen

---

## Bestandsoverzicht

| Bestand | Beschrijving |
|---|---|
| `game.py` | PyGame spel — jij speelt, data wordt opgeslagen |
| `train.py` | Neuraal netwerk trainen + experimenten |
| `ai_play.py` | AI speelt het spel met het getrainde model |
| `training_data.csv` | Verzamelde trainingsdata (wordt aangemaakt) |
| `model.keras` | Getraind model (wordt aangemaakt) |

---

## Hoe werkt het neuraal netwerk?

### Input (5 features per frame)
| Feature | Beschrijving |
|---|---|
| `player_x` | Genormaliseerde x-positie speler (0–1) |
| `block_x` | X-positie dichtstbijzijnde blok (0–1) |
| `block_y` | Y-positie dichtstbijzijnde blok (0–1) |
| `delta_x` | Horizontaal verschil speler–blok |
| `delta_y` | Verticaal verschil (hoe dichtbij?) |

### Architectuur
```
Input (5)  →  Dense 32 (ReLU)  →  Dense 16 (ReLU)  →  Output 3 (Softmax)
```

### Output (3 klassen)
- **0** = Stilstaan
- **1** = Links
- **2** = Rechts

### Twee-lagen principe
- PyGame tekent in pixels (visuele laag)
- Het netwerk ziet alleen 5 getallen (logische laag)
- Dit maakt de AI snel genoeg om elke frame een beslissing te nemen
