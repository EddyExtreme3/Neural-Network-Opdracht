"""
game.py — Speel het spel zelf en sla trainingsdata op.
Besturing: pijltje links / rechts, ESC om te stoppen.
Data wordt opgeslagen in training_data.csv
"""

import pygame
import random
import csv
import os
import sys

# ── Instellingen ─────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 600, 500
FPS = 60

PLAYER_W, PLAYER_H = 50, 20
PLAYER_SPEED = 6
PLAYER_Y = SCREEN_H - 50

BLOCK_W, BLOCK_H = 30, 30
BASE_BLOCK_SPEED = 3
BASE_BLOCK_COUNT = 1          # beginaantal blokken
SCORE_PER_EXTRA_BLOCK = 10    # elke 10 punten +1 blok
SPEED_INCREMENT = 0.003       # snelheidstoename per frame

CSV_FILE = "training_data.csv"

# ── Kleuren ───────────────────────────────────────────────────────────────────
BG_COLOR        = (15, 15, 30)
PLAYER_COLOR    = (0, 220, 180)
BLOCK_COLOR     = (220, 60, 60)
SCORE_COLOR     = (255, 255, 255)
GAMEOVER_COLOR  = (255, 80, 80)
GRID_COLOR      = (25, 25, 50)
ACCENT_COLOR    = (0, 220, 180)

# ─────────────────────────────────────────────────────────────────────────────

def normalize(value, max_val):
    return value / max_val


def nearest_block(blocks, player_x):
    """Geeft het blok terug dat het dichtst bij de speler is (laagste y = meeste gevaar)."""
    if not blocks:
        return None
    # Sorteer op y (meest naar beneden = meest gevaarlijk) én op horizontale afstand
    return min(blocks, key=lambda b: (
        abs((b[0] + BLOCK_W / 2) - (player_x + PLAYER_W / 2)) * 0.3
        - b[1]  # hoe lager op het scherm, hoe gevaarlijker
    ))


def spawn_block():
    x = random.randint(0, SCREEN_W - BLOCK_W)
    return [x, -BLOCK_H]


def draw_grid(surface):
    for x in range(0, SCREEN_W, 40):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, SCREEN_H))
    for y in range(0, SCREEN_H, 40):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (SCREEN_W, y))


def draw_player(surface, x):
    rect = pygame.Rect(x, PLAYER_Y, PLAYER_W, PLAYER_H)
    pygame.draw.rect(surface, PLAYER_COLOR, rect, border_radius=4)
    # Gloed-effect
    glow = pygame.Surface((PLAYER_W + 10, PLAYER_H + 10), pygame.SRCALPHA)
    pygame.draw.rect(glow, (*PLAYER_COLOR, 40), glow.get_rect(), border_radius=6)
    surface.blit(glow, (x - 5, PLAYER_Y - 5))


def draw_block(surface, bx, by):
    rect = pygame.Rect(bx, by, BLOCK_W, BLOCK_H)
    pygame.draw.rect(surface, BLOCK_COLOR, rect, border_radius=3)
    # Highlight bovenrand
    pygame.draw.rect(surface, (255, 120, 120), (bx, by, BLOCK_W, 4), border_radius=3)


def draw_danger_line(surface, block, player_x):
    """Trekt een lijn van de dichtstbijzijnde steen naar de speler — visueel hulpmiddel."""
    bx = block[0] + BLOCK_W // 2
    by = block[1] + BLOCK_H
    px = player_x + PLAYER_W // 2
    py = PLAYER_Y
    pygame.draw.line(surface, (255, 200, 0, 100), (bx, by), (px, py), 1)


def collect_data_row(player_x, blocks, action):
    nb = nearest_block(blocks, player_x)
    if nb is None:
        bx, by = 0.5, 0.0   # geen blokken → midden, ver weg
    else:
        bx = normalize(nb[0] + BLOCK_W / 2, SCREEN_W)
        by = normalize(nb[1] + BLOCK_H / 2, SCREEN_H)

    px = normalize(player_x + PLAYER_W / 2, SCREEN_W)
    dx = px - bx
    dy = by   # hoe groter, hoe dichterbij (y 0=boven 1=onder)

    return [round(px, 4), round(bx, 4), round(by, 4),
            round(dx, 4), round(dy, 4), action]


def game_over_screen(surface, font_big, font_small, score):
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    txt = font_big.render("GAME OVER", True, GAMEOVER_COLOR)
    surface.blit(txt, txt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 40)))

    stxt = font_small.render(f"Score: {score}", True, SCORE_COLOR)
    surface.blit(stxt, stxt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 10)))

    rtxt = font_small.render("Druk R om opnieuw te spelen  |  ESC om te stoppen", True, (180, 180, 180))
    surface.blit(rtxt, rtxt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 50)))


def run_game():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Dodge Blocks — Speel & Verzamel Data")
    clock = pygame.time.Clock()

    font_big   = pygame.font.SysFont("consolas", 48, bold=True)
    font_small = pygame.font.SysFont("consolas", 22)
    font_hud   = pygame.font.SysFont("consolas", 18)

    # CSV openen / aanmaken
    write_header = not os.path.exists(CSV_FILE)
    csv_file = open(CSV_FILE, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["player_x", "block_x", "block_y", "delta_x", "delta_y", "action"])

    rows_written = 0

    def reset():
        return {
            "player_x": SCREEN_W // 2 - PLAYER_W // 2,
            "blocks": [spawn_block()],
            "speed": BASE_BLOCK_SPEED,
            "score": 0,
            "frame": 0,
            "alive": True,
        }

    state = reset()

    while True:
        dt = clock.tick(FPS)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                csv_file.close()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    csv_file.close()
                    pygame.quit()
                    sys.exit()
                if not state["alive"] and event.key == pygame.K_r:
                    state = reset()

        if not state["alive"]:
            draw_grid(screen)
            game_over_screen(screen, font_big, font_small, state["score"])
            pygame.display.flip()
            continue

        # ── Input & actie ─────────────────────────────────────────────────────
        keys = pygame.key.get_pressed()
        action = 0  # stilstaan
        if keys[pygame.K_LEFT]:
            action = 1
            state["player_x"] = max(0, state["player_x"] - PLAYER_SPEED)
        elif keys[pygame.K_RIGHT]:
            action = 2
            state["player_x"] = min(SCREEN_W - PLAYER_W, state["player_x"] + PLAYER_SPEED)

        # ── Data opslaan ──────────────────────────────────────────────────────
        row = collect_data_row(state["player_x"], state["blocks"], action)
        writer.writerow(row)
        rows_written += 1

        # ── Logica ────────────────────────────────────────────────────────────
        state["frame"] += 1
        state["score"] = state["frame"] // FPS
        state["speed"] = BASE_BLOCK_SPEED + state["score"] * SPEED_INCREMENT * FPS

        # Gewenst aantal blokken op basis van score
        desired = BASE_BLOCK_COUNT + state["score"] // SCORE_PER_EXTRA_BLOCK
        while len(state["blocks"]) < desired:
            state["blocks"].append(spawn_block())

        # Blokken bewegen
        for b in state["blocks"]:
            b[1] += state["speed"]

        # Blokken die onderkant bereiken → respawn
        for b in state["blocks"]:
            if b[1] > SCREEN_H:
                b[0] = random.randint(0, SCREEN_W - BLOCK_W)
                b[1] = -BLOCK_H

        # Botsingsdetectie
        player_rect = pygame.Rect(state["player_x"], PLAYER_Y, PLAYER_W, PLAYER_H)
        for b in state["blocks"]:
            if player_rect.colliderect(pygame.Rect(b[0], b[1], BLOCK_W, BLOCK_H)):
                state["alive"] = False

        # ── Tekenen ───────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)
        draw_grid(screen)

        nb = nearest_block(state["blocks"], state["player_x"])
        if nb:
            draw_danger_line(screen, nb, state["player_x"])

        for b in state["blocks"]:
            draw_block(screen, b[0], b[1])

        draw_player(screen, state["player_x"])

        # HUD
        score_txt = font_small.render(f"Score: {state['score']}", True, SCORE_COLOR)
        screen.blit(score_txt, (10, 10))

        speed_txt = font_hud.render(f"Snelheid: {state['speed']:.1f}  |  Blokken: {len(state['blocks'])}  |  Data: {rows_written} rijen", True, (140, 140, 160))
        screen.blit(speed_txt, (10, SCREEN_H - 28))

        pygame.display.flip()

    csv_file.close()


if __name__ == "__main__":
    run_game()
