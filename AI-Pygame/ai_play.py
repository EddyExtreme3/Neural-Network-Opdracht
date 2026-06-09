"""
ai_play.py — Het getrainde neuraal netwerk bestuurt de speler.
Het model neemt elke frame een beslissing op basis van spelstatus.
Druk ESC om te stoppen, R om opnieuw te starten.
"""

import pygame
import random
import sys
import numpy as np
import tensorflow as tf

# ── Zelfde constanten als game.py ─────────────────────────────────────────────
SCREEN_W, SCREEN_H = 600, 500
FPS = 60

PLAYER_W, PLAYER_H = 50, 20
PLAYER_SPEED = 6
PLAYER_Y = SCREEN_H - 50

BLOCK_W, BLOCK_H = 30, 30
BASE_BLOCK_SPEED = 3
BASE_BLOCK_COUNT = 1
SCORE_PER_EXTRA_BLOCK = 10
SPEED_INCREMENT = 0.003

MODEL_FILE = "model.keras"

# ── Kleuren ───────────────────────────────────────────────────────────────────
BG_COLOR       = (15, 15, 30)
PLAYER_COLOR   = (80, 255, 180)
BLOCK_COLOR    = (220, 60, 60)
SCORE_COLOR    = (255, 255, 255)
GAMEOVER_COLOR = (255, 80, 80)
GRID_COLOR     = (25, 25, 50)
AI_COLOR       = (255, 220, 0)
PANEL_COLOR    = (20, 20, 45)

# ─────────────────────────────────────────────────────────────────────────────

def normalize(value, max_val):
    return value / max_val


def nearest_block(blocks, player_x):
    if not blocks:
        return None
    return min(blocks, key=lambda b: (
        abs((b[0] + BLOCK_W / 2) - (player_x + PLAYER_W / 2)) * 0.3
        - b[1]
    ))


def spawn_block():
    return [random.randint(0, SCREEN_W - BLOCK_W), -BLOCK_H]


def get_features(player_x, blocks):
    nb = nearest_block(blocks, player_x)
    if nb is None:
        bx, by = 0.5, 0.0
    else:
        bx = normalize(nb[0] + BLOCK_W / 2, SCREEN_W)
        by = normalize(nb[1] + BLOCK_H / 2, SCREEN_H)

    px = normalize(player_x + PLAYER_W / 2, SCREEN_W)
    dx = px - bx
    dy = by
    return np.array([[px, bx, by, dx, dy]], dtype=np.float32)


def draw_grid(surface):
    for x in range(0, SCREEN_W, 40):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, SCREEN_H))
    for y in range(0, SCREEN_H, 40):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (SCREEN_W, y))


def draw_player(surface, x):
    rect = pygame.Rect(x, PLAYER_Y, PLAYER_W, PLAYER_H)
    pygame.draw.rect(surface, PLAYER_COLOR, rect, border_radius=4)
    # AI-label boven speler
    font = pygame.font.SysFont("consolas", 12)
    txt = font.render("AI", True, AI_COLOR)
    surface.blit(txt, (x + PLAYER_W // 2 - txt.get_width() // 2, PLAYER_Y - 18))


def draw_block(surface, bx, by):
    rect = pygame.Rect(bx, by, BLOCK_W, BLOCK_H)
    pygame.draw.rect(surface, BLOCK_COLOR, rect, border_radius=3)
    pygame.draw.rect(surface, (255, 120, 120), (bx, by, BLOCK_W, 4), border_radius=3)


def draw_danger_line(surface, block, player_x):
    bx = block[0] + BLOCK_W // 2
    by = block[1] + BLOCK_H
    px = player_x + PLAYER_W // 2
    pygame.draw.line(surface, (255, 200, 0), (bx, by), (px, PLAYER_Y), 1)


def draw_ai_panel(surface, font, probs, action, score, best_score):
    """Paneel rechtsonder met AI-beslissing en kansen."""
    panel_x, panel_y = SCREEN_W - 200, SCREEN_H - 110
    pygame.draw.rect(surface, PANEL_COLOR, (panel_x - 8, panel_y - 8, 200, 110), border_radius=6)
    pygame.draw.rect(surface, AI_COLOR, (panel_x - 8, panel_y - 8, 200, 110), 1, border_radius=6)

    labels   = ["Stilstaan", "Links    ", "Rechts   "]
    actie_nl = labels[action]

    for i, (lbl, p) in enumerate(zip(labels, probs)):
        color = AI_COLOR if i == action else (160, 160, 160)
        txt = font.render(f"{lbl}: {p:.2f}", True, color)
        surface.blit(txt, (panel_x, panel_y + i * 22))

    best_txt = font.render(f"Best: {best_score}", True, (180, 255, 200))
    surface.blit(best_txt, (panel_x, panel_y + 70))


def game_over_screen(surface, font_big, font_small, score, best_score):
    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    txt = font_big.render("GAME OVER", True, GAMEOVER_COLOR)
    surface.blit(txt, txt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 50)))

    stxt = font_small.render(f"Score: {score}  |  Best: {best_score}", True, SCORE_COLOR)
    surface.blit(stxt, stxt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 5)))

    rtxt = font_small.render("Druk R om opnieuw  |  ESC om te stoppen", True, (180, 180, 180))
    surface.blit(rtxt, rtxt.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 45)))


def run_ai():
    # Model laden
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        print(f"Model geladen: {MODEL_FILE}")
        model.summary()
    except Exception as e:
        print(f"Fout bij laden model: {e}")
        print("Train eerst het model met train.py")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Dodge Blocks — AI speelt")
    clock = pygame.time.Clock()

    font_big   = pygame.font.SysFont("consolas", 48, bold=True)
    font_small = pygame.font.SysFont("consolas", 22)
    font_hud   = pygame.font.SysFont("consolas", 16)

    best_score = 0

    def reset():
        return {
            "player_x": SCREEN_W // 2 - PLAYER_W // 2,
            "blocks":   [spawn_block()],
            "speed":    BASE_BLOCK_SPEED,
            "score":    0,
            "frame":    0,
            "alive":    True,
        }

    state = reset()
    last_probs  = [0.33, 0.33, 0.33]
    last_action = 0

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if not state["alive"] and event.key == pygame.K_r:
                    state = reset()

        if not state["alive"]:
            draw_grid(screen)
            game_over_screen(screen, font_big, font_small, state["score"], best_score)
            pygame.display.flip()
            continue

        # ── AI beslist ────────────────────────────────────────────────────────
        features = get_features(state["player_x"], state["blocks"])
        probs    = model.predict(features, verbose=0)[0]   # shape (3,)
        action   = int(np.argmax(probs))

        last_probs  = probs.tolist()
        last_action = action

        # Actie uitvoeren
        if action == 1:
            state["player_x"] = max(0, state["player_x"] - PLAYER_SPEED)
        elif action == 2:
            state["player_x"] = min(SCREEN_W - PLAYER_W, state["player_x"] + PLAYER_SPEED)

        # ── Spellogica ────────────────────────────────────────────────────────
        state["frame"] += 1
        state["score"]  = state["frame"] // FPS
        state["speed"]  = BASE_BLOCK_SPEED + state["score"] * SPEED_INCREMENT * FPS

        desired = BASE_BLOCK_COUNT + state["score"] // SCORE_PER_EXTRA_BLOCK
        while len(state["blocks"]) < desired:
            state["blocks"].append(spawn_block())

        for b in state["blocks"]:
            b[1] += state["speed"]

        for b in state["blocks"]:
            if b[1] > SCREEN_H:
                b[0] = random.randint(0, SCREEN_W - BLOCK_W)
                b[1] = -BLOCK_H

        player_rect = pygame.Rect(state["player_x"], PLAYER_Y, PLAYER_W, PLAYER_H)
        for b in state["blocks"]:
            if player_rect.colliderect(pygame.Rect(b[0], b[1], BLOCK_W, BLOCK_H)):
                state["alive"] = False
                best_score = max(best_score, state["score"])

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

        speed_txt = font_hud.render(
            f"Snelheid: {state['speed']:.1f}  |  Blokken: {len(state['blocks'])}",
            True, (140, 140, 160)
        )
        screen.blit(speed_txt, (10, SCREEN_H - 28))

        draw_ai_panel(screen, font_hud, last_probs, last_action, state["score"], best_score)

        pygame.display.flip()


if __name__ == "__main__":
    run_ai()
