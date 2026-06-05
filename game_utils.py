"""
game_utils.py
-------------
Contains:
  - All layout / game constants
  - Renderer  –  draws one agent's world (main view) or a mini cell
  - draw_hud  –  stats panel with Q-table heatmap & score graph
"""

import pygame
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_AGENTS   = 50

WIN_W, WIN_H = 900, 620

MAIN_W  = 560          # px – main view width
MAIN_H  = 280          # px – main view height (logical game is 800×400)

GRID_COLS = 10
GRID_ROWS = 5
CELL_W    = MAIN_W // GRID_COLS   # 56
CELL_H    = 60
GRID_Y    = MAIN_H + 4            # grid starts just below main view

HUD_X = MAIN_W + 4
HUD_W = WIN_W - HUD_X - 2

GROUND_Y = 300          # logical game units (player lands here)


# ── Renderer ───────────────────────────────────────────────────────────────────

class Renderer:
    """
    Handles all drawing.
    Call tick_anims() once per frame to advance sprite animation.
    draw_agent() renders a full scaled view of one agent's world.
    draw_mini()  renders a tiny coloured cell for the 50-agent grid.
    """

    def __init__(self, assets: dict):
        self.sky      = assets["sky"]
        self.ground   = assets["ground"]
        self.snail_f  = assets["snail_frames"]   # list[Surface, Surface]
        self.fly_f    = assets["fly_frames"]
        self.walk_f   = assets["walk_frames"]
        self.jump_img = assets["jump"]

        self.snail_idx = 0
        self.fly_idx   = 0
        self._snail_t  = 0
        self._fly_t    = 0

    # ── Animation clock ───────────────────────────────────────────────────────

    def tick_anims(self, now: int):
        """Flip snail frame every 500 ms, fly frame every 200 ms."""
        if now - self._snail_t > 500:
            self.snail_idx = 1 - self.snail_idx
            self._snail_t  = now
        if now - self._fly_t > 200:
            self.fly_idx = 1 - self.fly_idx
            self._fly_t  = now

    # ── Full view ─────────────────────────────────────────────────────────────

    def draw_agent(self, surface: pygame.Surface, agent,
                   rect: pygame.Rect, font_small=None, label: str = ""):
        """
        Draw agent's game world scaled to fit `rect` on `surface`.
        Used for the main view (best-alive agent).
        """
        W, H   = rect.width, rect.height
        sx     = W / 800
        sy     = H / 400
        ox, oy = rect.x, rect.y

        # Sky background
        bg = pygame.transform.scale(self.sky, (W, H))
        surface.blit(bg, (ox, oy))

        # Ground strip
        gnd_y   = oy + int(GROUND_Y * sy)
        gnd_img = pygame.transform.scale(
            self.ground, (W, max(1, H - int(GROUND_Y * sy)))
        )
        surface.blit(gnd_img, (ox, gnd_y))

        # Obstacles
        snail_s = self.snail_f[self.snail_idx]
        fly_s   = self.fly_f[self.fly_idx]
        for ob in agent.obstacle_list:
            img = snail_s if ob.bottom == GROUND_Y else fly_s
            sw  = max(1, int(img.get_width()  * sx))
            sh  = max(1, int(img.get_height() * sy))
            surface.blit(pygame.transform.scale(img, (sw, sh)),
                         (ox + int(ob.x * sx), oy + int(ob.y * sy)))

        # Player
        p_img = (self.jump_img if agent.player_bottom < GROUND_Y
                 else self.walk_f[int(agent.player_index) % 2])
        pw = max(1, int(p_img.get_width()  * sx))
        ph = max(1, int(p_img.get_height() * sy))
        scaled_p = pygame.transform.scale(p_img, (pw, ph))
        px = ox + int(80 * sx) - pw // 2
        py = oy + int(agent.player_bottom * sy) - ph
        surface.blit(scaled_p, (px, py))

        # Optional label (agent id + score)
        if font_small and label:
            surface.blit(font_small.render(label, True, (255, 255, 255)),
                         (ox + 2, oy + 2))

    # ── Mini cell ─────────────────────────────────────────────────────────────

    def draw_mini(self, surface: pygame.Surface, agent, rect: pygame.Rect):
        """
        Tiny 56×58 px cell showing agent status:
          green  = alive on ground
          bright = alive in air (jumping)
          red    = dead / just reset
          yellow bar at bottom = normalised score
        """
        if agent.alive and agent.player_bottom < GROUND_Y:
            color = (30, 200, 80)     # jumping – brighter green
        elif agent.alive:
            color = (50, 150, 60)     # running
        else:
            color = (160, 45, 45)     # dead

        pygame.draw.rect(surface, color, rect, border_radius=3)
        if getattr(agent, "is_champion", False):
            pygame.draw.rect(surface, (255, 240, 90), rect, 2, border_radius=3)

        # Score bar (yellow, fills left→right)
        max_score = 60
        bar_w = int(min(agent.score / max_score, 1.0) * (rect.width - 4))
        if bar_w > 0:
            pygame.draw.rect(surface,
                             (255, 215, 50),
                             pygame.Rect(rect.x + 2, rect.bottom - 6,
                                         bar_w, 4),
                             border_radius=2)


# ── HUD ────────────────────────────────────────────────────────────────────────

def draw_hud(surface: pygame.Surface, qtable, agents: list,
             global_ep: int, rect: pygame.Rect,
             generation: int = 1, champion_id: int = 0):
    """
    Render the right-side stats panel:
      • Live swarm metrics (alive count, episodes, Q-updates)
      • Score stats (max / avg / best ever)
      • Epsilon stats across the population
      • Q-table heatmap  (dist × height → learned value)
      • Top-20 agents best-score line graph
    """
    x, y, W, H = rect.x, rect.y, rect.width, rect.height

    # Semi-transparent dark background
    panel = pygame.Surface((W, H), pygame.SRCALPHA)
    panel.fill((10, 10, 20, 210))
    surface.blit(panel, (x, y))

    alive  = sum(1 for a in agents if a.alive)
    scores = [a.score      for a in agents]
    bests  = [a.best_score for a in agents]
    epss   = [a.epsilon    for a in agents]
    dodges = [getattr(a, "dodges", 0) for a in agents]

    sm = pygame.font.SysFont("consolas", 13, bold=True)

    lines = [
        ("── SWARM  Q-LEARNING ──",   (100, 220, 255)),
        (f"Generation    : {generation}",             (100, 220, 255)),
        (f"Champion      : #{champion_id}",            (255, 240, 90)),
        (f"Agents alive  : {alive} / {len(agents)}", (180, 255, 180)),
        (f"Global ep     : {global_ep}",              (180, 255, 180)),
        (f"Q-updates     : {qtable.total_updates:,}", (180, 255, 180)),
        (f"Dodges total  : {sum(dodges):,}",           (180, 255, 180)),
        ("",                                          None),
        (f"Score  max    : {max(scores)}",            (255, 220, 80)),
        (f"Score  avg    : {sum(scores)/len(scores):.1f}", (255, 220, 80)),
        (f"Best  ever    : {max(bests)}",             (80, 255, 160)),
        ("",                                          None),
        (f"ε  min        : {min(epss):.3f}",          (255, 160, 80)),
        (f"ε  max        : {max(epss):.3f}",          (255, 160, 80)),
        (f"ε  avg        : {sum(epss)/len(epss):.3f}",(255, 160, 80)),
        ("",                                          None),
        ("── Q-TABLE  HEATMAP ──",    (160, 160, 255)),
    ]

    cy = y + 6
    for text, col in lines:
        if text:
            surface.blit(sm.render(text, True, col), (x + 6, cy))
        cy += 17

    # Heatmap: distance bucket x obstacle-height bucket -> max Q-value
    heat_y  = cy + 2
    height_buckets = getattr(qtable, "HEIGHT_BUCKETS", 2)
    cell_w  = max(4, W // (qtable.DIST_BUCKETS * height_buckets))
    cell_h  = 18
    for d in range(qtable.DIST_BUCKETS):
        for h in range(height_buckets):
            maxq = float(np.max(qtable.table[d, h, ...]))
            norm = min(max(maxq, 0.0) / 20.0, 1.0)
            col  = (int(norm * 50), int(norm * 200), int(norm * 100))
            cx   = x + (d * height_buckets + h) * cell_w
            pygame.draw.rect(surface, col,
                             pygame.Rect(cx, heat_y, cell_w - 1, cell_h - 1),
                             border_radius=1)

    # Axis labels
    lbl_d = sm.render("dist ->", True, (100, 100, 140))
    surface.blit(lbl_d, (x + 4, heat_y + cell_h + 1))

    # Score graph: top-20 agents by best score (descending → line dips right)
    graph_y = heat_y + cell_h + 18
    graph_h = H - (graph_y - y) - 20
    graph_w = W - 8

    if graph_h > 20:
        pygame.draw.rect(surface, (22, 22, 40),
                         pygame.Rect(x + 4, graph_y, graph_w, graph_h))

        bests_sorted = sorted(bests, reverse=True)[:20]
        if len(bests_sorted) >= 2:
            mx  = max(bests_sorted) or 1
            pts = []
            for i, s in enumerate(bests_sorted):
                px2 = x + 4 + int(i / (len(bests_sorted) - 1) * graph_w)
                py2 = graph_y + graph_h - int(s / mx * graph_h)
                pts.append((px2, py2))
            pygame.draw.lines(surface, (80, 220, 140), False, pts, 2)

        gl = sm.render("Top-20 best scores", True, (90, 90, 130))
        surface.blit(gl, (x + 6, graph_y + graph_h + 2))
