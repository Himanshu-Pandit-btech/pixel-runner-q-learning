"""
main.py
-------
Entry point for Pixel Runner – 50 Parallel Q-Learning Agents.

Run:
    python main.py

Controls:
    S      →  manually save q_table.npy
    Q/Esc  →  save and quit
"""

import pygame
from sys import exit

from game_utils import (
    NUM_AGENTS,
    WIN_W, WIN_H,
    MAIN_W, MAIN_H,
    GRID_COLS, GRID_ROWS, CELL_W, CELL_H, GRID_Y,
    HUD_X, HUD_W,
    Renderer,
    draw_hud,
)
from agent import SharedQTable, Agent

GENERATION_WIN_SCORE = 120

# ── Asset loader ───────────────────────────────────────────────────────────────

def load_assets() -> dict:
    sky    = pygame.image.load("graphics/Sky.png").convert()
    ground = pygame.image.load("graphics/ground.png").convert()

    snail_frames = [
        pygame.image.load("graphics/snail/snail1.png").convert_alpha(),
        pygame.image.load("graphics/snail/snail2.png").convert_alpha(),
    ]
    fly_frames = [
        pygame.image.load("graphics/fly/fly1.png").convert_alpha(),
        pygame.image.load("graphics/fly/fly2.png").convert_alpha(),
    ]
    walk_frames = [
        pygame.image.load("graphics/Player/player_walk_1.png").convert_alpha(),
        pygame.image.load("graphics/Player/player_walk_2.png").convert_alpha(),
    ]
    jump_img = pygame.image.load("graphics/Player/jump.png").convert_alpha()

    return dict(
        sky          = sky,
        ground       = ground,
        snail_frames = snail_frames,
        fly_frames   = fly_frames,
        walk_frames  = walk_frames,
        jump         = jump_img,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(
        f"Pixel Runner — {NUM_AGENTS} Parallel Q-Learning Agents"
    )
    clock    = pygame.time.Clock()
    font_big = pygame.font.Font("font/Pixeltype.ttf", 36)
    font_sm  = pygame.font.SysFont("consolas", 11)

    # ── Assets ────────────────────────────────────────────────────────────────
    assets   = load_assets()
    snail_w, snail_h = assets["snail_frames"][0].get_size()
    fly_w,   fly_h   = assets["fly_frames"][0].get_size()

    # ── Audio ─────────────────────────────────────────────────────────────────
    try:
        bg_music = pygame.mixer.Sound("audio/music.wav")
        bg_music.set_volume(0.4)
        bg_music.play(loops=-1)
    except Exception as e:
        print(f"[Audio] Could not load music: {e}")

    # ── Q-table & agents ──────────────────────────────────────────────────────
    qtable = SharedQTable(alpha=0.18, gamma=0.96)

    # Keep a mix of exploiters and explorers so the shared table improves fast.
    agents = [
        Agent(
            agent_id      = i,
            qtable        = qtable,
            epsilon       = 0.08 + (0.95 - 0.08) * (i / (NUM_AGENTS - 1)),
            epsilon_min   = 0.03 if i < NUM_AGENTS // 2 else 0.08,
            epsilon_decay = 0.9992,
        )
        for i in range(NUM_AGENTS)
    ]

    renderer  = Renderer(assets)
    global_ep = 0
    generation = 1
    champion_id = 0
    last_autosave = pygame.time.get_ticks()

    # ── Layout rects ──────────────────────────────────────────────────────────
    main_rect = pygame.Rect(0, 0, MAIN_W, MAIN_H)
    hud_rect  = pygame.Rect(HUD_X, 0, HUD_W, WIN_H)

    grid_rects = [
        pygame.Rect(
            col * CELL_W,
            GRID_Y + row * CELL_H,
            CELL_W - 2,
            CELL_H - 2,
        )
        for row in range(GRID_ROWS)
        for col in range(GRID_COLS)
    ]

    def best_agent_idx() -> int:
        """Index of the best visible agent, preferring agents still alive."""
        alive_ids = [i for i, agent in enumerate(agents) if agent.alive]
        if alive_ids:
            return max(alive_ids, key=lambda i: agents[i].score)
        return max(range(NUM_AGENTS), key=lambda i: agents[i].fitness())

    def evolve_generation():
        nonlocal generation, champion_id

        champion = max(agents, key=lambda a: a.fitness())
        champion_id = champion.id
        generation += 1
        print(
            f"[Gen {generation}] champion=#{champion.id} "
            f"best={champion.best_score} fitness={champion.fitness():.1f}"
        )

        offspring_i = 0
        offspring_count = max(1, NUM_AGENTS - 1)
        for agent in agents:
            if agent is champion:
                agent.promote_champion(generation)
            else:
                explore_rank = offspring_i / offspring_count
                agent.become_offspring(champion, generation, explore_rank)
                offspring_i += 1

        qtable.save()

    # ── Game loop ─────────────────────────────────────────────────────────────
    while True:
        now = pygame.time.get_ticks()

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                qtable.save()
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    qtable.save()
                    pygame.quit()
                    exit()
                if event.key == pygame.K_s:
                    qtable.save()

        # Step all living agents. Dead agents wait for the next generation.
        for agent in agents:
            was_alive = agent.alive
            alive = agent.step(snail_w, snail_h, fly_w, fly_h)
            if was_alive and not alive:
                global_ep += 1

        if now - last_autosave > 30000:
            qtable.save()
            last_autosave = now

        if (
            all(not agent.alive for agent in agents)
            or max(agent.score for agent in agents) >= GENERATION_WIN_SCORE
        ):
            evolve_generation()

        # Animate obstacle sprites
        renderer.tick_anims(now)

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill((20, 20, 30))

        # Main view – best alive agent
        bi = best_agent_idx()
        renderer.draw_agent(
            screen, agents[bi], main_rect,
            font_sm, f"#{bi}  score:{agents[bi].score}"
        )

        # Score overlay on main view
        sc_surf = font_big.render(
            f"Best: {max(a.best_score for a in agents)}  "
            f"Now: {agents[bi].score}  "
            f"Gen: {generation}  "
            f"Ep: {global_ep}",
            False, (64, 64, 64),
        )
        sc_rect = sc_surf.get_rect(centerx=MAIN_W // 2, y=6)
        pygame.draw.rect(screen, (192, 232, 236),
                         sc_rect.inflate(16, 8), border_radius=4)
        screen.blit(sc_surf, sc_rect)

        # Divider – main view / mini grid
        pygame.draw.line(screen, (60, 60, 80),
                         (0, MAIN_H + 2), (MAIN_W, MAIN_H + 2))

        # Mini grid – one cell per agent
        for i, agent in enumerate(agents):
            renderer.draw_mini(screen, agent, grid_rects[i])
            screen.blit(
                font_sm.render(str(i), True, (0, 0, 0)),
                (grid_rects[i].x + 2, grid_rects[i].y + 2),
            )

        # HUD panel
        pygame.draw.rect(screen, (15, 15, 25), hud_rect)
        draw_hud(screen, qtable, agents, global_ep, hud_rect,
                 generation, champion_id)

        # Divider – main area / HUD
        pygame.draw.line(screen, (60, 60, 80),
                         (HUD_X - 2, 0), (HUD_X - 2, WIN_H))

        # Status bar
        status_txt = (
            f"[S] save  [Q/Esc] quit  |  "
            f"alive={sum(a.alive for a in agents)}/{NUM_AGENTS}  |  "
            f"gen={generation} champ=#{champion_id}  |  "
            f"Q-updates={qtable.total_updates:,}  |  "
            f"FPS={int(clock.get_fps())}"
        )
        screen.blit(
            font_sm.render(status_txt, True, (110, 110, 150)),
            (4, WIN_H - 16),
        )

        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()
