import pygame
from random import randint, choice, random as rnd
import numpy as np
import os
 
from game_utils import GROUND_Y

FPS = 60
MAX_SPEED_MULT = 2.8
 
# ── Shared Q-Table ─────────────────────────────────────────────────────────────
 
class SharedQTable:
    DIST_BUCKETS = 16
    HEIGHT_BUCKETS = 3
    PLAYER_Y_BUCKETS = 4
    GRAV_BUCKETS = 5
    SPEED_BUCKETS = 4
    Q_FILE       = "q_table_genetic_v3.npy"
    ACTIONS      = 2          # 0 = idle, 1 = jump

    def __init__(self, alpha=0.12, gamma=0.95):
        self.alpha = alpha
        self.gamma = gamma
        self.shape = (
            self.DIST_BUCKETS,
            self.HEIGHT_BUCKETS,
            self.PLAYER_Y_BUCKETS,
            self.GRAV_BUCKETS,
            self.SPEED_BUCKETS,
            self.ACTIONS,
        )

        self.table = None

        if os.path.exists(self.Q_FILE):
            try:
                loaded = np.load(self.Q_FILE)

                if loaded.shape == self.shape:
                    self.table = loaded
                    print(
                        f"[QTable] Loaded {self.Q_FILE} "
                        f"(shape={loaded.shape})"
                    )
                else:
                    print(
                        f"[QTable] Shape mismatch!\n"
                        f"Saved:    {loaded.shape}\n"
                        f"Expected: {self.shape}\n"
                        f"Creating new table..."
                    )
            except Exception as e:
                print(f"[QTable] Load failed: {e}")

        if self.table is None:
            self.table = np.zeros(self.shape, dtype=np.float32)
            self._add_jump_timing_prior()

        self.total_updates = 0

    def discretise(self, raw_state):
        dist_x, obs_y, player_bottom, player_grav, speed_mult = raw_state

        dist_b = min(
            int(max(dist_x, 0) / 50),
            self.DIST_BUCKETS - 1
        )

        if obs_y < 230:
            height_b = 2        # flying obstacle: usually stay down
        elif obs_y < GROUND_Y:
            height_b = 1
        else:
            height_b = 0        # ground obstacle: prepare to jump

        air = max(0.0, GROUND_Y - player_bottom)
        if air <= 2:
            player_y_b = 0
        elif air < 45:
            player_y_b = 1
        elif air < 95:
            player_y_b = 2
        else:
            player_y_b = 3

        if player_grav < -10:
            grav_b = 0
        elif player_grav < 0:
            grav_b = 1
        elif player_grav <= 4:
            grav_b = 2
        elif player_grav <= 10:
            grav_b = 3
        else:
            grav_b = 4

        if speed_mult < 1.35:
            speed_b = 0
        elif speed_mult < 1.8:
            speed_b = 1
        elif speed_mult < 2.4:
            speed_b = 2
        else:
            speed_b = 3

        return (dist_b, height_b, player_y_b, grav_b, speed_b)
    
    def get_action(self, raw_state, epsilon):
        if rnd() < epsilon:
            # Exploration is guided: random flailing teaches very slowly here.
            if rnd() < 0.70:
                return self.teacher_action(raw_state)
            return choice([0, 1])

        s = self.discretise(raw_state)

        q = self.table[s]

        # Use the timing heuristic until the table has a clear preference.
        if abs(q[0] - q[1]) < 0.15:
            return self.teacher_action(raw_state)

        return int(np.argmax(q))

    def teacher_action(self, raw_state):
        dist_x, obs_y, player_bottom, _, speed_mult = raw_state
        grounded = player_bottom >= GROUND_Y - 1
        ground_obstacle = obs_y >= 230
        if not grounded or not ground_obstacle:
            return 0

        sweet_min = max(75, 150 - int((speed_mult - 1.0) * 35))
        sweet_max = max(145, 245 - int((speed_mult - 1.0) * 25))
        return 1 if sweet_min <= dist_x <= sweet_max else 0

    def _add_jump_timing_prior(self):
        for dist_b in range(self.DIST_BUCKETS):
            approx_dist = dist_b * 50 + 25
            for speed_b in range(self.SPEED_BUCKETS):
                # Grounded, ground obstacle: prefer jump in the useful window.
                if 125 <= approx_dist <= 275:
                    self.table[dist_b, 0, 0, 2, speed_b, 1] = 1.5
                    self.table[dist_b, 0, 0, 3, speed_b, 1] = 1.0
                else:
                    self.table[dist_b, 0, 0, :, speed_b, 0] = 0.4

                # Flying obstacles: staying down is usually the winning move.
                self.table[dist_b, 2, :, :, speed_b, 0] = 1.0

    def update(self, rs, action, reward, rs2, done):
        s  = self.discretise(rs)
        s2 = self.discretise(rs2)

        target = (
            reward
            if done
            else reward + self.gamma * np.max(self.table[s2])
        )

        self.table[s][action] += (
            self.alpha *
            (target - self.table[s][action])
        )

        self.total_updates += 1

    def save(self):
        np.save(self.Q_FILE, self.table)
        print(f"[QTable] Saved -> {self.Q_FILE}")
 
 
# ── Individual Agent ───────────────────────────────────────────────────────────
 
class Agent:
    """
    Owns one game simulation (obstacles, player physics, score).
    Reads/writes the shared Q-table every frame.
 
    Epsilon spread: agent 0 = exploit-heavy, agent 49 = explore-heavy.
    All agents decay toward epsilon_min over time.
    """
 
    def __init__(self, agent_id, qtable: SharedQTable,
                 epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9985):
        self.id            = agent_id
        self.qtable        = qtable
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
 
        # Persistent stats
        self.episodes    = 0
        self.best_score  = 0
        self.last_score  = 0
        self.score_hist  = []       # last 50 episode scores (for graph)
        self.dodges      = 0
        self.generation  = 0
        self.parent_id   = None
        self.is_champion = False
 
        # Per-generation episode state. Dead agents wait until evolution.
        self.alive = True
        self.score = 0
        self._init_game()
 
    # ── Game reset ────────────────────────────────────────────────────────────
 
    def _init_game(self):
        self.obstacle_list  = []
        self.player_bottom  = float(GROUND_Y)
        self.player_gravity = 0.0
        self.player_index   = 0.0        # walk animation frame
        self.frame_count    = 0
        self.alive          = True
        self.score          = 0
        self.speed_mult     = 1.0
        self.dodged_ids     = set()
        self.spawned_count  = 0
        self.spawn_timer    = randint(66, 114)
 
    # ── Main step (called every frame) ───────────────────────────────────────
 
    def step(self, snail_w, snail_h, fly_w, fly_h):
        """
        Advance one simulation frame.
        Returns True while alive, False after death.
        """
        if not self.alive:
            return False

        # Score & difficulty
        self.frame_count += 1
        self.score = self.frame_count // FPS
        self.best_score = max(self.best_score, self.score)
        self.speed_mult = min(MAX_SPEED_MULT, 1.0 + self.score * 0.012)
 
        # Spawn obstacle
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            # Curriculum: learn clean ground jumps first, then add flies.
            etype = "snail"
            if self.score >= 18:
                etype = choice(["fly", "snail", "snail", "snail", "snail", "snail"])
            if etype == "snail":
                rect = pygame.Rect(0, 0, snail_w, snail_h)
                rect.midbottom = (randint(900, 1100), GROUND_Y)
            else:
                rect = pygame.Rect(0, 0, fly_w, fly_h)
                rect.midbottom = (randint(900, 1100), 210)
            self.obstacle_list.append(rect)
            self.spawned_count += 1
            gap_ms = max(860, int(randint(1350, 2150) / self.speed_mult))
            self.spawn_timer = max(50, int(gap_ms / 1000 * FPS))
 
        # Build raw state
        dist_x, obs_cy = self._nearest_obstacle()
        raw_state = (dist_x, obs_cy, self.player_bottom,
                     self.player_gravity, self.speed_mult)
 
        # Agent picks action
        action = self.qtable.get_action(raw_state, self.epsilon)
        jumped = False
        if action == 1 and self.player_bottom >= GROUND_Y:
            self.player_gravity = -20.0
            jumped = True
 
        # Physics
        self.player_gravity += 1.0
        self.player_bottom  += self.player_gravity
        if self.player_bottom >= GROUND_Y:
            self.player_bottom  = float(GROUND_Y)
            self.player_gravity = 0.0
 
        # Walk animation
        if self.player_bottom >= GROUND_Y:
            self.player_index = (self.player_index + 0.1) % 2.0
 
        # Move & clean obstacles
        speed = int(5 * self.speed_mult)
        for ob in self.obstacle_list:
            ob.x -= speed
        self.obstacle_list = [ob for ob in self.obstacle_list if ob.x > -120]
 
        # Collision (player logical rect)
        player_rect = pygame.Rect(10, int(self.player_bottom) - 67, 66, 67)
        survived    = all(not player_rect.colliderect(ob)
                          for ob in self.obstacle_list)
 
        # Dodge reward
        dodge_reward = 0.0

        for ob in self.obstacle_list:
            oid = id(ob)

            if ob.right < player_rect.left and oid not in self.dodged_ids:
                dodge_reward += 2.0
                self.dodges += 1
                self.dodged_ids.add(oid)
 
        # Total reward
        reward = self._compute_reward(survived, jumped, raw_state) + dodge_reward
 
        # Next state
        dist_x2, obs_cy2 = self._nearest_obstacle()
        raw_next = (dist_x2, obs_cy2, self.player_bottom,
                    self.player_gravity, self.speed_mult)
 
        # Q-update: credit the action that produced this exact transition.
        self.qtable.update(raw_state, action, reward, raw_next, not survived)

        # Death handling
        if not survived:
            self.alive = False
            self._end_episode()
            return False
        return True
 
    # ── Helpers ───────────────────────────────────────────────────────────────
 
    def _nearest_obstacle(self):
        px     = 80
        ahead  = [ob for ob in self.obstacle_list if ob.right > px]
        if not ahead:
            return 800.0, float(GROUND_Y)
        ob = min(ahead, key=lambda o: o.left)
        return float(ob.left - px), float(ob.centery)
 
    def _compute_reward(self, survived, jumped, raw_state):
        """
        Reward shaping:
          +1.0    each frame alive
          +2.0    obstacle dodged  (added in step())
          -200.0  death
          -4..10   wasteful, late, or wrong jumps
          +10.0   useful jump timing for ground obstacles
        """
        if not survived:
            return -200.0

        dist_x, obs_y, player_bottom, _, speed_mult = raw_state
        grounded = player_bottom >= GROUND_Y - 1
        flying_obstacle = obs_y < 230
        ground_obstacle = not flying_obstacle
        reward = 1.0

        if jumped and not grounded:
            reward -= 4.0

        if jumped and flying_obstacle:
            reward -= 10.0
        elif jumped and ground_obstacle:
            # Faster speeds need earlier jumps.
            sweet_min = max(75, 150 - int((speed_mult - 1.0) * 35))
            sweet_max = max(145, 245 - int((speed_mult - 1.0) * 25))
            if sweet_min <= dist_x <= sweet_max:
                reward += 10.0
            elif dist_x > sweet_max:
                reward -= 4.0
            elif dist_x < sweet_min:
                reward -= 2.0

        if not jumped and ground_obstacle and 0 <= dist_x <= 105 and grounded:
            reward -= 8.0

        return reward
 
    def _end_episode(self):
        self.episodes  += 1
        self.last_score = self.score
        if self.score > self.best_score:
            self.best_score = self.score
 
        self.score_hist.append(self.score)
        if len(self.score_hist) > 50:
            self.score_hist.pop(0)
 
        # Decay exploration rate. The next run begins only after evolution.
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def fitness(self):
        recent = self.score_hist[-10:]
        recent_avg = sum(recent) / len(recent) if recent else self.last_score
        return self.best_score * 100.0 + recent_avg * 10.0 + self.score

    def promote_champion(self, generation):
        self.generation = generation
        self.parent_id = self.id
        self.is_champion = True
        self.epsilon_min = min(self.epsilon_min, 0.01)
        self.epsilon = max(self.epsilon_min, min(self.epsilon, 0.12))
        self._init_game()

    def become_offspring(self, parent, generation, explore_rank):
        self.generation = generation
        self.parent_id = parent.id
        self.is_champion = False
        self.episodes = 0
        self.best_score = 0
        self.last_score = 0
        self.score_hist = []
        self.dodges = 0

        base_epsilon = parent.epsilon + (rnd() - 0.5) * 0.18
        explorer_boost = 0.05 + explore_rank * 0.60
        self.epsilon = max(0.02, min(0.95, base_epsilon + explorer_boost))
        self.epsilon_min = 0.03 + explore_rank * 0.12
        self.epsilon_decay = max(0.9985, min(0.9998, parent.epsilon_decay + (rnd() - 0.5) * 0.0008))
        self._init_game()
