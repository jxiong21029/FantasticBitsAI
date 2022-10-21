import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine import POLES, Bludger, Point, Snaffle, Wizard, engine_step
from utils import Logger

DIST_NORM = 8000
VEL_NORM = 1000
SZ_GLOBAL = 3
SZ_WIZARD = 6
SZ_SNAFFLE = 4
SZ_BLUDGER = 8

SCALE = 20
MARGIN = 25


@dataclass
class FantasticBitsConfig:
    bludgers_enabled: bool = True
    opponents_enabled: bool = True
    reward_win: float = 0.0
    reward_loss: float = 0.0
    reward_own_goal: float = 1.0
    reward_teammate_goal: float = 1.0
    reward_opponent_goal: float = 0.0
    reward_shaping_snaffle_goal_dist: bool = False
    reward_gamma: float = 0.99
    render: bool = False
    seed: Optional[int] = None
    logger: Optional[Logger] = None


class FantasticBits:
    def __init__(self, config: FantasticBitsConfig = None, **kwargs):
        if config is None:
            config = FantasticBitsConfig(**kwargs)

        self.rng = np.random.default_rng(seed=config.seed)
        self.logger = config.logger
        self.render = config.render

        self.bludgers_enabled = config.bludgers_enabled
        self.opponents_enabled = config.opponents_enabled

        self.reward_win = config.reward_win
        self.reward_loss = config.reward_loss
        self.reward_own_goal = config.reward_own_goal
        self.reward_teammate_goal = config.reward_teammate_goal
        self.reward_opponent_goal = config.reward_opponent_goal
        self.reward_shaping_snaffle_goal_dist = config.reward_shaping_snaffle_goal_dist
        self.reward_gamma = config.reward_gamma

        self.t = 0
        self.score = [0, 0]
        self.episode_rewards = np.zeros(2, dtype=np.float32)

        self.snaffles: list[Snaffle] = []
        self.bludgers: list[Bludger] = []
        self.agents: list[Wizard] = []
        self.opponents: list[Wizard] = []

        if config.render:
            import pygame

            self.screen = pygame.display.set_mode(
                (16000 / SCALE + MARGIN * 2, 7500 / SCALE + MARGIN * 2)
            )

    def get_obs(self):
        ret = {
            "global": np.array(
                [
                    self.t / 200,
                    self.score[0] / 7,
                    self.score[1] / 7,
                ],
                dtype=np.float32,
            )
        }
        for i, wizard in enumerate(self.agents + self.opponents):
            ret[f"wizard{i}"] = np.array(
                [
                    (wizard.x - 8000) / DIST_NORM,
                    (wizard.y - 3750) / DIST_NORM,
                    wizard.vx / VEL_NORM,
                    wizard.vy / VEL_NORM,
                    0 if i < 2 else 1,  # 0 if teammate, 1 if opponent
                    1 if wizard.grab_cd == 2 else 0,  # throw available
                ],
                dtype=np.float32,
            )

        for i, snaffle in enumerate(self.snaffles):
            ret[f"snaffle{i}"] = np.array(
                [
                    (snaffle.x - 8000) / DIST_NORM,
                    (snaffle.y - 3750) / DIST_NORM,
                    snaffle.vx / VEL_NORM,
                    snaffle.vy / VEL_NORM,
                ],
                dtype=np.float32,
            )

        for i, bludger in enumerate(self.bludgers):
            ret[f"bludger{i}"] = np.array(
                [
                    (bludger.x - 8000) / DIST_NORM,
                    (bludger.y - 3750) / DIST_NORM,
                    bludger.vx / VEL_NORM,
                    bludger.vy / VEL_NORM,
                    (bludger.last_target.x - 8000) / DIST_NORM
                    if bludger.last_target is not None
                    else 0,
                    (bludger.last_target.y - 3750) / DIST_NORM
                    if bludger.last_target is not None
                    else 0,
                    (bludger.current_target.x - 8000) / DIST_NORM
                    if bludger.current_target is not None
                    else 0,
                    (bludger.current_target.y - 3750) / DIST_NORM
                    if bludger.current_target is not None
                    else 0,
                ],
                dtype=np.float32,
            )

        return ret

    def reset(self):
        rng = self.rng

        self.t = 0
        self.score = [0, 0]
        self.episode_rewards = np.zeros(2, dtype=np.float32)
        if rng.random() < 0.5:
            tot_snaffles = 5
        else:
            tot_snaffles = 7

        self.snaffles = [Snaffle(8000, 3750)]
        self.bludgers = [Bludger(7450, 3750), Bludger(8550, 3750)]
        self.agents = [Wizard(1000, 2250), Wizard(1000, 5250)]
        self.opponents = [Wizard(15000, 2250), Wizard(15000, 5250)]

        other_things = self.bludgers + self.agents + self.opponents

        while len(self.snaffles) < tot_snaffles:
            newx = rng.integers(2000, 7750)
            newy = rng.integers(500, 7000)

            if any(
                Point(newx, newy).distance(other) < 350 + other.rad
                for other in other_things + self.snaffles
            ):
                continue

            self.snaffles.append(Snaffle(newx, newy))
            self.snaffles.append(Snaffle(16000 - newx, 7500 - newy))
        self.render_frame()
        return self.get_obs()

    def step(self, actions):
        # actions: {"id": np.array([0, 1]), "target": np.array([[-.8, .4], [.3, .2]])}

        rewards = np.zeros(2)
        for i, agent in enumerate(self.agents):
            direction = actions["target"][i]
            if actions["id"][i] == 1:  # throw
                if agent.grab_cd != 2:
                    warnings.warn(f"agent {i} attempted throw without snaffle")
                    continue

                for snaffle in self.snaffles:
                    if (agent.x, agent.y) == (snaffle.x, snaffle.y):
                        snaffle.yeet(agent.x + direction[0], agent.y + direction[1])
                        break
            else:
                agent.thrust(agent.x + direction[0], agent.y + direction[1])

        if self.opponents_enabled:
            for opponent in self.opponents:
                nearest_snaffle = min(
                    self.snaffles, key=lambda s: s.distance2(opponent)
                )
                if opponent.grab_cd == 2:
                    nearest_snaffle.yeet(0, 3750)
                else:
                    opponent.thrust(nearest_snaffle.x, nearest_snaffle.y)

        if self.bludgers_enabled:
            for bludger in self.bludgers:
                bludger.bludge(self.agents + self.opponents)

        total_snaffle_dist = sum(s.distance(Point(16000, 3750)) for s in self.snaffles)

        scored_goals = engine_step(
            self.agents + self.opponents + self.snaffles + self.bludgers
        )

        new_total_dist = sum(s.distance(Point(16000, 3750)) for s in self.snaffles)

        if self.reward_shaping_snaffle_goal_dist:
            rewards[0] += (
                self.reward_gamma * total_snaffle_dist - new_total_dist
            ) / 10000
            rewards[1] += (
                self.reward_gamma * total_snaffle_dist - new_total_dist
            ) / 10000

        for team, snaffle in scored_goals:
            self.snaffles.remove(snaffle)
            self.score[team - 1] += 1
            if team == 1:
                for i, wizard in enumerate(self.agents):
                    if snaffle.last_touched == wizard:
                        rewards[i] += self.reward_own_goal
                    else:
                        rewards[i] += self.reward_teammate_goal
            else:
                rewards += self.reward_opponent_goal  # expected to be nonpositive

        self.t += 1
        done = len(self.snaffles) == 0 or self.t == 200

        if done:
            if self.score[0] > self.score[1]:
                rewards += self.reward_win
            else:
                rewards += self.reward_loss
        self.episode_rewards += rewards

        if done and self.logger is not None:
            self.logger.log(
                goals_scored=self.score[0],
                episode_reward=self.episode_rewards.mean(),
                episode_len=self.t,
            )

        self.render_frame()
        return self.get_obs(), rewards, done

    def render_frame(self):
        if not self.render:
            return

        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                pygame.quit()
                pygame.display.quit()
                sys.exit()

        self.screen.fill((0, 0, 0))
        pygame.draw.line(
            self.screen,
            (150, 150, 0),
            (MARGIN, MARGIN),
            (MARGIN + 16000 / SCALE, MARGIN),
        )
        pygame.draw.line(
            self.screen,
            (150, 150, 0),
            (MARGIN + 16000 / SCALE, MARGIN),
            (MARGIN + 16000 / SCALE, MARGIN + 7500 / SCALE),
        )
        pygame.draw.line(
            self.screen,
            (150, 150, 0),
            (MARGIN + 16000 / SCALE, MARGIN + 7500 / SCALE),
            (MARGIN, MARGIN + 7500 / SCALE),
        )
        pygame.draw.line(
            self.screen,
            (150, 150, 0),
            (MARGIN, MARGIN + 7500 / SCALE),
            (MARGIN, MARGIN),
        )
        for entity in (
            self.agents + self.opponents + self.snaffles + self.bludgers + POLES
        ):
            if isinstance(entity, Wizard) and entity in self.agents:
                color = (255, 0, 0)
            elif isinstance(entity, Wizard):
                color = (255, 100, 100)
            elif isinstance(entity, Snaffle):
                color = (255, 255, 0)
            elif isinstance(entity, Bludger):
                if entity.current_target is not None:
                    pygame.draw.line(
                        self.screen,
                        (100, 100, 100),
                        (entity.x / SCALE + MARGIN, entity.y / SCALE + MARGIN),
                        (
                            entity.current_target.x / SCALE + MARGIN,
                            entity.current_target.y / SCALE + MARGIN,
                        ),
                    )
                color = (100, 100, 100)
            else:
                color = (50, 50, 50)
            pygame.draw.circle(
                self.screen,
                color,
                (entity.x / SCALE + MARGIN, entity.y / SCALE + MARGIN),
                entity.rad / SCALE,
            )
            pygame.draw.line(
                self.screen,
                color,
                (entity.x / SCALE + MARGIN, entity.y / SCALE + MARGIN),
                (
                    (entity.x + entity.vx) / SCALE + MARGIN,
                    (entity.y + entity.vy) / SCALE + MARGIN,
                ),
            )

        pygame.display.flip()
