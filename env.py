import sys
import time
import warnings

import numpy as np
import pygame

from engine import POLES, Bludger, Point, Snaffle, Wizard, engine_step

DIST_NORM = 8000
VEL_NORM = 1000
SZ_GLOBAL = 3
SZ_WIZARD = 6
SZ_SNAFFLE = 4
SZ_BLUDGER = 8

SCALE = 20
MARGIN = 25


class FantasticBits:
    def __init__(self, shape_snaffling=False, render=False, seed=None):
        self.rng = np.random.default_rng(seed)
        self.shape_snaffling = shape_snaffling
        self.render = render

        self.t = 0
        self.score = [0, 0]

        self.snaffles: list[Snaffle] = []
        self.bludgers: list[Bludger] = []
        self.agents: list[Wizard] = []
        self.opponents: list[Wizard] = []

        if render:
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

        # for opponent in self.opponents:
        #     nearest_snaffle = min(self.snaffles, key=lambda s: s.distance2(opponent))
        #     if opponent.grab_cd == 2:
        #         nearest_snaffle.yeet(0, 3750)
        #     else:
        #         opponent.thrust(nearest_snaffle.x, nearest_snaffle.y)

        # for bludger in self.bludgers:
        #     bludger.bludge(self.agents + self.opponents)

        total_snaffle_dist = sum(s.distance(Point(16000, 3750)) for s in self.snaffles)

        scored_goals = engine_step(
            self.agents + self.opponents + self.snaffles + self.bludgers
        )

        new_total_dist = sum(s.distance(Point(16000, 3750)) for s in self.snaffles)

        # NOTE: hardcoded discount factor
        if self.shape_snaffling:
            rewards[0] += (total_snaffle_dist - new_total_dist) / 10000
            rewards[1] += (total_snaffle_dist - new_total_dist) / 10000

        for team, snaffle in scored_goals:
            self.snaffles.remove(snaffle)
            self.score[team - 1] += 1
            if team == 1:
                # TODO: reward the agent which last affected the snaffle
                closer_wizard = min(self.agents, key=lambda w: w.distance2(snaffle))
                if closer_wizard is self.agents[0]:
                    rewards[0] += 10
                    rewards[1] += 1
                else:
                    rewards[0] += 1
                    rewards[1] += 10

        self.t += 1
        done = len(self.snaffles) == 0 or self.t == 200

        self.render_frame()
        return self.get_obs(), rewards, done

    def render_frame(self):
        if not self.render:
            return

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
