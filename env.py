import warnings

import numpy as np

from engine import Bludger, Point, Snaffle, Wizard, engine_step


DIST_NORM = 8000
VEL_NORM = 1000


class FantasticBits:
    def __init__(self, shape_snaffling=False, seed=None):
        self.rng = np.random.default_rng(seed)
        self.shape_snaffling = shape_snaffling

        self.t = 0
        self.score = [0, 0]

        self.snaffles: list[Snaffle] = []
        self.bludgers: list[Bludger] = []
        self.agents: list[Wizard] = []
        self.opponents: list[Wizard] = []

    def get_obs(self):
        ret = {"global": np.array(
            [
                self.t / 200,
                self.score[0] / 7,
                self.score[1] / 7,
            ]
        )}
        for i, wizard in enumerate(self.agents + self.opponents):
            ret[f"wizard{i}"] = np.array([
                (wizard.x - 8000) / DIST_NORM,
                (wizard.y - 3750) / DIST_NORM,
                wizard.vx / VEL_NORM,
                wizard.vy / VEL_NORM,
                0 if i < 2 else 1,  # 0 if teammate, 1 if opponent
                1 if wizard.grab_cd == 2 else 0,  # throw available
            ])

        for i, snaffle in enumerate(self.snaffles):
            ret[f"snaffle{i}"] = np.array([
                (snaffle.x - 8000) / DIST_NORM,
                (snaffle.y - 3750) / DIST_NORM,
                snaffle.vx / VEL_NORM,
                snaffle.vy / VEL_NORM,
            ])

        for i, bludger in enumerate(self.bludgers):
            ret[f"bludger{i}"] = np.array([
                (bludger.x - 8000) / DIST_NORM,
                (bludger.y - 3750) / DIST_NORM,
                bludger.vx / VEL_NORM,
                bludger.vy / VEL_NORM,
                (bludger.last_target.x - 8000) / DIST_NORM,
                (bludger.last_target.y - 3750) / DIST_NORM,
                (bludger.current_target.x - 8000) / DIST_NORM,
                (bludger.current_target.y - 3750) / DIST_NORM,
            ])

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
        return self.get_obs()

    def step(self, actions):
        assert sorted(list(actions.keys())) == ["wizard_0", "wizard_1"]

        rewards = [0, 0]

        for i, agent in enumerate(self.agents):
            action = actions[f"wizard_{i}"]
            if agent.grab_cd == 2 and "throw" in action.keys():
                held_snaffle = min(self.snaffles, key=lambda s: s.distance2(agent))
                held_snaffle.yeet(
                    agent.x + action["throw"][0], agent.y + action["throw"][1]
                )
            elif "move" in actions["wizard_0"].keys():
                agent.thrust(agent.x + action["move"][0], agent.y + action["move"][1])
            else:
                warnings.warn(f"agent {i} idling")

        for opponent in self.opponents:
            nearest_snaffle = min(self.snaffles, key=lambda s: s.distance2(opponent))
            if opponent.grab_cd == 2:
                nearest_snaffle.yeet(0, 3750)
            else:
                opponent.thrust(nearest_snaffle.x, nearest_snaffle.y)

        for bludger in self.bludgers:
            bludger.bludge(self.agents + self.opponents)

        total_snaffle_dist = sum(s.distance(Point(16000, 3750)) for s in self.snaffles)

        scored_goals = engine_step(
            self.agents + self.opponents + self.snaffles + self.bludgers
        )

        new_total_dist = sum(s.distance(Point(16000, 3750)) for s in self.snaffles)

        if self.shape_snaffling:
            rewards[0] += (total_snaffle_dist - new_total_dist) / 10000
            rewards[1] += (total_snaffle_dist - new_total_dist) / 10000

        for team, snaffle in scored_goals:
            self.snaffles.remove(snaffle)
            self.score[team - 1] += 1
            if team == 1:
                closer_wizard = min(self.agents, key=lambda w: w.distance2(snaffle))
                if closer_wizard is self.agents[0]:
                    rewards[0] += 10
                    rewards[1] += 1
                else:
                    rewards[0] += 1
                    rewards[1] += 10

        done = len(self.snaffles) == 0 or self.t == 200

        return self.get_obs(), rewards, done
