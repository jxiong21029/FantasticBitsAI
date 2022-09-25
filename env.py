import warnings

import gym
import numpy as np

from engine import Bludger, Point, Snaffle, Wizard, engine_step


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

        self.observation_spaces = {
            agent_id: gym.spaces.Dict(
                {
                    "global": gym.spaces.Box(0, 1, shape=(3,)),
                    "entities": gym.spaces.Sequence(gym.spaces.Box(0, 1, shape=(9,))),
                }
            )
            for agent_id in ("wizard_0", "wizard_1")
        }
        self.action_spaces = {
            agent_id: gym.spaces.Dict(
                {
                    "move": gym.spaces.Box(-1, 1, shape=(2,)),
                    "throw": gym.spaces.Box(-1, 1, shape=(2,)),
                }
            )
            for agent_id in ("wizard_0", "wizard_1")
        }

    def get_obs(self):
        ret = {}
        for i in range(2):
            global_obs = np.array([self.t / 200, self.score[0] / 7, self.score[1] / 7])
            entity_obs = [
                np.array(
                    [
                        (self.agents[i].x - 8000) / 8000,
                        (self.agents[i].y - 3750) / 8000,
                        self.agents[i].vx / 500,
                        self.agents[i].vy / 500,
                    ]
                    + [1, 0, 0, 0, 0]
                ),
                np.array(
                    [
                        (self.agents[1 - i].x - self.agents[i].x) / 8000,
                        (self.agents[1 - i].y - self.agents[i].y) / 8000,
                        self.agents[1 - i].vx / 500,
                        self.agents[1 - i].vy / 500,
                    ]
                    + [0, 1, 0, 0, 0]
                ),
            ]
            for entity in self.opponents + self.snaffles + self.bludgers:
                if isinstance(entity, Wizard):
                    class_embedding = [0, 0, 1, 0, 0]
                elif isinstance(entity, Snaffle):
                    class_embedding = [0, 0, 0, 1, 0]
                else:
                    class_embedding = [0, 0, 0, 0, 1]

                entity_obs.append(
                    np.array(
                        [
                            (entity.x - self.agents[i].x) / 8000,
                            (entity.y - self.agents[i].y) / 8000,
                            entity.vx / 500,
                            entity.vy / 500,
                        ]
                        + class_embedding
                    )
                )
            ret[f"wizard_{i}"] = {"global": global_obs, "entities": entity_obs}
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

        rewards = {"wizard_0": 0, "wizard_1": 1}

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
            rewards["wizard_0"] += (total_snaffle_dist - new_total_dist) / 16000
            rewards["wizard_1"] += (total_snaffle_dist - new_total_dist) / 16000

        for team, snaffle in scored_goals:
            self.snaffles.remove(snaffle)
            self.score[team - 1] += 1
            if team == 1:
                closer_wizard = min(self.agents, key=lambda w: w.distance2(snaffle))
                if closer_wizard is self.agents[0]:
                    rewards["wizard_0"] += 10
                    rewards["wizard_1"] += 1
                else:
                    rewards["wizard_0"] += 1
                    rewards["wizard_1"] += 10

        done = len(self.snaffles) == 0 or self.t == 200

        return self.get_obs(), rewards, done
