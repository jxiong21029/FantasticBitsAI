import sys
import time

import numpy as np
import pygame

from engine import POLES
from env import Bludger, FantasticBits, Snaffle, Wizard

SCALE = 20
MARGIN = 25

screen = pygame.display.set_mode(
    (16000 / SCALE + MARGIN * 2, 7500 / SCALE + MARGIN * 2)
)

env = FantasticBits()
env.reset()

for _ in range(200):
    for event in pygame.event.get():
        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
            pygame.quit()
            pygame.display.quit()
            sys.exit()

    env.step(
        {
            "id": np.array([0, 0], dtype=np.int64),
            "target": np.random.randn(2, 2),
        }
    )

    screen.fill((0, 0, 0))
    pygame.draw.line(
        screen, (150, 150, 0), (MARGIN, MARGIN), (MARGIN + 16000 / SCALE, MARGIN)
    )
    pygame.draw.line(
        screen,
        (150, 150, 0),
        (MARGIN + 16000 / SCALE, MARGIN),
        (MARGIN + 16000 / SCALE, MARGIN + 7500 / SCALE),
    )
    pygame.draw.line(
        screen,
        (150, 150, 0),
        (MARGIN + 16000 / SCALE, MARGIN + 7500 / SCALE),
        (MARGIN, MARGIN + 7500 / SCALE),
    )
    pygame.draw.line(
        screen, (150, 150, 0), (MARGIN, MARGIN + 7500 / SCALE), (MARGIN, MARGIN)
    )
    for entity in env.agents + env.opponents + env.snaffles + env.bludgers + POLES:
        if isinstance(entity, Wizard) and entity in env.agents:
            color = (255, 0, 0)
        elif isinstance(entity, Wizard):
            color = (255, 100, 100)
        elif isinstance(entity, Snaffle):
            color = (255, 255, 0)
        elif isinstance(entity, Bludger):
            if entity.current_target is not None:
                pygame.draw.line(
                    screen,
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
            screen,
            color,
            (entity.x / SCALE + MARGIN, entity.y / SCALE + MARGIN),
            entity.rad / SCALE,
        )
        pygame.draw.line(
            screen,
            color,
            (entity.x / SCALE + MARGIN, entity.y / SCALE + MARGIN),
            (
                (entity.x + entity.vx) / SCALE + MARGIN,
                (entity.y + entity.vy) / SCALE + MARGIN,
            ),
        )

    pygame.display.flip()

    time.sleep(1 / 3)
