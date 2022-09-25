import math
import warnings
from dataclasses import dataclass
from enum import Enum

# collision handling logic adapted from
# github.com/dreignier/fantastic-bits/blob/master/fantastic-bits.cpp


class Boundary(Enum):
    LEFT = 1
    TOP = 2
    RIGHT = 3
    BOTTOM = 4
    GOAL = 5


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance2(self, other: "Point"):
        return (self.x - other.x) * (self.x - other.x) + (self.y - other.y) * (
            self.y - other.y
        )

    def distance(self, other: "Point"):
        return math.sqrt(self.distance2(other))

    def closest(self, a: "Point", b: "Point"):
        da = b.y - a.y
        db = a.x - b.x
        c1 = da * a.x + db * a.y
        c2 = -db * self.x + da * self.y
        det = da * da + db * db

        if det != 0:
            cx = (da * c1 - db * c2) / det
            cy = (da * c2 + db * c1) / det
        else:
            # The point is already on the line
            cx = self.x
            cy = self.y

        return Point(cx, cy)


class Entity(Point):
    def __init__(self, x, y, vx, vy, rad, mass, friction):
        Point.__init__(self, x, y)
        self.vx = vx
        self.vy = vy
        self.rad = rad
        self.mass = mass
        self.friction = friction

    def move(self, t):
        self.x += self.vx * t
        self.y += self.vy * t

    def end(self):
        self.x = round(self.x)
        self.y = round(self.y)
        self.vx = round(self.vx * self.friction)
        self.vy = round(self.vy * self.friction)

    def map_collisions(self):
        for pole in POLES:
            if col := self.collision(pole):
                yield col
                break

        endx = self.x + self.vx
        endy = self.y + self.vy
        if endx < self.rad:
            yield Collision(self, Boundary.LEFT, (self.rad - self.x) / self.vx)
        elif endx > 16000 - self.rad:
            yield Collision(self, Boundary.RIGHT, (16000 - self.rad - self.x) / self.vx)

        if endy < self.rad:
            yield Collision(self, Boundary.TOP, (self.rad - self.y) / self.vy)
        elif endy > 7500 - self.rad:
            yield Collision(self, Boundary.BOTTOM, (7500 - self.rad - self.y) / self.vy)

    def collision(self, other):
        # Sum of the radii squared
        sr = (self.rad + other.rad) * (self.rad + other.rad)

        # Optimisation. Objects with the same speed will never collide
        if self.vx == other.vx and self.vy == other.vy:
            return None

        # We place ourselves in the reference frame of other. other is therefore
        # stationary and is at (0,0)
        x = self.x - other.x
        y = self.y - other.y
        myp = Point(x, y)
        vx = self.vx - other.vx
        vy = self.vy - other.vy
        up = Point(0, 0)

        # We look for the closest point to u (which is in (0,0)) on the line described
        # by our speed vector
        p = up.closest(myp, Point(x + vx, y + vy))

        # Square of the distance between u and the closest point to u on the line
        # described by our speed vector
        pdist = up.distance2(p)

        # Square of the distance between us and that point
        mypdist = myp.distance2(p)

        # If the distance between u and this line is less than the sum of the radii,
        # there might be a collision
        if pdist < sr:
            # Our speed on the line
            length = math.sqrt(vx * vx + vy * vy)

            # We move along the line to find the point of impact
            backdist = math.sqrt(sr - pdist)
            p.x = p.x - backdist * (vx / length)
            p.y = p.y - backdist * (vy / length)

            # If the point is now further away it means we are not going the right way,
            # therefore the collision won't happen
            if myp.distance2(p) > mypdist:
                return None

            pdist = p.distance(myp)

            # The point of impact is further than what we can travel in one turn
            if pdist > length:
                return None

            # Time needed to reach the impact point
            t = pdist / length

            return Collision(self, other, t)

    def bounce(self, other):
        if isinstance(other, Boundary):
            if self.x <= self.rad:
                self.vx = abs(self.vx)
            elif self.x >= 16000 - self.rad:
                self.vx = -abs(self.vx)

            if self.y <= self.rad:
                self.vy = abs(self.vy)
            elif self.y >= 7500 - self.rad:
                self.vy = -abs(self.vy)
            return

        m1 = self.mass
        m2 = other.mass

        mcoeff = (m1 + m2) / (m1 * m2)

        nx = self.x - other.x
        ny = self.y - other.y

        # Square of the distance between the 2 entities.
        nxnysquare = nx * nx + ny * ny

        dvx = self.vx - other.vx
        dvy = self.vy - other.vy

        # fx and fy are the components of the impact vector. product is just there for
        # optimisation purposes
        product = nx * dvx + ny * dvy
        fx = (nx * product) / (nxnysquare * mcoeff)
        fy = (ny * product) / (nxnysquare * mcoeff)

        if self.mass is not None:
            self.vx -= fx / m1
            self.vy -= fy / m1
        if other.mass is not None:
            other.vx += fx / m2
            other.vy += fy / m2

        # If the norm of the impact vector is less than 100, we normalize it to 100
        impulse = math.sqrt(fx * fx + fy * fy)
        if impulse < 100:
            fx = fx * 100 / impulse
            fy = fy * 100 / impulse

        if self.mass is not None:
            self.vx -= fx / m1
            self.vy -= fy / m1
        if other.mass is not None:
            other.vx += fx / m2
            other.vy += fy / m2


class Wizard(Entity):
    def __init__(self, x, y, vx=0, vy=0, rad=400, mass=1, friction=0.75, grab_cd=0):
        super().__init__(x, y, vx, vy, rad, mass, friction)
        self.grab_cd = grab_cd

    def thrust(self, x, y, power=150):
        dx = x - self.x
        dy = y - self.y

        norm = math.sqrt(dx * dx + dy * dy)
        if norm == 0:
            return
        dx /= norm
        dy /= norm

        self.vx += dx * power / self.mass
        self.vy += dy * power / self.mass

    def collision(self, other):
        if isinstance(other, Snaffle):
            if self.grab_cd > 0:
                return None
            old_rad = other.rad
            try:
                other.rad = 0
                return super().collision(other)
            finally:
                other.rad = old_rad
        return super().collision(other)

    def bounce(self, other):
        if isinstance(other, (Snaffle, Bludger)):
            other.bounce(self)
        else:
            super().bounce(other)

    def end(self):
        if self.grab_cd > 0:
            self.grab_cd -= 1
        super().end()


class Snaffle(Entity):
    def __init__(
        self, x, y, vx=0, vy=0, rad=150, mass=0.5, friction=0.75, grabbed=False
    ):
        super().__init__(x, y, vx, vy, rad, mass, friction)
        self.grabbed = grabbed

    def yeet(self, x, y, power=500):
        dx = x - self.x
        dy = y - self.y

        norm = math.sqrt(dx * dx + dy * dy)
        if norm == 0:
            return
        dx /= norm
        dy /= norm

        self.vx += dx * power / self.mass
        self.vy += dy * power / self.mass

    def map_collisions(self):
        endx = self.x + self.vx
        endy = self.y + self.vy
        for pole in POLES:
            if col := self.collision(pole):
                yield col
                break
        else:
            if endx < 0 and (1500 <= endy <= 6000):
                yield Collision(self, Boundary.GOAL, self.x / self.vx)
                return
            elif endx > 16000 and (1500 <= endy <= 6000):
                yield Collision(self, Boundary.GOAL, (16000 - self.x) / self.vx)
                return

        if endx < self.rad and not (1500 <= endy <= 6000):
            yield Collision(self, Boundary.LEFT, (self.rad - self.x) / self.vx)
        elif endx > 16000 - self.rad and not (1500 <= endy <= 6000):
            yield Collision(self, Boundary.RIGHT, (16000 - self.rad - self.x) / self.vx)

        if endy < self.rad:
            yield Collision(self, Boundary.TOP, (self.rad - self.y) / self.vy)
        elif endy > 7500 - self.rad:
            yield Collision(self, Boundary.BOTTOM, (7500 - self.rad - self.y) / self.vy)

    def collision(self, other):
        if isinstance(other, Wizard):
            return other.collision(self)
        return super().collision(other)

    def bounce(self, other):
        if isinstance(other, Wizard):
            if self.collision(other):
                self.grabbed = 1
                other.grab_cd = 3

                self.x = other.x
                self.y = other.y
                self.vx = other.vx
                self.vy = other.vy

        else:
            super().bounce(other)

    def end(self):
        if self.grabbed == 2:
            self.grabbed = 0
        else:
            self.grabbed += 1
        super().end()


class Bludger(Entity):
    def __init__(
        self, x, y, vx=0, vy=0, rad=200, mass=8, friction=0.9, last_target=None
    ):
        super().__init__(x, y, vx, vy, rad, mass, friction)
        self.last_target = last_target
        self.current_target = None

    def bludge(self, all_wizards: list[Wizard]):
        self.current_target = min(
            [wizard for wizard in all_wizards if wizard is not self.last_target],
            key=lambda w: self.distance2(w),
        )
        dx = self.current_target.x - self.x
        dy = self.current_target.y - self.y

        norm = math.sqrt(dx * dx + dy * dy)
        if norm == 0:
            return
        dx /= norm
        dy /= norm

        self.vx += dx * 1000 / self.mass
        self.vy += dy * 1000 / self.mass

    def bounce(self, other):
        if isinstance(other, Wizard):
            self.last_target = other
        super().bounce(other)


POLES = [
    Entity(0, 1750, 0, 0, 300, 99999, 0),
    Entity(0, 5750, 0, 0, 300, 99999, 0),
    Entity(16000, 1750, 0, 0, 300, 99999, 0),
    Entity(16000, 5750, 0, 0, 300, 99999, 0),
]


@dataclass
class Collision:
    a: Entity
    b: Entity
    t: float


def engine_step(entities):
    scored_goals = []

    # This tracks the time during the turn. The goal is to reach 1.0
    t = 0.0
    _iters = 0

    while t < 1.0:
        _iters += 1
        if _iters >= 999:
            warnings.warn("collisions broke")
            break
        first_col = None

        # We look for all the collisions that are going to occur during the turn
        for i, entity in enumerate(entities):
            for col in entity.map_collisions():
                if col.t + t < 1.0 and (first_col is None or col.t < first_col.t):
                    first_col = col

            # Collision with another pod?
            for j in range(i + 1, len(entities)):
                col = entity.collision(entities[j])

                # If the collision occurs earlier than the current one, keep it
                if (
                    col is not None
                    and col.t + t < 1.0
                    and (first_col is None or col.t < first_col.t)
                ):
                    first_col = col

        if first_col is None:
            # No collision, we can move the entities until the end of the turn
            for entity in entities:
                entity.move(1.0 - t)

            # End of the turn
            t = 1.0
        else:
            if first_col.b == Boundary.GOAL:
                snaffle = first_col.a

                entities.remove(snaffle)

                if snaffle.x + snaffle.vx < 0:
                    scored_goals.append((2, snaffle))  # returns which team scored
                else:
                    scored_goals.append((1, snaffle))

                continue

            # Move the entities to reach the time `t` of the collision
            for entity in entities:
                entity.move(first_col.t)

            # Play out the collision
            first_col.a.bounce(first_col.b)

            t += first_col.t

    for entity in entities + POLES:
        entity.end()

    return scored_goals
