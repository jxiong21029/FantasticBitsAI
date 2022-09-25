import math
from dataclasses import dataclass

# collision handling logic adapted from
# github.com/dreignier/fantastic-bits/blob/master/fantastic-bits.cpp


@dataclass
class Point:
    x: float
    y: float

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


@dataclass
class Entity(Point):
    vx: float
    vy: float
    rad: float
    mass: float
    friction: float

    def move(self, t):
        self.x += self.vx * t
        self.y += self.vy * t

    def end(self):
        self.x = round(self.x)
        self.y = round(self.y)
        self.vx = round(self.vx * self.friction)
        self.vy = round(self.vy * self.friction)

    def collision(self, other: "Entity"):
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
        m1 = self.mass
        m2 = other.mass

        mcoeff = (m1 + m2) / (m1 * m2) if m1 is not None and m2 is not None else 1

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


@dataclass
class Collision:
    a: Entity
    b: Entity
    t: float


def play(entities):
    # This tracks the time during the turn. The goal is to reach 1.0
    t = 0.0

    while t < 1.0:
        first_col = None

        # We look for all the collisions that are going to occur during the turn
        for i in range(len(entities)):
            # Collision with another pod?
            for j in range(i + 1, len(entities)):
                col = entities[i].collision(entities[j])

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
            print("> wham! <")
            # Move the entities to reach the time `t` of the collision
            for entity in entities:
                entity.move(first_col.t)

            # Play out the collision
            first_col.a.bounce(first_col.b)

            t += first_col.t

    for entity in entities:
        entity.end()


def main():
    entities = [
        Entity(4383, 2231, 4183 - 4383, 2195 - 2231, 150, 0.5, 0.75),
        Entity(5061, 2132, 4600 - 5061, 2071 - 2132, 150, 0.5, 0.75),
    ]
    for t in range(4):
        print(f"######## t={t} ########")
        for i, entity in enumerate(entities):
            print(f"({entity.x}, {entity.y}) w/ velocity ({entity.vx}, {entity.vy})")

        play(entities)


if __name__ == "__main__":
    main()
