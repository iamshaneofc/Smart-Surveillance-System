Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, p2, q2):
        return True
    if o3 == 0 and _on_segment(q1, q2, p1):
        return True
    if o4 == 0 and _on_segment(q1, q2, p2):
        return True
    return False


def is_simple_polygon(polygon: list[Point]) -> bool:
    """True when a closed polygon has >= 3 vertices and no non-adjacent edge crossings."""
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        a1 = polygon[i]
        a2 = polygon[(i + 1) % n]
        if a1 == a2:
            return False
        for j in range(i + 1, n):
            if j == i or (i + 1) % n == j or (j + 1) % n == i:
                continue
            b1 = polygon[j]
            b2 = polygon[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True


def crossing_direction(prev: Point, curr: Point, line_a: Point, line_b: Point) -> str | None:
    """Direction relative to the directed line a->b: left_to_right / right_to_left."""
    if not segments_intersect(prev, curr, line_a, line_b):
        return None
    side_prev = _orient(line_a, line_b, prev)
    side_curr = _orient(line_a, line_b, curr)
    if side_prev == 0 or side_curr == 0 or side_prev == side_curr:
        return None
    return "left_to_right" if side_prev > 0 else "right_to_left"


def bbox_center(x: float, y: float, w: float, h: float) -> Point:
    return (x + w / 2.0, y + h / 2.0)


ANCHOR_POINTS = ("center", "top_center", "bottom_center")


def anchor_point(x: float, y: float, w: float, h: float, anchor: str = "center") -> Point:
    """Anchor used for point-in-zone tests. Default and recommended: bbox center."""
    if anchor == "top_center":
        return (x + w / 2.0, y)
    if anchor == "bottom_center":
        return (x + w / 2.0, y + h)
    return bbox_center(x, y, w, h)
