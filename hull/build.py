"""Contracts for incremental 3D convex-hull construction.

Coordinates are right-handed floating-point triples ``(x, y, z)``. A hull is a
plain dictionary::

    {
        "positions": [(x, y, z), ...],
        "faces": [(i, j, k), ...],
    }

Every face winds counter-clockwise when viewed from outside the hull, so the
normal ``cross(position[j] - position[i], position[k] - position[i])`` points
away from the hull interior. Unless a function takes an explicit ``eps``, the
geometric tolerance is ``TOL = 1e-6``.
"""

import math
from collections import Counter

TOL = 1e-6


class HullError(ValueError):
    """Raised when input points or hull topology cannot produce a valid hull."""


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _distance(a, b):
    return _norm(_sub(a, b))


def _face_normal(a, b, c):
    return _cross(_sub(b, a), _sub(c, a))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _as_point(value):
    """Return ``value`` as a finite float triple or raise ``HullError``."""
    try:
        coords = tuple(value)
    except TypeError:
        raise HullError("point must be a sequence of three coordinates")
    if len(coords) != 3:
        raise HullError("point must have exactly three coordinates")
    result = []
    for coord in coords:
        if isinstance(coord, bool) or not isinstance(coord, (int, float)):
            raise HullError("coordinates must be real numbers")
        if not math.isfinite(coord):
            raise HullError("coordinates must be finite")
        result.append(float(coord))
    return tuple(result)


def _coerce_points(points):
    if points is None:
        raise HullError("points must be a sequence")
    try:
        raw = list(points)
    except TypeError:
        raise HullError("points must be a sequence")
    return [_as_point(point) for point in raw]


def _check_eps(eps):
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise HullError("eps must be a positive finite number")
    if not math.isfinite(eps) or eps <= 0.0:
        raise HullError("eps must be a positive finite number")
    return float(eps)


def _check_hull(hull):
    """Validate a hull's structure and return normalized positions/faces.

    Structural problems raise ``HullError``; geometric problems are left to
    the callers that report them.
    """
    if not isinstance(hull, dict) or "positions" not in hull or "faces" not in hull:
        raise HullError("hull must be a mapping with 'positions' and 'faces'")
    try:
        raw_positions = list(hull["positions"])
        raw_faces = list(hull["faces"])
    except TypeError:
        raise HullError("hull positions and faces must be sequences")

    positions = [_as_point(point) for point in raw_positions]

    faces = []
    for raw_face in raw_faces:
        try:
            face = tuple(raw_face)
        except TypeError:
            raise HullError("each face must be a sequence of three indices")
        if len(face) != 3:
            raise HullError("each face must be a triangle")
        for index in face:
            if isinstance(index, bool) or not isinstance(index, int):
                raise HullError("face indices must be integers")
            if index < 0 or index >= len(positions):
                raise HullError("face index out of range")
        faces.append(face)
    return positions, faces


def _signed_distance(a, b, c, point):
    """Signed distance from ``point`` to the oriented plane ``a, b, c``.

    Positive means the point lies on the side the face normal points to.
    """
    normal = _face_normal(a, b, c)
    length = _norm(normal)
    if length == 0.0:
        raise HullError("degenerate face; plane normal is undefined")
    return _dot(normal, _sub(point, a)) / length


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def _edge_problems(faces):
    """Return undirected edges that are not traversed once in each direction."""
    counts = Counter()
    for a, b, c in faces:
        for edge in ((a, b), (b, c), (c, a)):
            counts[edge] += 1

    problems = []
    seen = set()
    for (a, b), _count in counts.items():
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        if a == b:
            problems.append("non-manifold edge (%d, %d): self-loop" % (a, b))
            continue
        forward = counts.get((key[0], key[1]), 0)
        reverse = counts.get((key[1], key[0]), 0)
        if forward != 1 or reverse != 1:
            problems.append(
                "non-manifold edge (%d, %d): %d forward, %d reverse"
                % (key[0], key[1], forward, reverse)
            )
    return problems


def _oriented_outward(a, b, c, positions, inside_index):
    """Return indices ``(a, b, c)`` ordered so the normal points away from
    ``positions[inside_index]``."""
    normal = _face_normal(positions[a], positions[b], positions[c])
    if _dot(normal, _sub(positions[inside_index], positions[a])) > 0.0:
        return (a, c, b)
    return (a, b, c)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dedupe_points(points, eps):
    """Return points merged within Euclidean distance ``eps``.

    Args:
        points: Sequence of coordinate triples ``(x, y, z)``.
        eps: Positive finite merge radius.

    Returns:
        A new list preserving first-occurrence order. A point is dropped when
        its Euclidean distance to an earlier kept point is ``<= eps``. The
        input sequence is not mutated.

    Raises:
        HullError: ``eps <= 0`` or non-finite, malformed/non-finite
            coordinates.
    """
    eps = _check_eps(eps)
    coordinates = _coerce_points(points)

    kept = []
    for point in coordinates:
        merged = False
        for previous in kept:
            if _distance(point, previous) <= eps:
                merged = True
                break
        if not merged:
            kept.append(point)
    return kept


def initial_tetra(points):
    """Choose a non-degenerate tetrahedron from ``points``.

    Returns:
        A hull dictionary with exactly four ``positions`` and four outward
        counter-clockwise ``faces``. The selected points must span a 3D
        volume larger than ``TOL``.

    Raises:
        HullError: Fewer than four distinct points, or every candidate
            tetrahedron is coplanar/degenerate (absolute volume ``<= TOL``).
    """
    coordinates = _coerce_points(points)
    if len(coordinates) < 4:
        raise HullError("at least four distinct points are required")

    p0 = coordinates[0]

    # p1: farthest point from p0.
    i1 = max(
        range(len(coordinates)),
        key=lambda i: _distance(p0, coordinates[i]),
    )
    p1 = coordinates[i1]
    if _distance(p0, p1) <= TOL:
        raise HullError("fewer than four distinct points")

    # p2: farthest point from the line through p0 and p1.
    direction = _sub(p1, p0)
    direction_len = _norm(direction)

    def line_distance(point):
        offset = _sub(point, p0)
        along = _dot(offset, direction) / direction_len
        projection = (
            p0[0] + direction[0] / direction_len * along,
            p0[1] + direction[1] / direction_len * along,
            p0[2] + direction[2] / direction_len * along,
        )
        return _distance(point, projection)

    i2 = max(
        range(len(coordinates)),
        key=lambda i: line_distance(coordinates[i]),
    )
    p2 = coordinates[i2]
    if line_distance(p2) <= TOL:
        raise HullError("fewer than four distinct points")

    # p3: farthest point from the plane through p0, p1 and p2.
    normal = _face_normal(p0, p1, p2)
    normal_len = _norm(normal)
    if normal_len == 0.0:
        raise HullError("fewer than four distinct points")

    i3 = max(
        range(len(coordinates)),
        key=lambda i: abs(_dot(normal, _sub(coordinates[i], p0))) / normal_len,
    )
    p3 = coordinates[i3]

    # Ensure the selected tetrahedron has positive orientation.
    triple = _dot(normal, _sub(p3, p0))
    if abs(triple) / 6.0 <= TOL:
        raise HullError("all candidate tetrahedra are degenerate")
    if triple < 0.0:
        p2, p3 = p3, p2

    positions = [p0, p1, p2, p3]
    # Each face is oriented away from the tetrahedron vertex it does not use.
    faces = [
        _oriented_outward(0, 1, 2, positions, 3),
        _oriented_outward(0, 1, 3, positions, 2),
        _oriented_outward(0, 2, 3, positions, 1),
        _oriented_outward(1, 2, 3, positions, 0),
    ]
    return {"positions": positions, "faces": faces}


def conflict_list(tetra, points):
    """Return indices of ``points`` visible from at least one tetra face.

    A point is visible when its signed distance to a face plane is
    ``>= -TOL``. This deliberately includes points exactly on the plane
    (coplanar points), so a strict ``> 0`` visibility test is incorrect. The
    returned indices refer to ``points``; no indices are excluded, and callers
    normally pass points other than the tetrahedron's own vertices.

    Returns:
        Sorted list of indices into ``points``.

    Raises:
        HullError: Malformed tetrahedron or points.
    """
    positions, faces = _check_hull(tetra)
    coordinates = _coerce_points(points)
    if not faces:
        raise HullError("tetrahedron has no faces")

    visible = []
    for index, point in enumerate(coordinates):
        for face in faces:
            a, b, c = (positions[vertex] for vertex in face)
            if _signed_distance(a, b, c, point) >= -TOL:
                visible.append(index)
                break
    return visible


def expand(hull, point, visible):
    """Grow ``hull`` by ``point`` after removing the visible faces.

    Args:
        hull: Current hull dictionary.
        point: New coordinate triple; it is appended at index
            ``len(hull["positions"])``.
        visible: Non-empty list of face indices to remove.

    Returns:
        A new hull. The boundary of the removed visible patch is the horizon;
        each horizon edge is connected to the new point with outward winding.
        The result is a 2-manifold: every undirected edge belongs to exactly
        two faces, and the two faces traverse a shared edge in opposite
        directions. No visible face remains, and no internal face is emitted.

    Raises:
        HullError: Empty/invalid ``visible`` indices, non-closed horizon,
            duplicate point, or a result that is not a closed 2-manifold.
    """
    positions, faces = _check_hull(hull)
    new_point = _as_point(point)

    if not visible:
        raise HullError("visible face list must not be empty")
    try:
        raw_visible = list(visible)
    except TypeError:
        raise HullError("visible must be a sequence of face indices")
    for index in raw_visible:
        if isinstance(index, bool) or not isinstance(index, int):
            raise HullError("visible face indices must be integers")
        if index < 0 or index >= len(faces):
            raise HullError("visible face index out of range")
    visible_set = set(raw_visible)
    if len(visible_set) != len(raw_visible):
        raise HullError("visible face indices must be unique")
    if not visible_set:
        raise HullError("visible face list must not be empty")

    for existing in positions:
        if _distance(existing, new_point) <= TOL:
            raise HullError("duplicate point")

    # Collect the directed edges of the removed patch. An edge whose reverse
    # is also in the patch is interior to the patch and cancels out; the rest
    # form the horizon.
    directed = Counter()
    for face_index in visible_set:
        a, b, c = faces[face_index]
        for edge in ((a, b), (b, c), (c, a)):
            directed[edge] += 1

    horizon = []
    for edge, count in directed.items():
        if count != 1:
            raise HullError("visible patch is not a valid oriented surface")
        if directed.get((edge[1], edge[0]), 0) == 0:
            horizon.append(edge)

    if len(horizon) < 3:
        raise HullError("non-closed horizon")

    sources = [edge[0] for edge in horizon]
    targets = [edge[1] for edge in horizon]
    if (
        len(set(sources)) != len(sources)
        or len(set(targets)) != len(targets)
        or set(sources) != set(targets)
    ):
        raise HullError("non-closed horizon")

    apex = len(positions)
    new_faces = [
        face for face_index, face in enumerate(faces) if face_index not in visible_set
    ]
    for a, b in sorted(horizon):
        new_faces.append((a, b, apex))

    grown = {
        "positions": positions + [new_point],
        "faces": new_faces,
    }

    problems = _edge_problems(new_faces)
    if problems:
        raise HullError("expand produced a non-manifold hull: " + problems[0])
    return grown


def _strict_visible(hull, point):
    """Indices of faces the point lies strictly outside of (beyond ``TOL``)."""
    positions = hull["positions"]
    visible = []
    for index, (a, b, c) in enumerate(hull["faces"]):
        pa, pb, pc = positions[a], positions[b], positions[c]
        if _signed_distance(pa, pb, pc, point) > TOL:
            visible.append(index)
    return visible


def _build(points, eps):
    """Deduplicate, seed a tetrahedron and expand it one point at a time."""
    coordinates = dedupe_points(points, eps)
    if len(coordinates) < 4:
        raise HullError("at least four distinct points are required")

    tetra = initial_tetra(coordinates)
    hull = {
        "positions": list(tetra["positions"]),
        "faces": list(tetra["faces"]),
    }

    chosen = {tuple(position) for position in hull["positions"]}
    remaining = [point for point in coordinates if tuple(point) not in chosen]

    for point in remaining:
        visible = _strict_visible(hull, point)
        if visible:
            hull = expand(hull, point, visible)

    return hull


def build_hull(points):
    """Build the complete convex hull of ``points`` incrementally.

    Points are deduplicated with ``eps = 1e-6`` before construction, then an
    initial non-degenerate tetrahedron is expanded one point at a time.

    Returns:
        A hull dictionary whose ``positions`` are hull vertices and whose
        ``faces`` are outward, counter-clockwise triangles. Every face is on
        the boundary; every edge is shared by exactly two faces. Every input
        point lies inside or on the hull within ``TOL``.

    Raises:
        HullError: Fewer than four unique points, no non-degenerate initial
            tetrahedron, or inconsistent input.
    """
    return _build(points, TOL)


def volume(hull):
    """Return the absolute enclosed volume of a closed, oriented hull.

    The value is the absolute sum of
    ``dot(a, cross(b, c)) / 6`` over all outward faces. A degenerate or
    unclosed hull has volume ``0.0`` rather than raising ``HullError``.

    Returns:
        Non-negative float.
    """
    positions, faces = _check_hull(hull)
    if not faces:
        return 0.0
    if _edge_problems(faces):
        return 0.0

    total = 0.0
    for a, b, c in faces:
        total += _dot(positions[a], _cross(positions[b], positions[c]))
    return abs(total) / 6.0


def validate(hull, points):
    """Return a list of human-readable geometry problems; empty means valid.

    Checks performed:
      * every face's normal points outward from the hull centroid; a failure
        message contains ``"inward"``;
      * every undirected edge is shared by exactly two faces, with opposite
        directions; a failure message contains ``"non-manifold"``;
      * every point in ``points`` has signed distance ``<= TOL`` from every
        face plane; a failure message contains ``"outside"``.

    The number and ordering of messages are not significant. A structurally
    malformed hull (missing keys, bad indices, non-triangular face) raises
    ``HullError`` rather than being reported as a geometry problem.

    Returns:
        List of problem strings.
    """
    positions, faces = _check_hull(hull)
    coordinates = _coerce_points(points)

    problems = []

    if positions:
        centroid = (
            sum(point[0] for point in positions) / len(positions),
            sum(point[1] for point in positions) / len(positions),
            sum(point[2] for point in positions) / len(positions),
        )
        for index, (a, b, c) in enumerate(faces):
            pa, pb, pc = positions[a], positions[b], positions[c]
            normal = _face_normal(pa, pb, pc)
            if _norm(normal) == 0.0:
                problems.append("face %d is degenerate (inward)" % index)
                continue
            face_center = (
                (pa[0] + pb[0] + pc[0]) / 3.0,
                (pa[1] + pb[1] + pc[1]) / 3.0,
                (pa[2] + pb[2] + pc[2]) / 3.0,
            )
            if _dot(normal, _sub(face_center, centroid)) <= 0.0:
                problems.append("face %d points inward" % index)

    problems.extend(_edge_problems(faces))

    for index, (a, b, c) in enumerate(faces):
        pa, pb, pc = positions[a], positions[b], positions[c]
        for point_index, point in enumerate(coordinates):
            try:
                distance = _signed_distance(pa, pb, pc, point)
            except HullError:
                continue
            if distance > TOL:
                problems.append(
                    "point %d is outside face %d" % (point_index, index)
                )

    return problems


def hull_batch(point_sets, config):
    """Build many independent hulls transactionally.

    Args:
        point_sets: Sequence of point lists. Every set is built independently
            and may have a different size.
        config: Mapping containing ``"eps"``, a positive finite merge radius.

    Returns:
        ``{"hulls": [...]}`` in the same order as ``point_sets``. Each hull
        owns independent ``positions`` and ``faces`` containers, so a face
        table is never shared between point sets.

    Raises:
        HullError: ``config["eps"]`` is missing, non-finite, or ``<= 0``; or
            any point set is invalid. The operation is all-or-nothing: on
            failure no partial result is returned, and since this is a pure
            function, no externally visible state is mutated (rollback).
    """
    if config is None:
        raise HullError("config must provide 'eps'")
    try:
        eps_value = config["eps"]
    except (KeyError, TypeError):
        raise HullError("config must provide 'eps'")
    eps = _check_eps(eps_value)

    try:
        sets = list(point_sets)
    except TypeError:
        raise HullError("point_sets must be a sequence of point lists")

    hulls = [_build(points, eps) for points in sets]
    return {"hulls": hulls}
