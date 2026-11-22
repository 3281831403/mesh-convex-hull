import pytest

from hull.build import (
    HullError,
    build_hull,
    conflict_list,
    dedupe_points,
    expand,
    hull_batch,
    initial_tetra,
    validate,
    volume,
)

TETRA_POINTS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]

# Outward counter-clockwise winding, matching the module contract.
TETRA_FACES = [
    (0, 2, 1),
    (0, 1, 3),
    (0, 3, 2),
    (1, 2, 3),
]

CUBE_POINTS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]


def _tetra_hull():
    return {"positions": list(TETRA_POINTS), "faces": list(TETRA_FACES)}


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


def test_dedupe_merges_points_within_eps():
    points = [
        (0.0, 0.0, 0.0),
        (4e-7, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0 + 5e-7, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]

    kept = dedupe_points(points, 1e-6)

    assert len(kept) == 3
    assert kept[0] == pytest.approx((0.0, 0.0, 0.0))
    assert kept[1] == pytest.approx((1.0, 0.0, 0.0))
    assert kept[2] == pytest.approx((2.0, 0.0, 0.0))


def test_initial_tetra_rejects_coplanar_points():
    coplanar = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]

    with pytest.raises(HullError):
        initial_tetra(coplanar)


def test_conflict_list_includes_coplanar_points():
    # The first point is outside the tetrahedron yet exactly on the z=0 face
    # plane; a strict ">" visibility test would miss it.
    points = [(2.0, 2.0, 0.0), (0.2, 0.2, 0.2)]

    assert conflict_list(_tetra_hull(), points) == [0]


def test_expand_removes_visible_face_and_adds_apex():
    hull = _tetra_hull()
    removed_face = hull["faces"][3]

    grown = expand(hull, (0.25, 0.25, 1.0), [3])

    assert removed_face not in grown["faces"]
    assert any(4 in face for face in grown["faces"])
    assert all(index < len(grown["positions"]) for face in grown["faces"] for index in face)


def test_face_winding_is_outward():
    hull = build_hull(TETRA_POINTS)
    positions = hull["positions"]
    center = tuple(
        sum(point[axis] for point in positions) / len(positions)
        for axis in range(3)
    )

    for i, j, k in hull["faces"]:
        a, b, c = positions[i], positions[j], positions[k]
        normal = _cross(_sub(b, a), _sub(c, a))
        face_center = tuple((a[axis] + b[axis] + c[axis]) / 3.0 for axis in range(3))
        assert _dot(normal, _sub(face_center, center)) > 1e-9


def test_volume_divides_by_six():
    assert volume(_tetra_hull()) == pytest.approx(1.0 / 6.0, abs=1e-6)


def test_validate_detects_non_manifold_edges():
    hull = _tetra_hull()
    hull["faces"].append((0, 1, 2))

    problems = validate(hull, TETRA_POINTS)

    assert any("non-manifold" in problem.lower() for problem in problems)


def test_validate_flags_points_outside_hull():
    problems = validate(_tetra_hull(), TETRA_POINTS + [(2.0, 2.0, 2.0)])

    assert any("outside" in problem.lower() for problem in problems)


def test_batch_does_not_share_face_tables():
    result = hull_batch([TETRA_POINTS, CUBE_POINTS], {"eps": 1e-6})
    hulls = result["hulls"]

    assert len(hulls) == 2
    assert hulls[0]["positions"] is not hulls[1]["positions"]
    assert hulls[0]["faces"] is not hulls[1]["faces"]
    assert validate(hulls[0], TETRA_POINTS) == []
    assert validate(hulls[1], CUBE_POINTS) == []


def test_dedupe_rejects_nonpositive_eps():
    for eps in (0.0, -1e-6):
        with pytest.raises(HullError):
            dedupe_points(TETRA_POINTS, eps)


def test_unit_cube_volume_is_one():
    hull = build_hull(CUBE_POINTS)

    assert volume(hull) == pytest.approx(1.0, abs=1e-6)


def test_all_cube_points_are_inside():
    hull = build_hull(CUBE_POINTS)

    assert validate(hull, CUBE_POINTS) == []


def test_fewer_than_four_points_raises():
    with pytest.raises(HullError):
        build_hull([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])


def test_batch_preserves_order():
    result = hull_batch([TETRA_POINTS, CUBE_POINTS], {"eps": 1e-6})
    volumes = [volume(hull) for hull in result["hulls"]]

    assert volumes[0] == pytest.approx(1.0 / 6.0, abs=1e-6)
    assert volumes[1] == pytest.approx(1.0, abs=1e-6)


def test_batch_rejects_zero_eps():
    with pytest.raises(HullError):
        hull_batch([CUBE_POINTS], {"eps": 0.0})
