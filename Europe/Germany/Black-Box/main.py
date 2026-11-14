from itertools import combinations, product

import numpy as np

PAIRS = list(combinations(range(5), 2))
NAMES = [f'x{index + 1}' for index in range(5)]


def design(points):
    points = np.asarray(points, dtype=float)
    return np.column_stack([np.ones(len(points)), points, *[points[:, left] * points[:, right] for left, right in PAIRS]])


def spread(edges, boundary_points):
    midpoints = np.array([(left + right) / 2 for left, right in edges])
    if not boundary_points:
        return 0
    offsets = midpoints - boundary_points[0]
    if len(boundary_points) > 1:
        basis = np.array(boundary_points[1:]) - boundary_points[0]
        _, singular, vectors = np.linalg.svd(basis, full_matrices=False)
        rank = int(np.sum(singular > 1e-4))
        if rank:
            offsets -= (offsets @ vectors[:rank].T) @ vectors[:rank]
    return int(np.argmax(np.linalg.norm(offsets, axis=1)))


def boundary_fit(boundary_points, corners, labels, tolerance):
    best = None
    for size in range(1, 6):
        eligible = []
        for subset in combinations(range(5), size):
            coordinates = np.array(boundary_points)[:, subset]
            centered = coordinates - coordinates.mean(axis=0)
            _, _, vectors = np.linalg.svd(centered, full_matrices=True)
            direction = vectors[-1]
            normal = np.zeros(5)
            normal[list(subset)] = direction
            threshold = float(np.mean(np.array(boundary_points) @ normal))
            if float(np.mean(corners[labels == 1] @ normal)) < float(np.mean(corners[labels == 0] @ normal)):
                normal = -normal
                threshold = -threshold
            distances = np.array(boundary_points) @ normal - threshold
            corner_scores = corners @ normal - threshold
            violations = np.maximum(0, -(2 * labels - 1) * corner_scores)
            error = float(max(np.max(np.abs(distances)), np.max(violations)))
            eligible.append((error, normal, threshold))
        eligible.sort(key=lambda entry: entry[0])
        if best is None or eligible[0][0] < best[0]:
            best = eligible[0]
        if eligible[0][0] <= tolerance:
            return eligible[0][1:]
    return best[1:]


def solve(oracle, points):
    queried = []
    readings = []

    def ask_many(probes, phase):
        probes = [np.asarray(probe, dtype=float).tolist() for probe in probes]
        answers = oracle.ask_many(probes, phase=phase)
        queried.extend(probes)
        readings.extend(answers)
        return np.asarray(answers, dtype=float)

    corners = np.array(list(product((0.0, 1.0), repeat=5)))
    local = 0.5 + (corners - 0.5) * 0.2
    local_y = ask_many(local, 'local factorial design')
    smooth = np.linalg.lstsq(design(local), local_y, rcond=None)[0]
    local_error = float(np.max(np.abs(design(local) @ smooth - local_y)))
    if local_error > 2e-5:
        local = 0.25 + (corners - 0.5) * 0.08
        local_y = ask_many(local, 'alternate smooth neighbourhood')
        smooth = np.linalg.lstsq(design(local), local_y, rcond=None)[0]
    corner_y = ask_many(corners, 'global factorial design')
    residuals = corner_y - design(corners) @ smooth
    ordered = np.sort(residuals)
    split = int(np.argmax(np.diff(ordered)))
    cutoff = float((ordered[split] + ordered[split + 1]) / 2)
    labels = (residuals > cutoff).astype(int)
    low = float(np.median(residuals[labels == 0]))
    high = float(np.median(residuals[labels == 1]))
    jump = high - low
    smooth[0] += low

    edges = []
    for left in range(len(corners)):
        for right in range(left + 1, len(corners)):
            if labels[left] != labels[right] and np.sum(corners[left] != corners[right]) == 1:
                inactive, active = (left, right) if labels[left] == 0 else (right, left)
                edges.append((corners[inactive].copy(), corners[active].copy()))
    boundary_points = []
    rounds = min(10, max(5, (oracle.remaining - 10) // 6))
    for index in range(min(6, len(edges))):
        edge = spread(edges, boundary_points)
        inactive, active = edges.pop(edge)
        for iteration in range(rounds):
            midpoint = (inactive + active) / 2
            value = ask_many([midpoint], 'boundary bisection')[0]
            residual = value - design([midpoint])[0] @ smooth
            if residual > jump / 2:
                active = midpoint
            else:
                inactive = midpoint
        boundary_points.append((inactive + active) / 2)
    tolerance = 2.5 * 2 ** (-rounds)
    normal, threshold = boundary_fit(boundary_points, corners, labels, tolerance)

    rng = np.random.default_rng(314159)
    probes = rng.uniform(0.03, 0.97, size=(min(12, oracle.remaining), 5))
    if len(probes):
        ask_many(probes, 'independent confirmation')
    locations = np.array(queried)
    observed = np.array(readings)
    distance = locations @ normal - threshold
    keep = np.abs(distance) > max(0.006, tolerance * 2)
    matrix = np.column_stack([design(locations[keep]), (distance[keep] >= 0).astype(float)])
    fitted = np.linalg.lstsq(matrix, observed[keep], rcond=None)[0]
    pair_index = int(np.argmax(np.abs(fitted[6:16])))
    left, right = PAIRS[pair_index]
    reduced = np.column_stack([np.ones(np.sum(keep)), locations[keep], locations[keep, left] * locations[keep, right], (distance[keep] >= 0).astype(float)])
    coefficients = np.linalg.lstsq(reduced, observed[keep], rcond=None)[0]
    intercept = float(coefficients[0])
    linear = coefficients[1:6]
    interaction = float(coefficients[6])
    jump = float(coefficients[7])
    gate_indices = np.flatnonzero(np.abs(normal) > 0.005)
    relevant = {left, right} | set(gate_indices) | set(np.flatnonzero(np.abs(linear) > 0.001))
    boundary = {name: float(normal[index]) for index, name in enumerate(NAMES)}
    condition = ' + '.join(f'({normal[index]:.12g})*{name}' for index, name in enumerate(NAMES) if abs(normal[index]) > 0.005)
    expression = f'({intercept:.12g})'
    expression += ''.join(f' + ({coefficient:.12g})*{name}' for name, coefficient in zip(NAMES, linear))
    expression += f' + ({interaction:.12g})*{NAMES[left]}*{NAMES[right]}'
    expression += f' + ({jump:.12g} if {condition} >= {threshold:.12g} else 0.0)'
    challenge = np.asarray(points, dtype=float)
    predictions = intercept + challenge @ linear + interaction * challenge[:, left] * challenge[:, right]
    predictions += jump * (challenge @ normal >= threshold)
    fitted_error = float(np.sqrt(np.mean((reduced @ coefficients - observed[keep]) ** 2)))
    return {
        'relevant_variables': [NAMES[index] for index in sorted(relevant)],
        'irrelevant_variables': [NAMES[index] for index in range(5) if index not in relevant],
        'interactions': [{'variables': [NAMES[left], NAMES[right]], 'type': 'product', 'coefficient': interaction}],
        'discontinuities': [{
            'gate_variables': [NAMES[index] for index in gate_indices],
            'boundary': {'form': 'linear', 'coefficients': boundary, 'threshold': float(threshold), 'active_when': 'greater_equal'},
            'jump': jump,
            'condition': condition + f' >= {threshold:.12g}',
        }],
        'model_expression': expression,
        'predictions': predictions.tolist(),
        'notes': f'Factorial probes identify the smooth interaction; corner residuals identify the step. Adaptive edge bisections locate its plane. Queries: {oracle.spent}; smooth-region fit RMSE: {fitted_error:.8g}.',
    }
