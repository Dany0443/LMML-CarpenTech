from itertools import permutations

import numpy as np

ALPHABET = 'AKRXBGSZCDOQEFLPHMNWIJTY'
GROUPS = [list(range(start, start + 4)) for start in range(0, 24, 4)]
STYLES = 6


def fit_styles(truth_letters, read_letters, bird_indices, birds):
    prior = np.full((24, 24), 0.001)
    for group in GROUPS:
        prior[np.ix_(group, group)] = 1
    starts = []
    best_fit = -np.inf
    best_matrices = None
    best_assignments = None
    for seed in range(12):
        rng = np.random.default_rng(seed)
        matrices = np.tile(prior, (STYLES, 1, 1)) * rng.uniform(0.2, 2, (STYLES, 24, 24))
        for style in range(STYLES):
            for group in GROUPS:
                matrices[style, group, rng.permutation(group)] += 6
        matrices /= matrices.sum(axis=2, keepdims=True)
        for iteration in range(80):
            likelihood = np.zeros((birds, STYLES))
            for style in range(STYLES):
                np.add.at(likelihood[:, style], bird_indices, np.log(matrices[style, truth_letters, read_letters]))
            largest = likelihood.max(axis=1, keepdims=True)
            weights = np.exp(likelihood - largest)
            normalisers = weights.sum(axis=1, keepdims=True)
            fit = float((largest + np.log(normalisers)).sum())
            weights /= normalisers
            counts = np.tile(prior * 0.1, (STYLES, 1, 1))
            for style in range(STYLES):
                np.add.at(counts[style], (truth_letters, read_letters), weights[bird_indices, style])
            matrices = counts / counts.sum(axis=2, keepdims=True)
        starts.append((seed, fit))
        if fit > best_fit:
            best_fit = fit
            best_matrices = matrices.copy()
            best_assignments = likelihood.argmax(axis=1)
    return best_matrices, best_assignments, starts


def decode_ring(calibration, readings):
    rings = sorted({ring for ring, _ in calibration})
    ring_index = {ring: index for index, ring in enumerate(rings)}
    truth_letters = []
    read_letters = []
    bird_indices = []
    for ring, reading in calibration:
        for true_letter, read_letter in zip(ring, reading):
            if read_letter == '?':
                continue
            truth_letters.append(ALPHABET.index(true_letter))
            read_letters.append(ALPHABET.index(read_letter))
            bird_indices.append(ring_index[ring])
    truth_letters = np.array(truth_letters)
    read_letters = np.array(read_letters)
    bird_indices = np.array(bird_indices)
    matrices, assignments, starts = fit_styles(truth_letters, read_letters, bird_indices, len(rings))
    substitutions = np.zeros((STYLES, 24), dtype=int)
    for style, matrix in enumerate(matrices):
        for group in GROUPS:
            best = max(permutations(group), key=lambda perm: sum(matrix[left, right] for left, right in zip(group, perm)))
            substitutions[style, group] = best
    error_counts = np.ones((STYLES, 4))
    for true_letter, read_letter, bird in zip(truth_letters, read_letters, bird_indices):
        if true_letter // 4 != read_letter // 4:
            continue
        style = assignments[bird]
        expected = substitutions[style, true_letter] % 4
        error = expected ^ (read_letter % 4)
        error_counts[style, error] += 1
    error_rates = error_counts / error_counts.sum(axis=1, keepdims=True)
    votes = np.zeros((6, 24))
    for reading in readings:
        for position, letter in enumerate(reading):
            if letter != '?':
                votes[position, ALPHABET.index(letter)] += 1
    candidates = []
    probabilities = np.full((STYLES, 24, 24), 1e-40)
    for style in range(STYLES):
        for true_letter in range(24):
            for read_letter in range(24):
                if true_letter // 4 == read_letter // 4:
                    error = (substitutions[style, true_letter] % 4) ^ (read_letter % 4)
                    probabilities[style, true_letter, read_letter] = error_rates[style, error]
        scores = np.log(probabilities[style]) @ votes.T
        letters = scores.argmax(axis=0)
        inscription = ''.join(ALPHABET[index] for index in letters)
        candidates.append((style, inscription, float(scores[letters, np.arange(6)].sum())))
    candidates.sort(key=lambda row: -row[2])
    winning_style, ring, _ = candidates[0]
    modal = ''.join(ALPHABET[index] for index in votes.argmax(axis=1))
    transformations = []
    for position, letter in enumerate(ring):
        probability = probabilities[winning_style, ALPHABET.index(letter), ALPHABET.index(modal[position])]
        transformations.append((position + 1, letter, modal[position], float(probability)))
    return ring, candidates, starts, transformations
