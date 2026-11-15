import csv
import heapq
import re
from collections import defaultdict


GAIN = {
    'lookup': 0.0,
    'walk_indirect': 0.067,
    'stock_small': 0.091,
    'takings_fee': 0.286,
    'walk_direct': 0.378,
    'lookup_chain': 0.455,
    'stock_large': 0.524,
    'clock_pause': 0.586,
    'clock': 0.757,
    'takings': 0.846,
    'toll': 0.846,
    'toll_heavy': 0.926,
    'stock_sales': 1.0,
    'unknown': 0.6,
}


def category(prompt):
    text = prompt.lower()
    if 'board lists' in text:
        return 'lookup_chain' if 'whatever' in text else 'lookup'
    if 'gate charges' in text:
        return 'toll_heavy' if 'heavier' in text else 'toll'
    if 'stall opens' in text:
        return 'clock_pause' if 'pauses' in text else 'clock'
    if 'takings were' in text:
        return 'takings_fee' if 'stall fee' in text else 'takings'
    if 'loads' in text:
        if 'sells' in text:
            return 'stock_sales'
        crates = re.search(r'loads (\d+) crates .*? with (\d+)', text)
        if crates:
            return 'stock_small' if max(map(int, crates.groups())) <= 12 else 'stock_large'
    if 'walking times' in text:
        trip = re.search(r'walk from (\w+) to (\w+)', text)
        edges = re.findall(r'(\w+)-(\w+) (\d+) minutes', text)
        if not trip or not edges:
            return 'unknown'
        start, end = trip.groups()
        graph = defaultdict(list)
        direct = float('inf')
        for origin, destination, minutes in edges:
            minutes = int(minutes)
            graph[origin].append((destination, minutes))
            graph[destination].append((origin, minutes))
            if {origin, destination} == {start, end}:
                direct = min(direct, minutes)
        distances = {start: 0}
        queue = [(0, start)]
        while queue:
            distance, town = heapq.heappop(queue)
            if distance != distances[town]:
                continue
            for neighbor, minutes in graph[town]:
                candidate = distance + minutes
                if candidate < distances.get(neighbor, float('inf')):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        if end not in distances:
            return 'unknown'
        return 'walk_direct' if direct == distances[end] else 'walk_indirect'
    return 'unknown'


def route(prompts, gains=GAIN):
    priorities = [gains.get(category(prompt), 0.6) for prompt in prompts]
    ranked = sorted(range(len(prompts)), key=lambda index: (-priorities[index], index))
    cap = 78 * len(prompts) // 100
    selected = {index for index in ranked[:cap] if priorities[index] >= 0.2}
    return ['expensive' if index in selected else 'cheap' for index in range(len(prompts))]


with open('test-input.csv', encoding='utf-8-sig', newline='') as stream:
    requests = list(csv.DictReader(stream))
choices = route([request['prompt'] for request in requests])
with open('routing_log.csv', 'w', encoding='utf-8', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['request_id', 'model'])
    writer.writerows((request['request_id'], model) for request, model in zip(requests, choices))
