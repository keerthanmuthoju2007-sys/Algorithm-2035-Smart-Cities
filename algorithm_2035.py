"""
Algorithm 2035: Adaptive Algorithm Selection Framework for Smart Cities
Course: Design and Analysis of Algorithms (DAA)

Modules:
1. Adaptive route planning: BFS or Dijkstra
2. Emergency prioritization: linear scan or min-heap
3. Sensor record lookup: hash table or sorted index
4. Budget allocation: greedy baseline or 0/1 Knapsack DP
5. Service-tour planning: nearest-neighbour heuristic
6. Simple timing and decision logging

Run: python algorithm_2035.py
"""

from collections import deque
import heapq
import time


# ------------------------------------------------------------
# 1. ROUTE PLANNING: BFS OR DIJKSTRA
# ------------------------------------------------------------

def bfs(graph, start, goal):
    """Shortest path by number of edges; appropriate for equal-cost roads."""
    queue = deque([start])
    parent = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for neighbour, _weight in graph.get(current, []):
            if neighbour not in parent:
                parent[neighbour] = current
                queue.append(neighbour)

    if goal not in parent:
        return None, None

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path, len(path) - 1


def dijkstra(graph, start, goal):
    """Minimum-cost path for graphs with non-negative edge weights."""
    distances = {node: float("inf") for node in graph}
    parents = {start: None}
    distances[start] = 0
    heap = [(0, start)]

    while heap:
        distance, current = heapq.heappop(heap)
        if distance != distances[current]:
            continue
        if current == goal:
            break

        for neighbour, weight in graph.get(current, []):
            if weight < 0:
                raise ValueError("Dijkstra cannot handle negative road weights.")
            new_distance = distance + weight
            if new_distance < distances.get(neighbour, float("inf")):
                distances[neighbour] = new_distance
                parents[neighbour] = current
                heapq.heappush(heap, (new_distance, neighbour))

    if distances.get(goal, float("inf")) == float("inf"):
        return None, float("inf")

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    return path, distances[goal]


def route_selector(graph, start, goal, all_equal_weights=True):
    """Choose BFS for equal edge costs; otherwise choose Dijkstra."""
    if all_equal_weights:
        algorithm = "BFS"
        path, cost = bfs(graph, start, goal)
        reason = "All roads have equal cost; BFS finds a path with the fewest edges."
    else:
        algorithm = "Dijkstra"
        path, cost = dijkstra(graph, start, goal)
        reason = "Road costs differ; Dijkstra finds the minimum-cost path for non-negative weights."
    return algorithm, path, cost, reason


# ------------------------------------------------------------
# 2. EMERGENCY PRIORITIZATION: SCAN OR MIN-HEAP
# Lower priority number means more urgent.
# ------------------------------------------------------------

def most_urgent_scan(events):
    if not events:
        return None
    return min(events, key=lambda event: (event["priority"], event["id"]))


def most_urgent_heap(events):
    if not events:
        return None
    heap = [(event["priority"], event["id"], event) for event in events]
    heapq.heapify(heap)
    return heapq.heappop(heap)[2]


def emergency_selector(events):
    if len(events) < 10:
        method = "Linear scan"
        event = most_urgent_scan(events)
        reason = "The queue is small, so a simple scan has low overhead."
    else:
        method = "Min-heap"
        event = most_urgent_heap(events)
        reason = "The queue is larger; a heap supports efficient priority retrieval."
    return method, event, reason


# ------------------------------------------------------------
# 3. SENSOR RECORDS: HASH LOOKUP OR SORTED INDEX
# ------------------------------------------------------------

def sensor_lookup_selector(records, sensor_id, ordered_query=False):
    """
    records: dictionary such as {"S01": 24.5, "S02": 27.1}
    Hash/dictionary is used for exact key lookup.
    Sorted index is demonstrated for ordered listing.
    """
    if ordered_query:
        method = "Sorted index"
        result = sorted(records.items(), key=lambda item: item[0])
        reason = "An ordered query needs records arranged by sensor ID."
    else:
        method = "Hash table (Python dictionary)"
        result = records.get(sensor_id)
        reason = "An exact sensor-ID lookup is well suited to a hash table."
    return method, result, reason


# ------------------------------------------------------------
# 4. BUDGET ALLOCATION: 0/1 KNAPSACK DYNAMIC PROGRAMMING
# Each project can be selected at most once.
# ------------------------------------------------------------

def knapsack_dp(projects, budget):
    """
    projects: list of dicts with name, cost, value
    Returns chosen projects and total value/cost.
    """
    if budget < 0:
        raise ValueError("Budget cannot be negative.")

    n = len(projects)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    for i, project in enumerate(projects, start=1):
        cost = project["cost"]
        value = project["value"]
        for capacity in range(budget + 1):
            dp[i][capacity] = dp[i - 1][capacity]
            if cost <= capacity:
                dp[i][capacity] = max(
                    dp[i][capacity],
                    dp[i - 1][capacity - cost] + value
                )

    chosen = []
    capacity = budget
    for i in range(n, 0, -1):
        if dp[i][capacity] != dp[i - 1][capacity]:
            project = projects[i - 1]
            chosen.append(project)
            capacity -= project["cost"]

    chosen.reverse()
    total_cost = sum(project["cost"] for project in chosen)
    total_value = sum(project["value"] for project in chosen)
    return chosen, total_cost, total_value


# ------------------------------------------------------------
# 5. SERVICE TOUR: NEAREST-NEIGHBOUR HEURISTIC
# This is a heuristic, not a guaranteed optimal TSP solution.
# ------------------------------------------------------------

def nearest_neighbour_tour(distance_matrix, start=0):
    n = len(distance_matrix)
    if n == 0:
        return [], 0

    unvisited = set(range(n))
    unvisited.discard(start)
    route = [start]
    total_cost = 0
    current = start

    while unvisited:
        next_stop = min(unvisited, key=lambda stop: distance_matrix[current][stop])
        total_cost += distance_matrix[current][next_stop]
        route.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop

    if n > 1:
        total_cost += distance_matrix[current][start]
        route.append(start)

    return route, total_cost


# ------------------------------------------------------------
# 6. ADAPTIVE FRAMEWORK DEMONSTRATION
# ------------------------------------------------------------

def main():
    print("=" * 68)
    print("ALGORITHM 2035: ADAPTIVE ALGORITHM SELECTION FRAMEWORK")
    print("=" * 68)

    # A. Routing demonstration
    # The graph is undirected. Each edge is (neighbour, travel cost).
    equal_cost_graph = {
        "A": [("B", 1), ("C", 1)],
        "B": [("A", 1), ("C", 1), ("D", 1)],
        "C": [("A", 1), ("B", 1), ("D", 1)],
        "D": [("B", 1), ("C", 1)],
    }
    weighted_graph = {
        "A": [("B", 4), ("C", 1)],
        "B": [("A", 4), ("C", 2), ("D", 1)],
        "C": [("A", 1), ("B", 2), ("D", 5)],
        "D": [("B", 1), ("C", 5)],
    }

    print("\\n1. SMART-CITY ROUTE PLANNING")
    for graph, equal_weights in ((equal_cost_graph, True), (weighted_graph, False)):
        method, path, cost, reason = route_selector(
            graph, "A", "D", all_equal_weights=equal_weights
        )
        print(f"Selected algorithm: {method}")
        print(f"Reason: {reason}")
        print(f"Route: {path}")
        print(f"Route cost / edge count: {cost}\\n")

    # B. Emergency priority demonstration
    print("2. EMERGENCY PRIORITIZATION")
    events = [
        {"id": "E101", "priority": 3, "description": "Water leakage"},
        {"id": "E102", "priority": 1, "description": "Major road accident"},
        {"id": "E103", "priority": 2, "description": "Traffic signal failure"},
    ]
    method, event, reason = emergency_selector(events)
    print(f"Selected structure: {method}")
    print(f"Reason: {reason}")
    print(f"Next emergency: {event}\n")

    # C. Sensor record lookup demonstration
    print("3. SENSOR RECORD LOOKUP")
    sensors = {"S01": 24.5, "S03": 29.2, "S02": 26.8}
    method, result, reason = sensor_lookup_selector(sensors, "S02")
    print(f"Selected structure: {method}")
    print(f"Reason: {reason}")
    print(f"Lookup result: S02 = {result} degrees\n")

    method, result, reason = sensor_lookup_selector(
        sensors, "S02", ordered_query=True
    )
    print(f"Selected structure: {method}")
    print(f"Reason: {reason}")
    print(f"Ordered records: {result}\n")

    # D. Budget allocation demonstration
    print("4. SMART-CITY BUDGET ALLOCATION")
    projects = [
        {"name": "Repair traffic lights", "cost": 4, "value": 8},
        {"name": "Install water sensors", "cost": 3, "value": 6},
        {"name": "Upgrade streetlights", "cost": 5, "value": 9},
        {"name": "Improve emergency alerts", "cost": 2, "value": 5},
    ]
    budget = 7
    chosen, total_cost, total_value = knapsack_dp(projects, budget)
    print(f"Budget limit: {budget}")
    print("Selected projects:")
    for project in chosen:
        print(f"  - {project['name']} (cost={project['cost']}, value={project['value']})")
    print(f"Total cost: {total_cost}")
    print(f"Total value: {total_value}\n")

    # E. Service tour demonstration
    print("5. WASTE-COLLECTION / MAINTENANCE TOUR")
    distances = [
        [0, 4, 8, 6],
        [4, 0, 3, 7],
        [8, 3, 0, 2],
        [6, 7, 2, 0],
    ]
    start_time = time.perf_counter()
    tour, tour_cost = nearest_neighbour_tour(distances, start=0)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print("Selected method: Nearest-neighbour heuristic")
    print("Reason: A quick approximate tour is useful for this demonstration.")
    print(f"Tour (locations indexed from 0): {tour}")
    print(f"Total tour distance: {tour_cost}")
    print(f"Execution time: {elapsed_ms:.4f} ms")
    print("Note: the heuristic does not guarantee the globally shortest tour.\n")

    # F. Summary of the adaptive decision
    print("6. FRAMEWORK SUMMARY")
    print("- Equal-cost roads       -> BFS")
    print("- Different non-negative road costs -> Dijkstra")
    print("- Small emergency queue  -> Linear scan")
    print("- Larger emergency queue -> Min-heap")
    print("- Exact sensor ID        -> Hash table")
    print("- Ordered sensor records -> Sorted index")
    print("- Budget-limited projects -> 0/1 Knapsack DP")
    print("- Multi-stop route        -> Nearest-neighbour heuristic")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
