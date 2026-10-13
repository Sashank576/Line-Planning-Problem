# Line planning with demand groups - exact MILP (Gurobi)
import sys
import time
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
import networkx as nx
from gurobipy import GRB


# ---------------- settings ----------------

DATASET = "athens"
if len(sys.argv) > 1:
    DATASET = sys.argv[1]
ROOT = Path(".")

# where the edge travel time comes from: "lower", "upper", "average" or "length"
# (bounds in Edge.giv are in time units, we divide by time_units_per_minute)
TRAVEL_TIME_SOURCE = "lower"

# transfer penalty in minutes. None = read from Config.cnf (or 5 if no config)
TRANSFER_PENALTY = None

# budget: a number, "auto" (BUDGET_FACTOR * total travel time of all edges)
# or "config" (lc_budget from Config.cnf, but that is NOT in minutes!)
BUDGET = "auto"
BUDGET_FACTOR = 2.0

MIN_STOPS_PER_LINE = 2
MAX_STOPS_PER_LINE = 8
MAX_CANDIDATE_LINES = 20000
L_MIN = 1
L_MAX = 20

K_PATHS = 5        # routes per OD pair
ALPHA = 2.0        # route <= ALPHA * shortest path

K_GROUPS = 3
GROUP_SCHEME = "demand-share"   # or "count"
RHO = 2.0

TIME_LIMIT = 600
MIP_GAP = None     # e.g. 0.01, None = gurobi default
SHOW_OD_DETAILS = False


# ---------------- reading files ----------------

def find_file(name, folder):
    # look for the file under the dataset folder, then under the current folder
    # (prefers a "basis" subfolder, like OpenLinTim uses)
    for base in (folder, ROOT):
        if not base.exists():
            continue
        hits = [p for p in base.rglob("*") if p.is_file() and p.name.lower() == name.lower()]
        if hits:
            hits.sort(key=lambda p: ("basis" not in str(p.parent).lower(), len(p.parts), str(p)))
            return hits[0]
    raise FileNotFoundError("Could not find " + name + " in " + str(folder))


def read_rows(filename):
    # rows of a .giv file split on ';', skipping empty lines and comments
    rows = []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append([x.strip() for x in line.split(";")])
    return rows


def read_config(path, seen=None):
    # Config.cnf: "name; value" per line, "include" lines point to other configs
    settings = {}
    if path is None or not Path(path).exists():
        return settings
    seen = seen or set()
    path = Path(path).resolve()
    if path in seen:
        return settings
    seen.add(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ";" not in line:
                continue
            name, value = [x.strip() for x in line.split(";", 1)]
            value = value.split("#")[0].strip().strip('"')
            if name == "setting-name":
                continue
            if name in ("include", "include_if_exists"):
                settings.update(read_config(path.parent / value, seen))
            else:
                settings[name] = value
    return settings


def config_float(config, key, default):
    try:
        return float(config[key])
    except (KeyError, ValueError):
        return default


print("Reading data...")

dataset_dir = ROOT / DATASET
stop_file = find_file("Stop.giv", dataset_dir)
data_dir = stop_file.parent
edge_file = data_dir / "Edge.giv"
od_file = data_dir / "OD.giv"
if not edge_file.exists():
    edge_file = find_file("Edge.giv", dataset_dir)
if not od_file.exists():
    od_file = find_file("OD.giv", dataset_dir)

# Config.cnf is usually one folder above "basis"
config_file = None
for folder in (data_dir, data_dir.parent):
    if (folder / "Config.cnf").exists():
        config_file = folder / "Config.cnf"
        break

print("Stop file :", stop_file)
print("Edge file :", edge_file)
print("OD file   :", od_file)
print("Config    :", config_file)

config = read_config(config_file)
units_per_minute = config_float(config, "time_units_per_minute", 1.0)
network_name = config.get("ptn_name", DATASET)

if TRANSFER_PENALTY is None:
    change_units = config_float(config, "ean_change_penalty", None)
    if change_units is None:
        TRANSFER_PENALTY = 5.0
        print("No ean_change_penalty in config, using 5 min")
    else:
        TRANSFER_PENALTY = change_units / units_per_minute
        print("Transfer penalty from config:", TRANSFER_PENALTY, "min")

# stops: id; short name; long name; x; y
stops = {}
for p in read_rows(stop_file):
    stops[int(p[0])] = {"short_name": p[1], "x": float(p[-2]), "y": float(p[-1])}

# edges: id; left; right; length; lower; upper
edges = {}
for p in read_rows(edge_file):
    length = float(p[3])
    lower = float(p[4]) if len(p) > 4 else length
    upper = float(p[5]) if len(p) > 5 else lower
    edges[int(p[0])] = {"left": int(p[1]), "right": int(p[2]),
                        "length": length, "lower": lower, "upper": upper}

# demand: origin; destination; customers (only keep > 0)
demand = {}
for p in read_rows(od_file):
    o, d, c = int(p[0]), int(p[1]), float(p[2])
    if c > 0 and o != d:
        demand[(o, d)] = c

print("Network:", network_name)
print("Stops:", len(stops), " Edges:", len(edges), " OD pairs:", len(demand))


# travel time of every edge in minutes
for e_id, e in edges.items():
    if e["left"] not in stops or e["right"] not in stops:
        raise ValueError("Edge " + str(e_id) + " uses an unknown stop")
    if TRAVEL_TIME_SOURCE == "lower":
        t = e["lower"] / units_per_minute
    elif TRAVEL_TIME_SOURCE == "upper":
        t = e["upper"] / units_per_minute
    elif TRAVEL_TIME_SOURCE == "average":
        t = (e["lower"] + e["upper"]) / (2 * units_per_minute)
    else:
        t = e["length"]
    if t <= 0:
        t = e["length"]     # no usable bounds, fall back to the length
    if t <= 0:
        raise ValueError("Edge " + str(e_id) + " has travel time <= 0")
    e["time"] = t

total_network_time = sum(e["time"] for e in edges.values())
total_demand = sum(demand.values())
print("Total demand:", total_demand)
print("Total edge travel time:", round(total_network_time, 2), "min")

if BUDGET == "auto":
    budget_value = BUDGET_FACTOR * total_network_time
elif BUDGET == "config":
    budget_value = config_float(config, "lc_budget", None)
    if budget_value is None:
        raise ValueError("BUDGET = 'config' but there is no lc_budget in the config")
    print("WARNING: lc_budget is in OpenLinTim cost units, not minutes")
else:
    budget_value = float(BUDGET)
print("Budget:", round(budget_value, 2))


# ---------------- network ----------------

G = nx.Graph()
G.add_nodes_from(stops)
for e_id, e in edges.items():
    u, v = e["left"], e["right"]
    if G.has_edge(u, v):
        # parallel edge, keep the faster one
        print("WARNING: parallel edges between", u, "and", v)
        if e["time"] >= G[u][v]["time"]:
            continue
    G.add_edge(u, v, edge_id=e_id, time=e["time"])

edge_id_by_pair = {frozenset((u, v)): d["edge_id"] for u, v, d in G.edges(data=True)}

if not nx.is_connected(G):
    print("WARNING: network is not connected")


# ---------------- candidate lines ----------------
# lines are simple paths. We build them by length (2 stops, then 3, ...) so if
# we hit MAX_CANDIDATE_LINES it is the longest lines that get cut.

print("Generating candidate lines...")

line_pool = {}    # line id -> edge ids
line_stops = {}   # line id -> stop ids
line_costs = {}   # line id -> cost (total travel time)
cap_reached = False


def add_line(path):
    line_id = len(line_pool) + 1
    line_pool[line_id] = [edge_id_by_pair[frozenset((a, b))] for a, b in zip(path[:-1], path[1:])]
    line_stops[line_id] = list(path)
    line_costs[line_id] = sum(G[a][b]["time"] for a, b in zip(path[:-1], path[1:]))


# all 2-stop paths, both directions
level = []
for u, v in G.edges():
    level.append([u, v])
    level.append([v, u])

n_stops = 2
while level and n_stops <= MAX_STOPS_PER_LINE and not cap_reached:
    if n_stops >= MIN_STOPS_PER_LINE:
        for path in sorted(level):
            # a path and its reverse are the same line, only keep one
            if tuple(path) < tuple(reversed(path)):
                if len(line_pool) >= MAX_CANDIDATE_LINES:
                    cap_reached = True
                    break
                add_line(path)
    # add one more stop at the end of every path
    new_level = []
    if n_stops < MAX_STOPS_PER_LINE and not cap_reached:
        for path in level:
            for nxt in G.neighbors(path[-1]):
                if nxt not in path:
                    new_level.append(path + [nxt])
    level = new_level
    n_stops += 1

print("Candidate lines:", len(line_pool))
if cap_reached:
    print("WARNING: MAX_CANDIDATE_LINES reached, longest lines were left out")

edge_to_lines = defaultdict(set)
for line_id, edge_list in line_pool.items():
    for e_id in edge_list:
        edge_to_lines[e_id].add(line_id)


# ---------------- shortest paths and candidate routes ----------------

print("Generating candidate routes...")

all_shortest = dict(nx.all_pairs_dijkstra_path_length(G, weight="time"))

# OD pairs with no path at all are dropped
unreachable = [od for od in demand if od[1] not in all_shortest[od[0]]]
if unreachable:
    print("WARNING: dropping", len(unreachable), "OD pairs without a path")
    for od in unreachable:
        del demand[od]
    total_demand = sum(demand.values())

shortest_time = {od: all_shortest[od[0]][od[1]] for od in demand}

candidate_routes = {}
for od in demand:
    limit = ALPHA * shortest_time[od]
    routes = []
    # paths come out from shortest to longest, so we can stop at the first one that is too long
    for path in nx.shortest_simple_paths(G, od[0], od[1], weight="time"):
        length = sum(G[a][b]["time"] for a, b in zip(path[:-1], path[1:]))
        if length > limit + 1e-9:
            break
        routes.append({
            "stops": list(path),
            "edges": [edge_id_by_pair[frozenset((a, b))] for a, b in zip(path[:-1], path[1:])],
            "length": length,
        })
        if len(routes) >= K_PATHS:
            break
    candidate_routes[od] = routes

print("Candidate routes:", sum(len(r) for r in candidate_routes.values()))


# ---------------- L(q): lines that contain a piece of path ----------------

cover_cache = {}


def lines_covering(edge_list):
    key = tuple(edge_list)
    if key in cover_cache:
        return cover_cache[key]
    if len(key) == 0:
        result = frozenset()
    else:
        sets = sorted((edge_to_lines[e] for e in key), key=len)
        common = set(sets[0])
        for s in sets[1:]:
            common &= s
            if not common:
                break
        result = frozenset(common)
    cover_cache[key] = result
    return result


print("Precomputing route info...")

route_info = {}
for od, routes in candidate_routes.items():
    for r, route in enumerate(routes):
        stop_seq = route["stops"]
        edge_seq = route["edges"]

        direct_lines = lines_covering(edge_seq)

        # one transfer at interior stop u: first part and second part both need a line
        transfer_options = {}
        for i in range(1, len(stop_seq) - 1):
            first = lines_covering(edge_seq[:i])
            second = lines_covering(edge_seq[i:])
            if first and second:
                transfer_options[stop_seq[i]] = {"first_lines": first, "second_lines": second}

        route_info[(od, r)] = {
            "edges": edge_seq,
            "length": route["length"],
            "direct_lines": direct_lines,
            "transfer_options": transfer_options,
        }


# ---------------- demand groups ----------------
# sort OD pairs by demand and cut into K groups. OD pairs with the same demand
# always stay together (as in the report).

def create_demand_groups(demand, K, scheme):
    if scheme not in ("count", "demand-share"):
        raise ValueError("GROUP_SCHEME must be 'count' or 'demand-share'")

    # blocks of equal demand
    sorted_ods = sorted(demand, key=lambda od: (demand[od], od))
    blocks = []
    for od in sorted_ods:
        if blocks and demand[blocks[-1][0]] == demand[od]:
            blocks[-1].append(od)
        else:
            blocks.append([od])
    if len(blocks) < K:
        raise ValueError("Only " + str(len(blocks)) + " distinct demand levels, cannot make " + str(K) + " groups")

    # weight of a block = number of OD pairs, or its total demand
    if scheme == "count":
        weight = [len(b) for b in blocks]
    else:
        weight = [sum(demand[od] for od in b) for b in blocks]
    total_weight = sum(weight)

    # cut as close as possible to 1/K, 2/K, ... of the total weight,
    # but always leave at least one block for each group still to come
    groups = []
    start = 0
    cumulative = 0.0
    for g in range(K - 1):
        target = total_weight * (g + 1) / K
        groups_left = K - g - 1
        best_end = None
        best_diff = float("inf")
        running = cumulative
        for end in range(start + 1, len(blocks) - groups_left + 1):
            running += weight[end - 1]
            if abs(running - target) < best_diff:
                best_diff = abs(running - target)
                best_end = end
        groups.append([od for b in blocks[start:best_end] for od in b])
        cumulative += sum(weight[start:best_end])
        start = best_end
    groups.append([od for b in blocks[start:] for od in b])
    return groups


demand_groups = create_demand_groups(demand, K_GROUPS, GROUP_SCHEME)

if K_GROUPS == 3:
    group_names = ["low", "medium", "high"]
else:
    group_names = ["#" + str(g + 1) for g in range(K_GROUPS)]

print()
print("Demand groups:")
for g, group in enumerate(demand_groups):
    print("  Group", g + 1, "(" + group_names[g] + "):", len(group), "OD pairs, demand",
          sum(demand[od] for od in group))

if RHO < 1:
    raise ValueError("RHO must be >= 1")

# w_st = d_st / (tau*_st * demand of the group), so that R_g = sum w_st * tau_st
group_weights = {}
for group in demand_groups:
    group_demand = sum(demand[od] for od in group)
    for od in group:
        group_weights[od] = demand[od] / (shortest_time[od] * group_demand)


# ---------------- model ----------------

print()
print("Building model...")

model = gp.Model("line_planning")

# y[l] = 1 if line l is operated
y = model.addVars(line_pool.keys(), vtype=GRB.BINARY, name="y")

# routing variables, only created when that option can exist at all
x0 = {}   # (od, r)     direct
x1 = {}   # (od, r, u)  one transfer at u
x2 = {}   # (od, r)     two or more transfers

for od, routes in candidate_routes.items():
    for r in range(len(routes)):
        info = route_info[(od, r)]
        if info["direct_lines"]:
            x0[(od, r)] = model.addVar(lb=0.0, ub=1.0, name="x0_%d_%d_%d" % (od[0], od[1], r))
        for u in info["transfer_options"]:
            x1[(od, r, u)] = model.addVar(lb=0.0, ub=1.0, name="x1_%d_%d_%d_%d" % (od[0], od[1], r, u))
        if all(edge_to_lines[e] for e in info["edges"]):
            x2[(od, r)] = model.addVar(lb=0.0, ub=1.0, name="x2_%d_%d_%d" % (od[0], od[1], r))

tau = model.addVars(demand.keys(), lb=0.0, name="tau")

# travel time and demand satisfaction for every OD pair
for od, routes in candidate_routes.items():
    time_terms = []
    share_terms = []
    for r in range(len(routes)):
        info = route_info[(od, r)]
        base = info["length"]
        if (od, r) in x0:
            time_terms.append(base * x0[(od, r)])
            share_terms.append(x0[(od, r)])
        for u in info["transfer_options"]:
            time_terms.append((base + TRANSFER_PENALTY) * x1[(od, r, u)])
            share_terms.append(x1[(od, r, u)])
        if (od, r) in x2:
            time_terms.append((base + 2 * TRANSFER_PENALTY) * x2[(od, r)])
            share_terms.append(x2[(od, r)])
    model.addConstr(tau[od] == gp.quicksum(time_terms), name="time_%d_%d" % od)
    model.addConstr(gp.quicksum(share_terms) == 1, name="demand_%d_%d" % od)

# an option can only be used if the selected lines allow it
for (od, r), x in x0.items():
    model.addConstr(x <= gp.quicksum(y[l] for l in route_info[(od, r)]["direct_lines"]))

for (od, r, u), x in x1.items():
    option = route_info[(od, r)]["transfer_options"][u]
    model.addConstr(x <= gp.quicksum(y[l] for l in option["first_lines"]))
    model.addConstr(x <= gp.quicksum(y[l] for l in option["second_lines"]))

for (od, r), x in x2.items():
    for e_id in route_info[(od, r)]["edges"]:
        model.addConstr(x <= gp.quicksum(y[l] for l in edge_to_lines[e_id]))

# budget and number of lines
model.addConstr(gp.quicksum(line_costs[l] * y[l] for l in line_pool) <= budget_value, name="budget")
model.addConstr(gp.quicksum(y[l] for l in line_pool) >= L_MIN, name="min_lines")
model.addConstr(gp.quicksum(y[l] for l in line_pool) <= L_MAX, name="max_lines")

# R_g <= rho for every group
for g, group in enumerate(demand_groups, start=1):
    model.addConstr(gp.quicksum(group_weights[od] * tau[od] for od in group) <= RHO,
                    name="group_cap_%d" % g)

# objective: average travel time
model.setObjective(gp.quicksum(demand[od] * tau[od] for od in demand) / total_demand, GRB.MINIMIZE)

model.Params.TimeLimit = TIME_LIMIT
if MIP_GAP is not None:
    model.Params.MIPGap = MIP_GAP
model.update()

print("Candidate lines:", len(line_pool), " OD pairs:", len(demand))
print("Groups:", K_GROUPS, GROUP_SCHEME, " RHO:", RHO)
print("Budget:", round(budget_value, 2), " Lines:", L_MIN, "to", L_MAX)
print("Transfer penalty:", TRANSFER_PENALTY, "min")
print("Variables:", model.NumVars, " Constraints:", model.NumConstrs)


# ---------------- solve ----------------

print()
start_time = time.time()
model.optimize()
runtime = time.time() - start_time

print()
print("Status:", model.Status, "(2 = optimal, 9 = time limit, 3 = infeasible)")
print("Runtime: %.1f s" % runtime)

if model.SolCount == 0:
    if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        print("Infeasible. Try a higher RHO, a bigger budget / L_MAX, or longer lines.")
    else:
        print("No solution found.")
    sys.exit()

print("Objective (average travel time): %.4f min" % model.ObjVal)
print("Best bound: %.4f" % model.ObjBound)
print("MIP gap: %.4f %%" % (100 * model.MIPGap))


# ---------------- results ----------------

def stop_label(s):
    return stops[s]["short_name"] or str(s)


selected_lines = [l for l in line_pool if y[l].X > 0.5]
total_cost = sum(line_costs[l] for l in selected_lines)

print()
print("Selected lines:", len(selected_lines))
for l in selected_lines:
    print("  line %d  cost %.2f  %s" % (l, line_costs[l], " - ".join(stop_label(s) for s in line_stops[l])))
print("Total cost: %.2f of budget %.2f" % (total_cost, budget_value))

if SHOW_OD_DETAILS:
    print()
    for od in demand:
        print("OD", od, "demand", demand[od], "time %.2f" % tau[od].X,
              "shortest %.2f" % shortest_time[od], "relative %.3f" % (tau[od].X / shortest_time[od]))

print()
print("Demand groups:")
for g, group in enumerate(demand_groups, start=1):
    group_demand = sum(demand[od] for od in group)
    R_g = sum(group_weights[od] * tau[od].X for od in group)
    mean_time = sum(demand[od] * tau[od].X for od in group) / group_demand
    delta_g = sum(demand[od] * (tau[od].X - shortest_time[od]) for od in group) / group_demand
    print()
    print("Group", g, "(" + group_names[g - 1] + ")")
    print("  OD pairs         :", len(group))
    print("  Total demand     :", group_demand)
    print("  Mean travel time : %.3f min" % mean_time)
    print("  R_g              : %.4f  (RHO = %s)" % (R_g, RHO))
    print("  Cap satisfied    :", R_g <= RHO + 1e-6)
    print("  Delta_g          : %.3f min" % delta_g)

direct = sum(demand[od] * x.X for (od, r), x in x0.items())
one_transfer = sum(demand[od] * x.X for (od, r, u), x in x1.items())
multi = sum(demand[od] * x.X for (od, r), x in x2.items())

print()
print("How passengers travel:")
print("  direct           : %.1f (%.1f %%)" % (direct, 100 * direct / total_demand))
print("  one transfer     : %.1f (%.1f %%)" % (one_transfer, 100 * one_transfer / total_demand))
print("  2+ transfers     : %.1f (%.1f %%)  (charged 2*pi)" % (multi, 100 * multi / total_demand))