"""Conservative graph-to-instance grounding.

No numeric suffix or observation-dependent relation is identity evidence.
Similarity margins and geometric tolerances control match acceptance.
"""
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

PERSISTENT = frozenset(('on', 'under', 'above', 'below', 'adjacent to', 'inside',
                       'contains', 'against', 'attached to', 'facing',
                       'surrounding', 'surrounded by'))
OBSERVATION = frozenset(('in front of', 'behind', 'partially occluding', 'partially occluded by'))
BACKGROUND = frozenset(('wall', 'floor', 'ceiling', 'sky', 'ground', 'terrain',
                        'vegetation', 'distant_terrain', 'distant_vegetation'))


def category(node):
    return re.sub(r'_[0-9]+$', '', node)


def grounding_text(graph, node):
    return ' '.join(graph['objects'][node]['attributes'] + [category(node).replace('_', ' ')])


def validate_graph(graph):
    if set(graph) != {'objects', 'ranking', 'major'}:
        raise ValueError('Scene graph requires exactly objects, ranking, major; regenerate using gpt_prompt.txt')
    objects, ranking, major = (graph[k] for k in ('objects', 'ranking', 'major'))
    if not isinstance(objects, dict) or not isinstance(ranking, list) or not isinstance(major, list):
        raise ValueError('Invalid scene-graph field types')
    if len(set(ranking)) != len(ranking) or any(n not in objects for n in ranking):
        raise ValueError('Ranking must contain distinct existing semantic node IDs')
    if major != ranking[:len(major)] or len(major) > 10:
        raise ValueError('Major must be a prefix of ranking with at most ten entries')
    if any(category(n) in BACKGROUND for n in ranking):
        raise ValueError('Diffuse background nodes cannot be anchors')
    for node, data in objects.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', node) or set(data) != {'attributes', 'relations'}:
            raise ValueError(f'Invalid semantic node: {node}')
        if not isinstance(data['attributes'], list) or any(not isinstance(a, str) for a in data['attributes']):
            raise ValueError(f'Attributes must be strings: {node}')
        seen = set()
        for edge in data['relations']:
            if set(edge) != {'target', 'relation'}:
                raise ValueError('Each relation requires exactly target and relation')
            target, label = edge['target'], edge['relation']
            if target not in objects or target == node or label not in PERSISTENT | OBSERVATION:
                raise ValueError(f'Invalid edge: {node} {label} {target}')
            if label == 'adjacent to' and node >= target:
                raise ValueError('Adjacent pairs must be serialized once in lexicographic order')
            if (target, label) in seen:
                raise ValueError('Duplicate relation')
            seen.add((target, label))
    return graph


def persistent_edges(graph):
    return [(src, e['relation'], e['target']) for src, data in graph['objects'].items()
            for e in data['relations'] if e['relation'] in PERSISTENT]


def mask_geometry(points, mask):
    p = points[np.asarray(mask, dtype=bool)]
    p = p[np.isfinite(p).all(1) & (np.linalg.norm(p, axis=1) > 1e-5)]
    if len(p) < 3:
        return None
    return {'min': p.min(0), 'max': p.max(0), 'center': p.mean(0),
            'points': p[::max(1, len(p)//4096)]}


def verify_relation(label, a, b, relative_tolerance=.05):
    """True = verified, False = contradicted, None = insufficient geometry.

    Visible surface bounds cannot establish closed-volume containment, front
    orientation or attachment. Such edges remain unknown rather than being
    promoted to persistent facts based on an AABB overlap alone.
    """
    if a is None or b is None:
        return None
    amin, amax, bmin, bmax = a['min'], a['max'], b['min'], b['max']
    scale = max(np.linalg.norm(amax-amin), np.linalg.norm(bmax-bmin), 1e-6)
    tol = relative_tolerance*scale
    overlap = np.minimum(amax, bmax)-np.maximum(amin, bmin)
    if label == 'above':
        return True if amin[1] > bmax[1]+tol else (False if amax[1] < bmin[1]-tol else None)
    if label == 'below':
        return verify_relation('above', b, a, relative_tolerance)
    if label == 'on':
        contact = cKDTree(b['points']).query(a['points'])[0].min()
        if abs(amin[1]-bmax[1]) <= tol and np.all(overlap[[0, 2]] > tol) and contact <= tol:
            return True
        if amax[1] < bmin[1]-tol:
            return False
        return None
    if label == 'under':
        if amax[1] < bmin[1]-tol and np.all(overlap[[0, 2]] > tol):
            return True
        if amin[1] > bmax[1]+tol:
            return False
        return None
    if label == 'adjacent to':
        gap = np.maximum(np.maximum(amin-bmax, bmin-amax), 0)
        if np.linalg.norm(gap) > .5*scale:
            return False
        contact = cKDTree(b['points']).query(a['points'])[0].min()
        return True if contact <= .2*scale else None
    if label == 'against':
        contact = cKDTree(b['points']).query(a['points'])[0].min()
        return True if contact <= tol and overlap[1] > tol else None
    # For these relations single-view visible geometry is generally incomplete.
    return None


def ground_graph(graph, similarities, geometries, min_similarity=.20, ambiguity_margin=.02):
    """Iterated global one-to-one matching with confirmed geometric context.

    An edge is committed only if forbidding it reduces the global optimum by
    more than the ambiguity margin. Equal optima remain unresolved. Private
    dummy columns permit discarding nodes rather than forcing a bad match.
    """
    validate_graph(graph)
    nodes = list(graph['objects'])
    scores = np.asarray(similarities, dtype=float)
    q = len(geometries)
    if scores.shape != (len(nodes), q):
        raise ValueError('Similarity matrix must be graph nodes x candidate masks')
    if not nodes or not q:
        return {}, []
    feasible = np.isfinite(scores) & (scores >= min_similarity)
    feasible[:, [i for i, g in enumerate(geometries) if g is None]] = False
    # Retain semantically competitive physical candidates, not a softmax class ID.
    feasible &= scores >= scores.max(axis=1, keepdims=True)-ambiguity_margin
    edges = persistent_edges(graph)
    bindings = {}
    while True:
        pending = [n for n in nodes if n not in bindings]
        available = [j for j in range(q) if j not in bindings.values()]
        if not pending or not available:
            break
        weights = np.full((len(pending), len(available)+len(pending)), -1e6)
        weights[:, len(available):] = min_similarity
        for i, node in enumerate(pending):
            constraints = []
            for src, label, dst in edges:
                if src == node and dst in bindings:
                    verdicts = {c: verify_relation(label, geometries[c], geometries[bindings[dst]])
                                for c in available if feasible[nodes.index(node), c]}
                elif dst == node and src in bindings:
                    verdicts = {c: verify_relation(label, geometries[bindings[src]], geometries[c])
                                for c in available if feasible[nodes.index(node), c]}
                else:
                    continue
                # A VLM edge with no verified physical realization is ignored.
                if any(v is True for v in verdicts.values()):
                    constraints.append(verdicts)
            for j, candidate in enumerate(available):
                if not feasible[nodes.index(node), candidate]:
                    continue
                possible = all(verdicts.get(candidate) is not False for verdicts in constraints)
                if possible:
                    weights[i, j] = scores[nodes.index(node), candidate]
        rows, cols = linear_sum_assignment(-weights)
        optimum = weights[rows, cols].sum()
        resolved = {}
        for i, j in zip(rows, cols):
            if j >= len(available) or weights[i, j] <= min_similarity:
                continue
            alternate = weights.copy()
            alternate[i, j] = -1e6
            rr, cc = linear_sum_assignment(-alternate)
            if optimum-alternate[rr, cc].sum() > ambiguity_margin:
                resolved[pending[i]] = available[j]
        if not resolved:
            break
        bindings.update(resolved)
    verified = [(s, r, t) for s, r, t in edges if s in bindings and t in bindings
                and verify_relation(r, geometries[bindings[s]], geometries[bindings[t]]) is True]
    return bindings, verified


def guidance_text(graph, focus, bindings, verified):
    if focus is None or focus not in bindings:
        return ''
    sentences = [grounding_text(graph, focus)+'.']
    for src, label, dst in verified:
        if focus in (src, dst) and src in bindings and dst in bindings and label in PERSISTENT:
            sentences.append(f'{grounding_text(graph, src)} {label} {grounding_text(graph, dst)}.')
    return ' '.join(sentences)
