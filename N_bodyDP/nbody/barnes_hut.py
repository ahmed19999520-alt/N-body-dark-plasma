
import numpy as np

from .units import G_KPC


class Octree:
    def __init__(self, pos, mass, leaf_size: int = 8, max_depth: int = 40):
        self.pos = pos
        self.mass = mass
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.node_mass = []
        self.node_com = []
        self.node_size = []
        self.children = []
        self.leaf = []
        lo = pos.min(axis=0)
        hi = pos.max(axis=0)
        center = 0.5 * (lo + hi)
        half = 0.5 * float((hi - lo).max()) * 1.0001 + 1.0e-12
        self._build(np.arange(pos.shape[0]), center, half, 0)
        self.node_mass = np.array(self.node_mass)
        self.node_com = np.array(self.node_com)
        self.node_size = np.array(self.node_size)

    def _build(self, idx, center, half, depth):
        nid = len(self.node_mass)
        m = self.mass[idx]
        total = m.sum()
        com = (self.pos[idx] * m[:, None]).sum(axis=0) / total
        self.node_mass.append(total)
        self.node_com.append(com)
        self.node_size.append(2.0 * half)
        self.children.append(None)
        self.leaf.append(None)
        if idx.size <= self.leaf_size or depth >= self.max_depth:
            self.leaf[nid] = idx
            return nid
        octant = ((self.pos[idx] > center) * np.array([1, 2, 4])).sum(axis=1)
        kids = []
        for o in range(8):
            sel = idx[octant == o]
            if sel.size:
                off = np.array([(o & 1) * 2 - 1, ((o >> 1) & 1) * 2 - 1, ((o >> 2) & 1) * 2 - 1]) * half / 2.0
                kids.append(self._build(sel, center + off, half / 2.0, depth + 1))
        self.children[nid] = kids
        return nid

    def acceleration(self, i, theta, eps, g):
        p = self.pos[i]
        a = np.zeros(3)
        stack = [0]
        eps2 = eps * eps
        while stack:
            n = stack.pop()
            leaf = self.leaf[n]
            if leaf is not None:
                sel = leaf[leaf != i]
                if sel.size:
                    d = self.pos[sel] - p
                    r2 = np.einsum("ij,ij->i", d, d) + eps2
                    a += (self.mass[sel] * r2 ** -1.5) @ d
                continue
            d = self.node_com[n] - p
            r2 = float(d @ d)
            if self.node_size[n] ** 2 < theta * theta * r2:
                a += self.node_mass[n] * (r2 + eps2) ** -1.5 * d
            else:
                stack.extend(self.children[n])
        return g * a


def barnes_hut_accelerations(pos, mass, eps: float, theta: float = 0.5, g: float = G_KPC, leaf_size: int = 8):
    tree = Octree(pos, mass, leaf_size=leaf_size)
    out = np.empty_like(pos)
    for i in range(pos.shape[0]):
        out[i] = tree.acceleration(i, theta, eps, g)
    return out
