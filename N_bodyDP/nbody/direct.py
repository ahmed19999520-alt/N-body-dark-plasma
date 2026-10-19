import numpy as np

from .units import G_KPC


def direct_accelerations(pos, mass, eps: float, g: float = G_KPC, chunk: int = 256):
    n = pos.shape[0]
    acc = np.zeros_like(pos)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = pos[None, :, :] - pos[start:stop, None, :]
        r2 = np.einsum("ijk,ijk->ij", d, d) + eps ** 2
        inv = r2 ** -1.5
        acc[start:stop] = g * np.einsum("ij,ij,ijk->ik", np.broadcast_to(mass, inv.shape), inv, d)
    return acc


def direct_potential_energy(pos, mass, eps: float, g: float = G_KPC, chunk: int = 256):
    n = pos.shape[0]
    total = 0.0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = pos[None, :, :] - pos[start:stop, None, :]
        r2 = np.einsum("ijk,ijk->ij", d, d) + eps ** 2
        pair = mass[start:stop, None] * mass[None, :] / np.sqrt(r2)
        idx = np.arange(start, stop)
        pair[np.arange(stop - start), idx] = 0.0
        total += pair.sum()
    return -0.5 * g * total
