from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class DarkState:
    name: str
    isospin: Fraction
    hypercharge: int
    darkicity: int
    charge: int
    role: str

    @property
    def casimir_term(self) -> float:
        return float(Fraction(self.hypercharge ** 2, 4) - self.isospin * (self.isospin + 1))

    @property
    def theta(self) -> float:
        return 1.0 if self.charge > 0 else 0.0


OCTET = (
    DarkState("chi0", Fraction(3, 2), 1, 0, 0, "stable dark matter"),
    DarkState("chi+", Fraction(3, 2), 1, 0, 1, "decays to chi0 and phi+"),
    DarkState("chi-", Fraction(3, 2), 1, -1, -1, "decays to chi0 and phi-"),
    DarkState("chi++", Fraction(3, 2), 1, 1, 2, "decays to chi0 and two phi+"),
    DarkState("Sigma0", Fraction(1), 0, 0, 0, "decays to chi0 and phi0"),
    DarkState("Sigma-", Fraction(1), 0, -1, -1, "decays with CP-violating channel"),
    DarkState("Sigma+", Fraction(1), 0, 1, 1, "decays and contributes to eta_B"),
    DarkState("Lambda", Fraction(0), 0, 0, 0, "decays to chi0 and phi0"),
)

STATE_NAMES = tuple(s.name for s in OCTET)

_LOOKUP = {s.name: s for s in OCTET}


def get_state(name: str) -> DarkState:
    return _LOOKUP[name]
