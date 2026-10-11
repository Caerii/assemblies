"""A memory CELL: an area of n neurons, assemblies of k, connection probability p -- and what the
registrations derive from them.

    tau      the refraction's recovery time, by Amendment 41's measured rule: tau = n/k up to
             n/k ~ 50, tau = n/k / 2 from n/k ~ 100 (rho_50 then nearly constant across n/k).
             Between 50 and 100 the rule was not measured: the cell refuses to guess, and a
             registration there names its tau explicitly.
    beta     the learning rate theta(n, k, p) (laws.theta, the projection-convergence
             threshold), rounded to 5 places as every registration since has used it
    regime   k p >= 3 ln n: the recurrent in-degree the n/k laws need (AssemblyMemory.in_regime)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .laws import above_floor, theta


@dataclass(frozen=True)
class Cell:
    n: int
    k: int
    p: float
    tau: int

    @classmethod
    def of(cls, n, k, p, tau=None):
        """the cell, with tau by Amendment 41's rule unless given"""
        if tau is None:
            ratio = n / k
            if ratio >= 100:
                tau = round(n / (2 * k))
            elif ratio <= 50:
                tau = round(ratio)
            else:
                raise ValueError(f"n/k = {ratio:.0f}: Amendment 41's rule was measured up to 50 "
                                 "and from 100; name tau for this cell explicitly")
        return cls(int(n), int(k), float(p), int(tau))

    @property
    def ratio(self):
        return self.n / self.k

    @property
    def beta(self):
        return round(theta(self.n, self.k, self.p), 5)

    @property
    def in_regime(self):
        return above_floor(self.n, self.k, self.p)

    @property
    def key(self):
        """how a run record names it: "n/k/p" """
        return f"{self.n}/{self.k}/{self.p:g}"

    def as_tuple(self):
        return (self.n, self.k, self.p, self.tau)

    def spec(self, rho, **extra):
        """the per-cell parameters a registration records"""
        return {"n": self.n, "k": self.k, "p": self.p, "tau": self.tau, "rho": rho, "beta": self.beta, **extra}

    def describe(self):
        """the sentence a registration's protocol states about the cell"""
        return (f"({self.n}, {self.k}, {self.p:g}) (n/k = {self.ratio:.0f}, tau = {self.tau}); "
                f"k p = {self.k * self.p:.1f} against 3 ln n = {3 * math.log(self.n):.1f}")
