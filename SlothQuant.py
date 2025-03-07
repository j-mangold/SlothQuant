import scipy as scp
import numpy as np
import json
from itertools import product, combinations
import matplotlib.pyplot as plt
import time



z_lut = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10}

class molecule:

    def __init__(self, atoms, positions, z_lut=z_lut):
        self.Zs = [z_lut[atom] for atom in atoms]
        self.Atoms = [x for x in zip(atoms, positions, self.Zs)]
        self.Basis = {'pos': [], 'els': [], 'exp': [], 'coef': [], 'ns': [], 'ls': [], 'zs': []}
        self.k = 0
        for Atom, Position, Z in self.Atoms:
            self.k += Z
            with open('Basissets/{}_STO-3G.json'.format(Atom), 'r') as file:
                basis = json.load(file)
                for n, shell in enumerate(basis['elements'][str(Z)]['electron_shells']):
                    exponents = np.array(shell['exponents']).astype(float)
                    for l in shell['angular_momentum']:
                        coefficients = np.array(shell['coefficients'][l]).astype(float)
                        coefficients = (2 * exponents / np.sqrt(np.pi)) ** 0.75 * coefficients
                        self.Basis['pos'].append(Position)
                        self.Basis['els'].append(Atom)
                        self.Basis['exp'].append(exponents)
                        self.Basis['coef'].append(coefficients)
                        self.Basis['ns'].append(n)
                        self.Basis['ls'].append(l)
                        self.Basis['zs'].append(Z)
        self.k = int(0.5 * self.k)
        
    def S_ab(self, a, b, ca, cb, A, B):
        p = a + b
        g = a * b / (a + b)
        X_AB = np.linalg.norm(A - B)
        return ca * cb * (np.pi / p) ** 1.5 * np.exp(- g * X_AB ** 2)

    def T_ab(self, a, b, ca, cb, A, B):
        p = a + b
        g = a * b / (a + b)
        X_AB = np.linalg.norm(A - B)
        S_00 = ca * cb * (np.pi / p) ** 1.5 * np.exp(- g * X_AB ** 2) 
        return 0.5 * g * (6 - 4 * g * X_AB ** 2) * S_00

    def V_ab(self, a, b, ca, cb, A, B, C, Z):
        p = a + b
        R_P = (a * A + b * B)/ p
        X_PC = np.linalg.norm(R_P - C)
        X_AB = np.linalg.norm(A - B)
        t = p * X_PC ** 2
        S_00 = ca * cb * (np.pi / p) ** 1.5 * np.exp(- a * b / p * X_AB ** 2)
        if np.isclose(X_PC, 0): return - 2 * Z * S_00 * np.sqrt(p / np.pi)
        else: return - Z * S_00 * np.sqrt(p) / np.sqrt(t) * scp.special.erf(np.sqrt(t))

    def Q_acbd(self, a, c, b, d, ca, cc, cb, cd, A, C, B, D, exchange = False):
        prefactor = ca * cb * cc * cd
        if exchange:
            b_copy = b
            b = d
            d = b_copy
            B_copy = B
            B = D
            D = B_copy
        g1 = a * b / (a + b)
        g2 = c * d / (c + d)
        p1 = a + b
        p2 = c + d
        R1 = (a * A + b * B) / p1
        R2 = (c * C + d * D) / p2
        t = p1 * p2 / (p1 + p2) * np.linalg.norm(R1 - R2) ** 2
        res_0 = prefactor *  2 * np.pi ** 2.5 / (p1 * p2 * np.sqrt(p1 + p2))  * np.exp(- g1 * np.linalg.norm(A-B) ** 2 - g2 * np.linalg.norm(C-D) ** 2)
        if np.isclose(t, 0): return res_0
        else: return res_0  *  np.sqrt(np.pi) * 0.5 * scp.special.erf(np.sqrt(t)) / np.sqrt(t)

    def contractor(self, function, d, **kwargs):
        es = np.array(kwargs['es'])
        cs = np.array(kwargs['cs'])
        pos = np.array(kwargs['pos'])
        ranges = [range(len(e)) for e in es]
        array = np.array([function(*[es[i][y] for i, y in zip(range(d), ys)], *[cs[i][y] for i, y in zip(range(d) ,ys)], *pos, *kwargs['putz']) for ys in product(*ranges)])
        return np.sum(array)

    def matrix(self, function, d, *args):
        n = len(self.Basis['exp'])
        shape = np.ones(d) * n
        array = np.array([self.contractor(function, d, **{'es': [self.Basis['exp'][x] for x in xs],
                                                          'cs': [self.Basis['coef'][x] for x in xs], 
                                                          'pos': [self.Basis['pos'][x] for x in xs], 
                                                          'putz': [*args]}
                                                          ) for xs in product(range(n), repeat = d)]).reshape(*shape.astype(int))
        return array

    
    def compute_matrix_elements(self):
        self.S = self.matrix(self.S_ab, 2, *[])
        self.T = self.matrix(self.T_ab, 2, *[])
        self.Vs = []
        positions, indices = np.unique(self.Basis['pos'], axis=0, return_index = True)
        for pos, Z in zip(positions, np.array(self.Basis['zs'])[indices]):
            self.Vs.append(self.matrix(self.V_ab, 2, *[pos, Z]))
        self.D = self.matrix(self.Q_acbd, 4, *[False])
        self.Ex = self.matrix(self.Q_acbd, 4, *[True])


    def initialize_C(self):
        self.C = np.zeros((len(self.Basis['exp']), self.k))
        return 0
        
    def h_ab(self):
        self.h = self.T + np.sum(np.array([V for V in self.Vs]), axis=0)
        return 0

    def P_ab(self):
        self.P = 2 * np.einsum('ak,bk->ab', self.C, self.C)

        return 0

    def F_ab(self):
        self.F = self.h + np.einsum('cd,acbd->ab', self.P, self.D) - 0.5 * np.einsum('cd,acbd->ab', self.P, self.Ex)
        return 0

    def scf(self, cutoff = 1e-6):
        deltae = 1
        self.initialize_C()
        self.es = [np.zeros(self.k)]
        self.h_ab()
        self.P_ab()
        while deltae > cutoff:
            self.F_ab()
            Es, Cs = scp.linalg.eigh(self.F, b=self.S)
            self.C = Cs[:,0:self.k]
            print(self.C)
            self.P_ab()
            self.es.append(Es[0:self.k])
            deltae = abs(self.es[-2][0] - self.es[-1][0])
            print('Delta E: {:02E}'.format(deltae))

    def energy(self):
        self.Etot = 0.5  * np.einsum('pq,pq->', self.h, self.P) + sum(self.es[-1])
        positions, indices = np.unique(self.Basis['pos'], axis=0, return_index = True)
        for pos_pair, Z_pair in zip(combinations(positions, 2), combinations(np.array(self.Basis['zs'])[indices], 2)): self.Etot += Z_pair[0] * Z_pair[1] / np.linalg.norm(pos_pair[0] - pos_pair[1])
        return 0




atoms = ['H','H']
rs = np.arange(2, 4, 0.05)
Positions = []
Energies = []
for r in rs:
    Positions.append([[0,0,0], [r,0,0]])

start = time.time()

for positions in Positions:
    H2 = molecule(atoms, positions)
    H2.compute_matrix_elements()
    H2.scf()
    H2.energy()
    Energies.append(H2.Etot)
    print(H2.Etot)

end = time.time()
print(end-start)
