'''
File implementing the niggli reduction, taking in a dict of cell parameters
and returning a dict of the reduced cell parameters. Naming convention follows
the steps defined by "Unified Algorithm for Determining the Reduced (Niggli)
Cell" by I. KRIVY and B. GRUBER - 1975. Refactoring of original code by Cristian
Badoi
'''

from math import pow, cos, acos, degrees, radians, sqrt

class Niggli():
    def __init__(self,
                 lattice,
                 epsilon = 0.000001, # A little perturbation for when a = b
                 max_loops = 50):
        self.lattice = lattice
        self.epsilon = epsilon
        self.max_loops = max_loops
        self.niggli_lattice = self.initialise_niggli(lattice)
        self.niggli_transform()

    def initialise_niggli(self, lattice):
        niggli_lattice = {}

        niggli_lattice['A'] = pow(lattice['a'], 2)
        niggli_lattice['B'] = pow(lattice['b'], 2)
        niggli_lattice['C'] = pow(lattice['c'], 2)

        niggli_lattice['xi'] = 2 * lattice['b'] * lattice['c'] * cos(radians(lattice['alpha']))
        niggli_lattice['eta'] = 2 * lattice['a'] * lattice['c'] * cos(radians(lattice['beta']))
        niggli_lattice['zeta'] = 2 * lattice['a'] * lattice['b'] * cos(radians(lattice['gamma']))

        return niggli_lattice

    def niggli_transform(self):
        for i in range(self.max_loops):
            self.step_1()

            if self.step_2():
                pass
            else:
                continue

            self.step_3()
            self.step_4()

            if self.step_5():
                pass
            else:
                continue

            if self.step_6():
                pass
            else:
                continue

            if self.step_7():
                pass
            else:
                continue

            if self.step_8():
                break

        self.norm_niggli()

    '''
    If we satisfy the first condition of the algorithm swap A/B and xi/eta
    '''
    def step_1(self):
        nl = self.niggli_lattice

        if (nl['A'] > nl['B'] + self.epsilon
            or not abs(nl['A'] - nl['B']) > self.epsilon
            and abs(nl['xi']) > nl['eta'] + self.epsilon):

            nl['A'], nl['B'] = nl['B'], nl['A']
            nl['xi'], nl['eta'] = nl['eta'], nl['xi']

    '''
    If we satisfy the second condition of the algorithm swap B/C and eta/zeta
    and repeat step 1
    '''
    def step_2(self):
        nl = self.niggli_lattice

        if (nl['B'] > nl['C'] + self.epsilon
            or not abs(nl['B'] - nl['C']) > self.epsilon
            and abs(nl['eta']) > (nl['zeta'] + self.epsilon)):

            nl['B'], nl['C'] = nl['C'], nl['B']
            nl['eta'], nl['zeta'] = nl['zeta'], nl['eta']

            return False

        else:
            return True

    def step_3(self):
        nl = self.niggli_lattice

        if(nl['xi'] * nl['eta'] * nl['zeta'] > 0):
            nl['xi'] = abs(nl['xi'])
            nl['eta'] = abs(nl['eta'])
            nl['zeta'] = abs(nl['zeta'])

    def step_4(self):
        nl = self.niggli_lattice

        if(nl['xi'] * nl['eta'] * nl['zeta'] <= 0):

            nl['xi'] = -abs(nl['xi'])
            nl['eta'] = -abs(nl['eta'])
            nl['zeta'] = -abs(nl['zeta'])

    def step_5(self):
        nl = self.niggli_lattice

        if( abs(nl['xi']) > nl['B'] + self.epsilon
            or not abs(nl['B'] - nl['xi']) > self.epsilon
            and 2 * nl['eta'] < nl['zeta'] - self.epsilon
            or not abs(nl['B'] + nl['xi']) > self.epsilon
            and nl['zeta'] < -self.epsilon):

            nl['C'] = nl['B'] + nl['C'] - nl['xi'] * copysign(1, nl['xi'])
            nl['eta'] = nl['eta'] * copysign(1, nl['xi'])
            nl['xi'] = nl['xi'] - 2 * nl['B'] * copysign(1, nl['xi'])

            return False

        else:
            return True

    def step_6(self):
        nl = self.niggli_lattice

        if( abs(nl['eta']) > nl['A'] + self.epsilon
            or not abs(nl['A'] - nl['eta']) > self.epsilon
            and 2 * nl['xi'] < nl['zeta'] - self.epsilon
            or not abs(nl['A'] + nl['eta']) > self.epsilon
            and nl['zeta'] < -self.epsilon):

            nl['C'] = nl['A'] + nl['C'] - nl['eta'] * copysign(1, nl['eta'])
            nl['xi'] = nl['xi'] - nl['zeta'] * copysign(1, nl['eta'])
            nl['eta'] = nl['eta'] - 2 * nl['A'] * copysign(1, nl['eta'])

            return False

        else:
            return True

    def step_7(self):
        nl = self.niggli_lattice

        if( abs(nl['zeta']) > nl['A'] + self.epsilon
            or not abs(nl['A'] - nl['zeta']) > self.epsilon
            and 2 * nl['xi'] < nl['eta'] - self.epsilon
            or not abs(nl['A'] + nl['zeta']) > self.epsilon
            and nl['eta'] < -self.epsilon):

            nl['B'] = nl['A'] + nl['B'] - nl['zeta'] * copysign(1, nl['zeta'])
            nl['xi'] = nl['xi'] - nl['eta'] * copysign(1, nl['zeta'])
            nl['zeta'] = nl['zeta'] - 2 * nl['A'] * copysign(1, nl['zeta'])

            return False

        else:
            return True

    def step_8(self):
        nl = self.niggli_lattice

        if(nl['xi'] + nl['eta'] + nl['zeta'] + nl['A'] + nl['B'] < -self.epsilon
           or not abs(nl['xi'] + nl['eta'] + nl['zeta'] + nl['A'] + nl['B']) > self.epsilon
           and 2 * (nl['A'] + nl['eta']) + nl['zeta'] > self.epsilon):

            nl['C'] = nl['A'] + nl['C'] + nl['xi'] + nl['eta'] + nl['zeta']
            nl['xi'] = 2 * nl['B'] + nl['xi'] + nl['zeta']
            nl['eta'] = 2 * nl['A'] + nl['eta'] + nl['zeta']

            return False

        else:
            return True

    def norm_niggli(self):
        nl = self.niggli_lattice

        nl['A'] = sqrt(nl['A'])
        nl['B'] = sqrt(nl['B'])
        nl['C'] = sqrt(nl['C'])

        nl['xi'] = acos(degrees(nl['xi'] / (nl['B'] * nl['C'])))
        nl['eta'] = acos(degrees(nl['eta'] / (nl['A'] * nl['C'])))
        nl['zeta'] = acos(degrees(nl['zeta'] / (nl['A'] * nl['B'])))
