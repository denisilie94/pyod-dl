# Copyright (c) 2019 Paul Irofti <paul@irofti.net>
# Copyright (c) 2020 Denis Ilie-Ablachim <denis.ilie_ablachim@acse.pub.ro>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np


class ReplAtoms():
    '''
    Unused dictionary atoms replacement strategy
        ZERO - return a zero column
        RANDOM - return a random generated atom
        NO - perform no replacement
        WORST - replace ith the worst represented signal
    '''
    ZERO = 0
    RANDOM = 1
    NO = 2
    WORST = 3


def _new_atom(Y, D, X, atom_index, replatoms):
    '''
    Replace unused atom j
    INPUTS:
      replatoms -- unused dictionary atoms replacement strategy
        ZERO - return a zero column
        RANDOM - return a random generated atom
        NO - perform no replacement
        WORST - replace ith the worst represented signal
      Y -- training signals set
      D -- current dictionary
      X -- sparse representations
      j -- atom's column index in the dictionary D
    OUTPUTS:
      atom -- replacement atom
    '''
    if replatoms == ReplAtoms.ZERO:
        # Return a zero column
        return np.zeros(D.shape[0])

    if replatoms == ReplAtoms.RANDOM:
        # Return a random generated atom
        atom = np.random.rand(D.shape[0])
        atom = atom / np.linalg.norm(atom)
        return atom

    if replatoms == ReplAtoms.NO:
        # Perform no replacement
        return D[:, atom_index]

    if replatoms == ReplAtoms.WORST:
        # Replace with the worst represented signal
        E = Y - D @ X
        index = np.argmax(np.linalg.norm(E, axis=0))
        return Y[:, index] / np.linalg.norm(Y[:, index])
