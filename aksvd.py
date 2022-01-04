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

from _atom import _new_atom, _update_atom, _update_atom_ker


def _aksvd_incoh_anom_update_atom(F, D, X, atom_index, atom_usages, params, Yc):
    d = F @ X[atom_index, atom_usages].T - 2 * params.tau * (Yc @ Yc.T) @ D[:, atom_index]
    d_norm = np.linalg.norm(d)
    if d_norm >= params.atom_norm_tolerance:
        d /= d_norm
    x = F.T @ d
    return d, x


def _aksvd_update_atom(F, D, X, atom_index, atom_usages, params):
    d = F @ X[atom_index, atom_usages].T
    d_norm = np.linalg.norm(d)
    if d_norm >= params.atom_norm_tolerance:
        d /= d_norm
    x = F.T @ d
    return d, x


def aksvd_incoh_anom(Y, Yc, D, X, params):
    '''
    Approximate K-SVD
    INPUTS:
        Y -- training signals set
        D -- current dictionary
        X -- sparse representations
    OUTPUTS:
        D -- updated dictionary
    '''
    E = Y - D @ X
    for atom_index in range(D.shape[1]):
        atom_usages = np.nonzero(X[atom_index, :])[0]

        if len(atom_usages) == 0:
            # replace with the new atom
            atom = _new_atom(Y, D, X, atom_index, params.replatoms)
            D[:, atom_index] = atom
            continue
        else:
            F = (E[:, atom_usages] +
                 np.outer(D[:, atom_index], X[atom_index, atom_usages].T))
            atom, atom_codes = _aksvd_incoh_anom_update_atom(F, D, X, atom_index,
                                                             atom_usages, params, Yc)

            D[:, atom_index] = atom
            X[atom_index, atom_usages] = atom_codes
            E[:, atom_usages] = (F - np.outer(D[:, atom_index],
                                              X[atom_index, atom_usages]))

    return D, X


def aksvd(Y, D, X, params):
    '''
    Approximate K-SVD
    INPUTS:
        Y -- training signals set
        D -- current dictionary
        X -- sparse representations
    OUTPUTS:
        D -- updated dictionary
    '''
    D, X = _update_atom(Y, D, X, params, _aksvd_update_atom)
    return D, X


def ker_aksvd_anom(K_bar, K_hat, A, X, params):
    S = (A @ X).T
    for atom_index in range(A.shape[1]):
        R = S - np.outer(X[atom_index, :].T, A[:, atom_index].T)
        atom_usages = np.nonzero(X[atom_index, :])[0]

        if len(atom_usages) == 0:
            # replace with the new atom
            atom = np.random.rand(A.shape[0])
            A[:, atom_index] = atom / np.sqrt(atom.T @ K_bar @ atom)
        else:
            a = np.linalg.solve(X[atom_index, :] @ X[atom_index, :].T * K_bar +
                                params.alpha * np.eye(K_bar.shape[0]),
                                (K_hat.T + K_bar @ R.T) @ X[atom_index, :].T)
            a_norm = np.linalg.norm(a)
            if a_norm >= params.atom_norm_tolerance:
                a /= np.sqrt(a.T @ K_bar @ a)
            x = (K_hat - R @ K_bar) @ a

            A[:, atom_index] = a
            X[atom_index, :] = x

    return A, X


# def ker_aksvd_anom(K_bar, K_hat, A, X, params):
#     for atom_index in range(A.shape[1]):
#         atom_usages = np.nonzero(X[atom_index, :])[0]

#         if len(atom_usages) == 0:
#             # replace with the new atom
#             atom = np.random.rand(A.shape[0])
#             A[:, atom_index] = atom / np.sqrt(atom.T @ K_bar @ atom)
#         else:
#             a = np.linalg.solve(X[atom_index, :] @ X[atom_index, :].T * K_bar +
#                                 params.alpha * np.eye(K_bar.shape[0]),
#                                 K_hat.T @ X[atom_index, :].T)
#             a_norm = np.linalg.norm(a)
#             if a_norm >= params.atom_norm_tolerance:
#                 a /= np.sqrt(a.T @ K_bar @ a)
#             x = K_hat @ a

#             A[:, atom_index] = a
#             X[atom_index, :] = x

#     return A, X
