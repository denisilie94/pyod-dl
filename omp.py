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

from sklearn.linear_model import OrthogonalMatchingPursuit


def omp(Y, D, n_nonzero_coefs, params=[]):
    """ Orthogonal Matching Pursuit wrapper """
    ompfun = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    X = ompfun.fit(D, Y).coef_.T
    err = (np.linalg.norm(Y - D @ X, 'fro') /
           np.sqrt(Y.size))
    return X, err
