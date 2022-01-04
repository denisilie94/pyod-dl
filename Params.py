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

from _atom import ReplAtoms


class Params():
    # atoms with square norm below are considered zero
    atom_norm_tolerance = 1e-10

    # default replace atom method
    replatoms = ReplAtoms.RANDOM

    # icoherent parameter
    tau = 0.01
    
    # kdl parameter
    alpha = 1e-6
    
    # rbf kernel parameter
    gamma_rbf = 0.1
    
    # poly kernel parameter
    gamma_poly = 0.1
    coef0_poly = 1
    degree_poly = 3

    def _safe_init(self, method):
        pass

    def _safe_update(self, iter):
        pass
