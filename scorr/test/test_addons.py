#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import pytest

import scorr.addons


@pytest.mark.parametrize("size, n_elements, id_rank, expected", [
    (10, 100, 9, (90, 100)),
    (3, 100, 0, (0, 34)),
    (3, 100, 1, (34, 67)),
    (3, 100, 2, (67, 100)),
    (100, 10, 9, (9, 10)),
    (100, 10, 10, (10, 10)),
    (100, 10, 99, (10, 10)),
    (120, 900, 0, (0, 8)),
])
def test_make_rank_element_dictionary(size, n_elements, id_rank, expected):
    test_dict = scorr.addons.make_rank_element_dictionary(size)(n_elements)
    assert test_dict[id_rank] == expected
    for i in range(len(test_dict) - 1):
        assert test_dict[i][1] == test_dict[i + 1][0]
        assert test_dict[i][0] <= n_elements
        assert test_dict[i][1] <= n_elements
