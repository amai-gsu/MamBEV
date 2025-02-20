from itertools import combinations, combinations_with_replacement, permutations
from random import randint


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run 65k combinations")
    parser.addoption(
        "--global_only", action="store_true", help="run global traversals only"
    )


def pytest_generate_tests(metafunc):
    if "traversals" in metafunc.fixturenames:
        parts = [["t", "b"], ["l", "r"], ["0", "1"], ["snake", "cross"]]

        def trav_gen(i):
            mask = [int(bit) for bit in bin(i)[2:].zfill(4)]
            return "".join(a[b] for a, b in zip(parts, mask))

        if metafunc.config.getoption("all"):
            all_global = list(trav_gen(i) for i in range(2**4))
            max_size = -1
        else:
            all_global = [trav_gen(randint(0, 16)) for _ in range(5)]
            max_size = 10

        if metafunc.config.getoption("global_only"):
            combos = list(combinations(all_global, r=2))
        else:
            all_local = list(permutations(all_global, r=2))
            all_traversals = all_local + all_global
            combos = list(combinations_with_replacement(all_traversals, r=2))

        metafunc.parametrize("traversals", combos[:max_size])
