import pstats
from pstats import SortKey
p_calls = pstats.Stats('profile_vae')
p_tottime = pstats.Stats('profile_vae_tottime')
p_cumtime = pstats.Stats('profile_vae_cumtime')

p_calls.strip_dirs().sort_stats(SortKey.CALLS).print_stats(10)
p_tottime.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
p_cumtime.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)
