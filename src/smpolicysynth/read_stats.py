import pstats 

p = pstats.Stats('restats')
p.sort_stats('time').print_stats(100)
