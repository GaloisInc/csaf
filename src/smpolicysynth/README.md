## Synthesizing Programmatic Policies that Inductively Generalize

# Dependencies 

Python 3

Numpy

Matplotlib

Scipy

Pyopt (Follow instructions from https://github.com/madebr/pyOpt
 	
OpenAI Gym

MuJoCo (optinal)

# Training policies 


  ```python -m synth.main.synth BENCH RUNID```
  
  where BENCH can be one of car, pen, quad, quadpo, cp, acrobot, mc, and swimmer (requires MuJoCo)
  
  RUNID is an integer
  
  The outputs will be stored in the directory out/BENCH_RUNID/


# Visualizing trained policies 

```python -m synth.main.run BENCH INP_LIMITS SM out/BENCH_RUNID/sm_min.txt```

where BENCH AND RUNID are as above, INP_LIMITS = [] by default. 
