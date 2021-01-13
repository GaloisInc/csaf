from run_parallel import run_workgroup
import csaf.config as cconf
import csaf.system as csys
from IPython.display import Image

model_conf = cconf.SystemConfig.from_toml("../examples/f16/f16_sensor_noise.toml")
import pathlib

plot_fname = f"noisy-airspeed-sub-plot.png"

# plot configuration pub/sub diagram as a file -- proj specicies a dot executbale and -Gdpi is a valid dot
# argument to change the image resolution
model_conf.plot_config(fname=pathlib.Path(plot_fname).resolve(), prog=["dot", "-Gdpi=150"])

# display written file to notebook
Image(plot_fname)
