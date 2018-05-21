# Signed Distance Function Renderer
Pure tensorflow implementation of a signed distance function renderer where intersections are computed via sphere-marching.

## Setup
Pull this repository and add the parent directory to your `PYTHONPATH`
```
cd /path/to/parent_dir
git clone https://github.com/jackd/sdf_renderer.git
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```

You may wish to append this `PYTHONPATH` modification to your `~/.bashrc`.

# TODO
* Spectral lighting
* Sphere-march from light sources? Currently we assume all intersections can view all light sources, but that's obviously wrong. Maybe refactor `get_intersections` to take inputs with an extra rank?
* Further debugging of `linearize_roots`. Where does it fail? See `examples/opt.py` rgb behvariour.
* Other intersection algorithms.
* More SDF smoothing functions.
