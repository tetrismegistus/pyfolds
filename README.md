# pyfolds

GPU flame-fractal style transforms implemented in a ModernGL compute-shader pipeline.

The program runs an interactive window and applies iterative variation transforms similar to flame fractals, but executed entirely on the GPU.

## Requirements

* Linux / macOS / Windows with OpenGL 4.3+
* [uv](https://github.com/astral-sh/uv)

## Notes

* The first run may compile shaders.
* Performance depends primarily on GPU compute capability.
* Tested with Python 3.11.

## Development

```bash
uv sync --dev
uv run pyfolds
```
