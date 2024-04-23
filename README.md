
This repo is a Stable Diffusion + ControlNet + TripoSR implementation.
## Getting Started
### Installation
- Python >= 3.8
- Install CUDA if available

```bash
pip install diffusion-webui
```
make sure to install PyTorch compiled with CUDA 11.x.
- `setuptools>=49.6.0`. If not, upgrade by `pip install --upgrade setuptools`.

Then re-install `torchmcubes` by:

```sh
pip uninstall torchmcubes
pip install git+https://github.com/tatsy/torchmcubes.git
```

### Usage
```python
from diffusion_webui import gradio_app

app()
```
