"""Video-PiD: pixel-space diffusion decoder for Wan 2.1.

Fixes the "plastic" / waxy look of Wan-VAE-decoded video by running a small 3D
diffusion model in pixel space, conditioned on the original Wan latent.

Architecture is pinned in docs/ARCHITECTURE.md.
"""

__version__ = "0.0.1"
__author__ = "madxmoron"
__license__ = "Apache-2.0"
