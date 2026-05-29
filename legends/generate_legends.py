"""Generate standalone colormap legend images for each nav-graph layer.

Saves 4 variants per layer into the same directory as this script:
  <layer>_legend_transparent.svg
  <layer>_legend_transparent.png
  <layer>_legend_white.svg
  <layer>_legend_white.png

Colormap: RViz rainbow (0=blue, 1=red) — matches the on-screen
/frontier_score_cloud and car-layer debug images.
"""

import colorsys
import io
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

OUT_DIR = Path(__file__).parent

LAYERS = [
    ('frontier_score',  'Frontier Score',    '0.0 (low)', '1.0 (high)', True),
    ('traversability',  'Traversability',    '0.0 (low)', '1.0 (high)', False),
    ('car',             'Car Score',         '0.0 (low)', '1.0 (high)', True),
    ('visited_time',    'Visited Time',      '0.0 (never)', '1.0 (always)', False),
    ('path_proximity',  'Path Proximity',    '0.0 (near)', '1.0 (far)', False),
    ('grass',           'Grass/Bush Score',  '0.0 (grass)', '1.0 (not grass)', True),
]

N_LUT = 256

def _build_rviz_rainbow(n: int = N_LUT) -> np.ndarray:
    """Reproduce the RViz intensity rainbow: 0→blue, 1→red."""
    out = np.empty((n, 3), dtype=np.float32)
    for i, t in enumerate(np.linspace(0.0, 1.0, n)):
        h = (1.0 - float(t)) * 5.0 / 6.0
        out[i] = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return out

def _rviz_cmap() -> mcolors.ListedColormap:
    lut = _build_rviz_rainbow(N_LUT)
    return mcolors.ListedColormap(lut, name='rviz_rainbow')

CMAP = _rviz_cmap()
CMAP_INV = CMAP.reversed()

FIG_W = 5.0   # inches
FIG_H = 1.1   # inches
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 12


def _make_fig(layer_key: str, layer_label: str,
              lo_label: str, hi_label: str,
              bg: str, invert: bool = False) -> plt.Figure:
    """Return a matplotlib Figure for the legend.

    bg: 'white' or 'none' (transparent).
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(bg)
    fig.patch.set_alpha(0.0 if bg == 'none' else 1.0)

    ax = fig.add_axes([0.08, 0.38, 0.84, 0.32])
    ax.set_facecolor('none')

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cb = ColorbarBase(ax, cmap=CMAP_INV if invert else CMAP,
                      norm=norm, orientation='horizontal')
    cb.outline.set_visible(True)
    cb.outline.set_linewidth(0.8)
    cb.set_ticks([0.0, 1.0])
    cb.set_ticklabels([lo_label, hi_label])
    cb.ax.tick_params(labelsize=LABEL_FONTSIZE, colors='black' if bg == 'white' else 'white',
                      length=4, width=0.8)

    # Tick label colours need to be visible on both backgrounds.
    tick_color = '#222222' if bg == 'white' else '#ffffff'
    for lbl in cb.ax.get_xticklabels():
        lbl.set_color(tick_color)
        lbl.set_fontsize(LABEL_FONTSIZE)

    title_color = '#111111' if bg == 'white' else '#ffffff'
    ax.set_title(layer_label, fontsize=TITLE_FONTSIZE, color=title_color, pad=6,
                 fontweight='bold')

    return fig


def save_layer(layer_key: str, layer_label: str, lo_label: str, hi_label: str,
               invert: bool = False) -> None:
    for bg, bg_tag in [('white', 'white'), ('none', 'transparent')]:
        fig = _make_fig(layer_key, layer_label, lo_label, hi_label, bg, invert)
        transparent = bg == 'none'

        png_path = OUT_DIR / f'{layer_key}_legend_{bg_tag}.png'
        fig.savefig(str(png_path), format='png', dpi=150,
                    transparent=transparent, bbox_inches='tight', pad_inches=0.05)

        svg_path = OUT_DIR / f'{layer_key}_legend_{bg_tag}.svg'
        fig.savefig(str(svg_path), format='svg',
                    transparent=transparent, bbox_inches='tight', pad_inches=0.05)

        plt.close(fig)
        print(f'  saved {png_path.name}  {svg_path.name}')


def save_grass_key() -> None:
    """Save a two-circle key image for the grass layer.

    Purple circle = grass / bush  (score 0.0, LUT index 0)
    Red    circle = normal node   (score 1.0, LUT index 255)
    """
    lut = _build_rviz_rainbow(N_LUT)
    color_grass  = tuple(float(v) for v in lut[0])    # purple (index 0)
    color_normal = tuple(float(v) for v in lut[-1])   # red    (index 255)

    for bg, bg_tag in [('white', 'white'), ('none', 'transparent')]:
        fig, ax = plt.subplots(figsize=(3.2, 1.4))
        fig.patch.set_facecolor(bg)
        fig.patch.set_alpha(0.0 if bg == 'none' else 1.0)
        ax.set_facecolor('none')
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        text_color = '#111111' if bg == 'white' else '#ffffff'

        ax.scatter([0.08], [0.68], s=220, color=[color_grass],
                   edgecolors='black', linewidths=0.8, zorder=3)
        ax.text(0.18, 0.68, 'node on grassy region', va='center',
                fontsize=LABEL_FONTSIZE, color=text_color)

        ax.scatter([0.08], [0.28], s=220, color=[color_normal],
                   edgecolors='black', linewidths=0.8, zorder=3)
        ax.text(0.18, 0.28, 'normal node', va='center',
                fontsize=LABEL_FONTSIZE, color=text_color)

        ax.set_title('Grass/Bush Score', fontsize=TITLE_FONTSIZE,
                     color=text_color, fontweight='bold', pad=4)

        transparent = bg == 'none'
        png_path = OUT_DIR / f'grass_key_{bg_tag}.png'
        fig.savefig(str(png_path), format='png', dpi=150,
                    transparent=transparent, bbox_inches='tight', pad_inches=0.05)
        svg_path = OUT_DIR / f'grass_key_{bg_tag}.svg'
        fig.savefig(str(svg_path), format='svg',
                    transparent=transparent, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f'  saved {png_path.name}  {svg_path.name}')


def save_frontier_score_key() -> None:
    """Save a two-circle key image for the frontier score layer.

    #D9ECF9 circle = frontier node
    Red     circle = non-frontier node
    """
    color_frontier    = mcolors.to_rgb('#D9ECF9')
    color_nonfrontier = (1.0, 0.0, 0.0)  # red

    for bg, bg_tag in [('white', 'white'), ('none', 'transparent')]:
        fig, ax = plt.subplots(figsize=(3.2, 1.4))
        fig.patch.set_facecolor(bg)
        fig.patch.set_alpha(0.0 if bg == 'none' else 1.0)
        ax.set_facecolor('none')
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        text_color = '#111111' if bg == 'white' else '#ffffff'

        ax.scatter([0.08], [0.68], s=220, color=[color_frontier],
                   edgecolors='black', linewidths=0.8, zorder=3)
        ax.text(0.18, 0.68, 'frontier node', va='center',
                fontsize=LABEL_FONTSIZE, color=text_color)

        ax.scatter([0.08], [0.28], s=220, color=[color_nonfrontier],
                   edgecolors='black', linewidths=0.8, zorder=3)
        ax.text(0.18, 0.28, 'non-frontier node', va='center',
                fontsize=LABEL_FONTSIZE, color=text_color)

        ax.set_title('Frontier Score', fontsize=TITLE_FONTSIZE,
                     color=text_color, fontweight='bold', pad=4)

        transparent = bg == 'none'
        png_path = OUT_DIR / f'frontier_score_key_{bg_tag}.png'
        fig.savefig(str(png_path), format='png', dpi=150,
                    transparent=transparent, bbox_inches='tight', pad_inches=0.05)
        svg_path = OUT_DIR / f'frontier_score_key_{bg_tag}.svg'
        fig.savefig(str(svg_path), format='svg',
                    transparent=transparent, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f'  saved {png_path.name}  {svg_path.name}')


def save_path_proximity_key() -> None:
    """Save a two-circle key image for the path-proximity layer.

    Purple circle = node close to robot path  (score 0.0, LUT index 0)
    Red    circle = node far from robot path  (score 1.0, LUT index 255)
    """
    lut = _build_rviz_rainbow(N_LUT)
    color_close = tuple(float(v) for v in lut[0])    # purple (index 0)
    color_far   = tuple(float(v) for v in lut[-1])   # red    (index 255)

    for bg, bg_tag in [('white', 'white'), ('none', 'transparent')]:
        fig, ax = plt.subplots(figsize=(3.2, 1.4))
        fig.patch.set_facecolor(bg)
        fig.patch.set_alpha(0.0 if bg == 'none' else 1.0)
        ax.set_facecolor('none')
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        text_color = '#111111' if bg == 'white' else '#ffffff'

        ax.scatter([0.08], [0.68], s=220, color=[color_close],
                   edgecolors='black', linewidths=0.8, zorder=3)
        ax.text(0.18, 0.68, 'node near robot path', va='center',
                fontsize=LABEL_FONTSIZE, color=text_color)

        ax.scatter([0.08], [0.28], s=220, color=[color_far],
                   edgecolors='black', linewidths=0.8, zorder=3)
        ax.text(0.18, 0.28, 'node far from path', va='center',
                fontsize=LABEL_FONTSIZE, color=text_color)

        ax.set_title('Path Proximity Score', fontsize=TITLE_FONTSIZE,
                     color=text_color, fontweight='bold', pad=4)

        transparent = bg == 'none'
        png_path = OUT_DIR / f'path_proximity_key_{bg_tag}.png'
        fig.savefig(str(png_path), format='png', dpi=150,
                    transparent=transparent, bbox_inches='tight', pad_inches=0.05)
        svg_path = OUT_DIR / f'path_proximity_key_{bg_tag}.svg'
        fig.savefig(str(svg_path), format='svg',
                    transparent=transparent, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f'  saved {png_path.name}  {svg_path.name}')


if __name__ == '__main__':
    for layer_key, layer_label, lo_label, hi_label, invert in LAYERS:
        print(f'Generating: {layer_key}')
        save_layer(layer_key, layer_label, lo_label, hi_label, invert)
    print('Generating: frontier_score_key')
    save_frontier_score_key()
    print('Generating: grass_key')
    save_grass_key()
    print('Generating: path_proximity_key')
    save_path_proximity_key()
    print('Done.')
