# QubitPulseOpt Demo Materials

This directory contains visual demo materials for portfolio presentations, social media, and the main README.

## Contents

### Generated Materials

Run the generation script to create demo materials:

```bash
python scripts/generate_demo_materials.py
```

This will generate:

1. **`bloch_evolution.gif`** (< 2 MB)
   - 30-second animated loop showing pulse evolution on Bloch sphere
   - Shows X, Y, and Hadamard gate trajectories side-by-side
   - Optimized for README hero section

2. **`parameter_sweep.png`** (< 500 KB)
   - High-resolution heatmap of gate fidelity vs T1/T2 decoherence times
   - Contour lines at F=0.95, 0.99, 0.999
   - Annotated with typical superconducting qubit parameters

3. **`optimization_convergence.gif`** (< 2 MB)
   - Animation showing GRAPE optimization convergence
   - Dual plot: fidelity (linear) and infidelity (log scale)
   - Shows progression from random initialization to F > 0.999

4. **`dashboard_screenshot.png`** (< 1 MB)
   - Comprehensive 6-panel dashboard showing:
     - Bloch sphere evolution
     - Optimized control pulse
     - Convergence curve
     - Key metrics panel
     - Filter function analysis
     - Error budget pie chart
   - High-resolution (300 DPI) for professional presentations

### Manual Generation (If Script Fails)

If the automated script doesn't work, you can generate materials manually using the notebooks:

1. **Bloch Sphere Animation:**
   - Open `notebooks/02_rabi_oscillations.ipynb`
   - Run cells to generate state evolution
   - Use QuTiP's Bloch sphere animation features
   - Export as GIF using matplotlib animation

2. **Parameter Sweep:**
   - Open `notebooks/06_robustness_analysis.ipynb`
   - Run parameter sweep cells
   - Export heatmap figure

3. **Optimization Convergence:**
   - Open `notebooks/05_gate_optimization.ipynb`
   - Run GRAPE optimization cells
   - Plot convergence history
   - Export as static PNG or animate with matplotlib

### Optimization Tips

**GIF File Size Reduction:**
```bash
# Using gifsicle (install with: apt-get install gifsicle)
gifsicle -O3 --colors 128 input.gif -o output_optimized.gif

# Or using ImageMagick
convert input.gif -fuzz 10% -layers Optimize output_optimized.gif
```

**PNG Compression:**
```bash
# Using optipng
optipng -o7 input.png

# Or using pngquant
pngquant --quality=80-95 input.png
```

## Usage Guidelines

### README Integration

Add to the main README.md:

```markdown
## Demonstration

![Bloch Sphere Evolution](examples/demo_materials/bloch_evolution.gif)

*Pulse-driven evolution on the Bloch sphere showing X, Y, and Hadamard gates*

### Optimization Performance

![Optimization Convergence](examples/demo_materials/optimization_convergence.gif)

*GRAPE algorithm converging to F > 0.999 in ~50 iterations*
```

### Social Media

**LinkedIn Post:**
- Use `dashboard_screenshot.png` as the main image
- Highlight key metrics (99.9% fidelity, 97.5% compliance)
- Professional tone, emphasize real-world applications

**Twitter/X Thread:**
- Tweet 1: Dashboard screenshot + intro
- Tweet 2: Bloch sphere animation (upload GIF directly)
- Tweet 3: Key results and GitHub link
- Tweet 4: Technical details + parameter sweep

**Reddit (r/QuantumComputing):**
- Self-post with technical details
- Embed all GIFs using image hosting (Imgur)
- Link to GitHub and documentation
- Focus on implementation details and open-source nature

### Presentations

- `dashboard_screenshot.png`: High-res for slides (300 DPI)
- `parameter_sweep.png`: Technical deep-dives
- GIFs: Use in introduction/demo sections

## File Size Targets

- Total demo materials: < 5 MB
- Individual GIFs: < 2 MB each
- PNGs: < 1 MB each (except dashboard at 300 DPI)

## Notes

- All materials are generated from actual simulation data
- No mock-ups or placeholder content
- Animations use real QuTiP Bloch sphere rendering
- Colors and styling match project branding
- All figures include proper axis labels and titles