# Porous Two-Phase Flow

Development of a two-phase flow model specifically targeting two-phase flow 
in gas diffusion layers (GDL) or porous transport layers (PTL) in fuel cells 
or electrolyzers, respectively.

### Usage
Execute
```python
python main.py
```
with your Python environment. Choose either 1D or 2D model within main.py-File.
Specific configurations (discretization, boundary conditions, parameters, 
etc.) have to be set within the corresponding model files 
(src/capillary_two_phase_flow_1d.py, src/capillary_two_phase_flow_2d.py) 
for now. Structure and usability will be improved in the future.

### Requirements
- Python >= 3.10
- Libraries: NumPy, SciPy, Matplotlib, FiPy

### Model specifications:
- Macrohomogeneous approach:
  - Effective properties for porous medium
  - Continuous transport model
  - Finite-volume discretization using FiPy for 2D
  - Finite-difference discretization for 1D

- Physical model:
  - Capillary-based two-phase model according to:  
  *Secanell, M., A. Jarauta, A. Kosakian, M. Sabharwal, and J. Zhou. 
  “PEM Fuel Cells, Modeling.” In Encyclopedia of Sustainability Science and
  Technology, edited by Robert A. Meyers, 1–61. New York, NY: 
  Springer New York, 2017. https://doi.org/10.1007/978-1-4939-2493-6_1019-1.*
  - Saturation-capillary pressure correlations from:  
  *Pasaogullari, Ugur, and C. Y. Wang. “Liquid Water Transport in Gas Diffusion 
  Layer of Polymer Electrolyte Fuel Cells.” Journal of The Electrochemical 
  Society 151, no. 3 (2004): A399. https://doi.org/10.1149/1.1646148.*  
  *Zhou, J., A. Putz, and M. Secanell. “A Mixed Wettability Pore Size 
  Distribution Based Mathematical Model for Analyzing Two-Phase Flow in Porous 
  Electrodes: I. Mathematical Model.” Journal of The Electrochemical Society 
  164, no. 6 (2017): F530–39. https://doi.org/10.1149/2.0381706jes.*