# SynthGen: A Gravitational Simulator for Planetary Interior Modelling

Determining the internal structure of planetary bodies from gravitational observations is a key challenge in planetary geophysics. Gravity inversion techniques make it possible to estimate mass distribution by combining information on a body's shape, gravitational field, and rotational dynamics. However, gravity data alone present a well-known ambiguity between mass magnitude and depth, making the interpretation of internal layering a complex inverse problem. 

We present **SynthGen**, a code developed to **simulate the gravitational response of planetary bodies based on parametric interior models**. It exploits the spherical harmonics framework described in Wieczorek [1], through the computation of gravitational harmonic coefficients [C_{nm}, S_{nm}] which characterise the global gravitational field, thanks to the **SHTools** [2] routines. *SynthGen* takes as input a simplified multi-layer interior model, assuming them homogenous. Model parameters consist of the **number of internal layers**, their **mean thickness** and **density**, and eventually, the **topography of internal interfaces**. On the latter, several ways are implemented: sphere, polar/equatorial flattened ellipsoid, randomly generated topography, downwarded Bouguer anomaly (see Wieczorek & Phillips [3], avoiding isostacy assumptions) and finally an input user grid. Given these inputs, *SynthGen* computes the corresponding gravitational potential U, Free-Air anomalies, and Bouguer anomaly fields for the modelled body, generating full-resolution global maps.

*SynthGen* outputs can be used in two ways: 
1) If it is used to simulate a **known body**, so a gravity model is already available, the synthetic results can be compared to the real measurements, assessing the validity of the evaluated interior model and measuring its performance through different metrics. In this case, *SynthGen* performed an **automated parameter-space exploration** (controlled by the user). By randomly sampling model parameters within physically plausible bounds that it is user-configurable (constrained by the satisfaction of the conservation of total mass and moment of inertia, together with external shape constraints), it iteratively evaluates a wide range of configurations. The optimal internal structure is determined by identifying the parameter set that minimises discrepancies between simulated and observed gravitational data. This is performed through a suite of statistical metrics (e.g. RMSE, MAE, R2, SSIM, NCC, PSNR, etc.), finally combined into one.

2) In addition to this procedure, *SynthGen* can be used predictively in case of an **“unmeasured” body**. It enables forward modelling of gravitational signals expected from future targets (for example, Ganymede, for ESA’s JUICE mission). It can thus serve as a valuable tool for **testing theoretical interior structures** and simulating their measurable gravitational signatures.

By combining analytical modelling, numerical efficiency, and flexibility across planetary scenarios, *SynthGen* offers a useful platform for planetary interior investigations from the gravitational point of view. It can handle various planetary shapes, datasets, and scientific objectives, and it is user configurable, together with already implemented configuration files for **Mercury, Venus, Earth and Moon, together with a model of Ganymede**.

Please cite this if you use it in your research



#### Example 1a: SynthGen layer generation for 4-layer Mercury: Layers' topography - Layers' Gravitational Potential U 
![U-interface](https://github.com/user-attachments/assets/6da4616f-458f-4c4d-9ba5-452f3cb87b9e)

#### Example 1b: Comparison on Mercury between Synthetic generated data and MESSENGER-derived model
![Comparison on Mercury between Synthetic generated data and MESSENGER-derived model](https://github.com/user-attachments/assets/7fa9d8eb-02e0-483a-9b52-42c1022995bc)


#### Example 2: Realistic synthetic model of the Jovian icy moon Ganymede
![U_h_FreeAir_spectrum_nmin3_nmax50](https://github.com/user-attachments/assets/b6f053e3-f112-4ad0-b323-a5d414007995)





# Environment Setup


## Linux/Windows WSL
```bash
# Create a virtual environment
python -m venv myenv

# Activate the environment
source myenv/bin/activate

# Install packages
pip install -r requirements.txt
```


  

# Code Description


## Libraries
- **lib**: Core library main folder, containing all functions for model generation, analysis, spectrum computation, metrics evaluation, and utility routines used throughout the project. Each function is provided with documentation and help for the input variables and is divided between subdirectories:
  - `globe_analysis` = functions for preliminary globe analysis: gravitational maps, crust thicknesses and spectrum analysis;
  - `io` = configuration classes for planetary bodies (Mercury, Venus, Earth, Moon, Ganymede, Ceres) and data file import/export;
  - `synthgen` = main SynthGen gravitational modelling function for layer-based gravity synthesis;
  - `grid` = random parameter space exploration and grid-based metrics analysis routines;
  - `love_numbers` = Love number computation and complex rheology evaluation;
  - `misc` = miscellaneous utilities including custom map plotting, M-MoI solver, mass/volume calculations, and data smoothing;
  - *lib_dep.py* = dependencies and required Python packages.

- **main_lib.py**: Import file from `lib` directory
  

## Planets
- Configuration module aggregating all planetary body configuration classes. Each class contains three main methods:
  - `bulk()`: Returns reference bulk parameters (mean radius, GM, mass, density, angular velocity, moment of inertia, Love numbers)
  - `data()`: Returns data file paths and parameters (gravity coefficients, topography, Bouguer density, crustal thickness filter)
  - `interiors()`: Returns interior structure parameters for different layer configurations (densities, layer radii, interface types, layer names, rheology properties)
  - Implemented planetary bodies:
    - Mercury
    - Venus
    - Earth
    - Moon
    - Ganymede
    - Ceres
    - Custom (user-defined)

    
- **requirements.txt**: Lists all Python package dependencies and their required versions for the project.
 

## Scripts
- **main.py**: Processes gravitational and topographic data for a planetary body (defined in the config file). Computes gravity power spectra, Free-Air and Bouguer anomalies in spherical harmonics representation, visualizes projected maps, and generates analysis figures and data files.
  - Input variables:
    - `body`           = planetary body to study
    - `n_min`, `n_max` = minimum and maximum degree of spherical harmonics expansion
    - `r`              = evaluation radius in meters [m]
    - `i_max`          = Bouguer Taylor series index (default: 7)
    - `region`         = regional subset as [lon_min, lon_max, lat_min, lat_max] or None (global)
    - `proj_opt`       = map projection type (default: ccrs.Mollweide())
    - `load_opt`       = whether to load existing coefficients (True/False)
    - `verbose_opt`    = verbose output flag (True/False)

                
- **main_synthgen.py**: Generates and analyzes synthetic gravitational fields for user-defined planetary interior models. Computes gravity coefficients, spectra, and anomaly maps for multi-layer interior structures with customizable layer properties and interface geometries.
  - Input variables:
    - `body`          = planetary body to simulate (Mercury, Earth, Venus, Moon, Ganymede, Ceres)
    - `n_layers`      = number of interior layers in the synthetic model
    - `n_min`, `n_max` = minimum and maximum degree of spherical harmonics expansion
    - `r`            = evaluation radius in meters [m]
    - `i_max`         = Bouguer Taylor series index (default: 7)
    - `mode`          = generation mode: 'layer' (contributions per layer) or 'interface' (contributions per interface/density contrast)
    - `save_opt`      = saving option: None (no saving), 'all' (all layers/interfaces), 'total' (final global coefficients)
    - `load_opt`      = whether to load existing coefficients (True/False)
    - `sub_dir`       = subdirectory naming: 'auto' (default) or custom name
    - `region`       = regional subset or None (global)
    - `proj_opt`      = map projection type (default: ccrs.Mollweide())
    - `verbose_opt`   = verbose output flag (True/False)


- **main_synthgen_grid.py**: Performs automated grid-based parameter space exploration to generate an ensemble of synthetic interior models. Randomly samples layer densities, interface radii, and crustal filter parameters within user-defined ranges, computes synthetic gravity for each model, and evaluates performance metrics by comparison with real observational data.
  - Input variables:
    - `body`          = planetary body to study (Mercury, Earth, Venus, Moon, Ganymede, Ceres)
    - `n_layers`      = number of interior layers
    - `n_min`, `n_max` = minimum and maximum degree of spherical harmonics expansion
    - `r`            = evaluation radius in meters [m]
    - `i_max`         = Bouguer Taylor series index (default: 7)
    - `save_opt`      = saving option: None (no saving), 'all' or 'total'
    - `metrics_list`  = list of statistical metrics to compute (SSIM, NCC, RMSE, MAE, R², PSNR, etc.)
    - `threshold_arr` = array of percentile thresholds for selecting top models
    - `region`       = regional subset or None (global)
    - `proj_opt`      = map projection type
    - `plot_results`  = visualization option: 'top', 'average', or 'both'
             
       


## Citation
If you use SynthGen in your research, please cite:
Santero Mormile, E., et al. "SynthGen: A Gravity Field Simulator For Planetary Interior Modelling", Icarus, 2025.






# References
* [1] M. A. Wieczorek, "Gravity and Topography of the Terrestrial Planets", in Treatise on Geophysics, Elsevier, 2015, pp. 153–193. doi:10.1016/B978-0-444-53802-4.00169-X
* [2] M A. Wieczorek and Matthias Meschede (2018). “SHTools — Tools for working with spherical harmonics”, Geochemistry, Geophysics, Geosystems, 19, 2574-2592, doi:10.1029/2018GC007529.
* [3] M. A. Wieczorek and R. J. Phillips, “Potential anomalies on a sphere: Applications to the thickness of the lunar crust”, JGR: Planets, vol. 103, no. E1, pp. 1715–1724, 1998. doi:10.1029/97JE03136
* [4] A. Genova et al., “Regional variations of Mercury’s crustal density and porosity from MESSENGER gravity data”, Icarus, vol. 391, p. 115332, 2023.
* [5] A. Rivoldini, T. Van Hoolst, “The interior structure of Mercury constrained by the low-degree gravity field and the rotation of Mercury”, Earth and Planetary Science Letters, 2013, https://doi.org/10.1016/j.epsl.2013.07.021.
* [6] J. L. Margot, et al. "Mercury's internal structure." arXiv preprint arXiv:1806.02024 (2018).
* [7] D. M. Fabrizio et al., ‘Observability of Ganymede’s gravity anomalies related to surface features by the 3GM experiment onboard ESA’s JUpiter ICy moons Explorer (JUICE) mission’, Icarus, 2021.
* [8] Schubert, G., & Anderson, J. (2004). Interior composition, structure and dynamics of the galilean satellites. Jupiter: The planet, satellites and magnetosphere, 1 ,281–306.



# License
This project uses the BSD 3-Clause license, as found in the LICENSE file.
If you want to request access to the development repository, contact: edoardo.santeromormile@gmail.com
```


