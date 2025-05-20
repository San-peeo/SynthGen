# SynthGen: A Gravitational Simulator for Planetary Interior Modelling

Determining the internal structure of planetary bodies from gravitational observations is a key challenge in planetary geophysics. Gravity inversion techniques make it possible to estimate mass distribution by combining information on a body's shape, gravitational field, and rotational dynamics. However, gravity data alone present a well-known ambiguity between mass magnitude and depth, making the interpretation of internal layering a complex inverse problem. 

We present **SynthGen**, a code developed to simulate the gravitational response of planetary bodies based on parametric interior models. It exploits the spherical harmonics framework described in Wieczorek [1], through the computation of gravitational harmonic coefficients [C_{nm}, S_{nm}] which characterise the global gravitational field, thanks to the SHTools [2] routines. *SynthGen* takes as input a simplified multi-layer interior model, assuming them homogenous. Model parameters consist of the number of internal layers, their mean thickness and density, and eventually, the topography of internal interfaces. On the latter, several ways are implemented: sphere, polar/equatorial flattened ellipsoid, randomly generated topography, downwarded Bouguer anomaly (see Wieczorek & Phillips [2], avoiding isostacy assumptions) and finally an input user grid. Given these inputs, *SynthGen* computes the corresponding gravitational potential, Free-Air anomalies, and Bouguer anomaly fields for the modelled body, generating full-resolution global maps.

*SynthGen* outputs can be used in two ways: 
1) If it is used to simulate a known body, so a gravity model is already available, the synthetic results can be compared to the real measurements, assessing the validity of the evaluated interior model and measuring its performance through different metrics. In this case, *SynthGen* performed an automated parameter-space exploration (controlled by the user). By randomly sampling model parameters within physically plausible bounds that it is user-configurable (constrained by the satisfaction of the conservation of total mass and moment of inertia, together with external shape constraints), it iteratively evaluates a wide range of configurations. The optimal internal structure is determined by identifying the parameter set that minimises discrepancies between simulated and observed gravitational data. This is performed through a suite of statistical metrics (e.g. RMSE, MAE, R2, SSIM, NCC, PSNR, etc.), finally combined into one.

2) In addition to this procedure, *SynthGen* can be used predictively in case of an “unmeasured” body. It enables forward modelling of gravitational signals expected from future targets (for example Ganymede, for ESA’s JUICE mission). It can thus serve as a valuable tool for testing theoretical interior structures and simulating their measurable gravitational signatures.

By combining analytical modelling, numerical efficiency, and flexibility across planetary scenarios, *SynthGen* offers a useful platform for planetary interior investigations from the gravitational point of view. It can handle various planetary shapes, datasets, and scientific objectives, and it is user configurable, together with already implemented configuration files for Mercury, Venus, Earth and Moon, together with a model of Ganymede.

Please cite this if you use it in your research



#### Example 1: Comparison on Mercury between Synthetic generated data and MESSENGER-derived model
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


# Datasets:
see Planets_ConfigFiles.py; In this module, configuration classes for the main terrestrial planets and Ganymede, including their physical parameters, gravity and topography data sources, and interior structure models, are provided.  
Each planet class contains methods to retrieve bulk parameters (.bulk()), data file information (.data()), and interior structure parameters for different numbers of layers (.interiors()).

## Mercury 
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 2439.4               | km           | Reference radius                            |
| GM_const          | 2.2031863566e+13     | m³/s²        | Gravitational constant                      |
| ref_mass          | 3.301e+23            | kg           | Reference mass                              |
| ref_rho           | 5427                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 8.264e-07            | rad/s        | Angular velocity                            |
| ref_MoI           | 0.34597              | (I/MR²)      | Moment of inertia factor (Margot et al 2018)|
| r_e_fact          | 1.0005               | -            | Equatorial flattening factor                |
| r_p_fact          | 0.9995               | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                                      | Format   | Header | Reference                        |
|-------------|------------------------------------------------|----------|--------|------------------------------|
| Gravity     | HgM009.sha                 | shtools  | True   | A. Genova et al., ‘Regional variations of Mercury’s crustal density and porosity from MESSENGER gravity data’, Icarus, 2023, doi: 10.1016/j.icarus.2022.115332.          |
| Topography  | gtmes_150v05_sha_nohead.txt | shtools  | False  | pds-geosciences.wustl.edu - /messenger/mess-h-rss_mla-5-sdp-v1/messrs_1001/                 |

- **Bouguer density:** 2900 kg/m³  
- **Crustal thickness filter (n_half):** 40

### Implemented Interior Models

#### 3 Layers
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Core      | 6992            | 2039            | polar flattened sphere|
| Mantle      | 3200            | 2404            | downwarded Bouguer anomalies|
| Crust      | 2900            | 2439.4            | surface|

#### 4 Layers 
J.-L. Margot, S. A. H. II, E. Mazarico, S. Padovan, and S. J. Peale, ‘Mercury’s Internal Structure’, 2018, pp. 85–113. doi: 10.1017/9781316650684.005
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Inner Core      | 8652.52            | 666.577            | sphere|
| Outer Core      | 6909.98            | 2023.66            | polar flattened sphere|
| Mantle      | 3343.35            |  2402.61            | downwarded Bouguer anomalies|
| Crust      | 2903.03            | 2439.4            | surface|



---

## Venus 
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 6051.8               | km           | Reference radius                            |
| GM_const          | 3.248585920790000e+14| m³/s²        | Gravitational constant                      |
| ref_mass          | 4.8673e+24           | kg           | Reference mass                              |
| ref_rho           | 5243                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 3.232e-07            | rad/s        | Angular velocity                            |
| ref_MoI           | 0.337                | (I/MR²)      | Moment of inertia factor (Margot et al 2021)|
| r_e_fact          | 1.0                  | -            | Equatorial flattening factor                |
| r_p_fact          | 1.0                  | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                        | Format   | Header | References                        |
|-------------|----------------------------------|----------|--------|------------------------------|
| Gravity     | shgj180u_noheader.a01            | shtools  | True   | A. S. Konopliv, W. B. Banerdt, and W. L. Sjogren, ‘Venus Gravity: 180th Degree and Order Model’, Icarus, 1999, doi: 10.1006/icar.1999.6086.                             |
| Topography  | VenusTopo719.shape               | shtools  | False  | Wieczorek, M. A. (2015). Spherical harmonic model of the planet Venus: VenusTopo719 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3870926 |

- **Bouguer density:** 2800 kg/m³  
- **Crustal thickness filter (n_half):** 80

### Implemented Interior Models
#### 3 Layers
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Core      | 13000            | 3200            | polar flattened sphere|
| Mantle      | 3300            | 6020            | downwarded Bouguer anomalies|
| Crust      | 2800            | 6051.8          | surface|



---

## Earth 
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 6378.137             | km           | Reference radius                            |
| GM_const          | 3.986004418e+14      | m³/s²        | Gravitational constant                      |
| ref_mass          | 5.9722e+24           | kg           | Reference mass                              |
| ref_rho           | 5514                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 1.992e-07            | rad/s        | Angular velocity                            |
| ref_MoI           | 0.3308               | (I/MR²)      | Moment of inertia factor (Williams and James, 1994)|
| r_e_fact          | 0.9999               | -            | Equatorial flattening factor                |
| r_p_fact          | 0.9970               | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                        | Format   | Header | Notes                        |
|-------------|----------------------------------|----------|--------|------------------------------|
| Gravity     | EGM2008_to2190_TideFree          | shtools  | False  | N. K. Pavlis, S. A. Holmes, S. C. Kenyon, and J. K. Factor, ‘The development and evaluation of the Earth Gravitational Model 2008 (EGM2008)’, Journal of Geophysical Research: Solid Earth, 2012, doi: 10.1029/2011JB008916. |
| Topography  | Earth2014.BED2014.degree10800.bshc| bshc    | False  | Hirt, C. and M. Rexer (2015), Earth2014: 1 arc-min shape, topography, bedrock and  ice-sheet models - available as gridded data and degree-10,800 spherical harmonics, International Journal of Applied Earth Observation and Geoinformation, doi:10. 10.1016/j.jag.2015.03.001.|

- **Bouguer density:** 1800 kg/m³  
- **Crustal thickness filter (n_half):** 80

### Implemented Interior Models
#### 5 Layers
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Inner Core      | 13088.5            | 1221.5            | sphere|
| Outer Core      | 12581.5            | 3480.0            | polar flattened sphere|
| Lower Mantle    | 7956.5             | 5701.0            | polar flattened sphere|
| Upper Mantle    | 7090.9             | 6151.0            | polar flattened sphere|
| Crust           | 2800               | 6371.0            | surface|

#### 8 Layers 
| Layer           | Density (kg/m³) | Radius (km) | Interface Type              |
|-----------------|-----------------|-------------|-----------------------------|
| Inner Core      | 13088.5         | 1221.5      | sphere                      |
| Outer Core      | 12581.5         | 3480.0      | polar flattened sphere      |
| Lower Mantle    | 7956.5          | 5701.0      | polar flattened sphere      |
| Upper Mantle    | 7090.9          | 6151.0      | polar flattened sphere      |
| Transition Zone | 2691.0          | 6346.0      | downwarded Bouguer anomalies|
| Lower Crust     | 2900            | 6356.0      | polar flattened sphere      |
| Upper Crust     | 2600            | 6368.0      | surface                     |
| Sediments       | 1020            | 6371.0      | surface                     |



---

## Moon (WIP)
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 1738.1               | km           | Reference radius                            |
| GM_const          | 4.9028001218467998e+12| m³/s²       | Gravitational constant                      |
| ref_mass          | 0.07346e+24          | kg           | Reference mass                              |
| ref_rho           | 3344                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 2.7e-06              | rad/s        | Angular velocity                            |
| ref_MoI           | 0.3929               | (I/MR²)      | Moment of inertia factor (Williams and James, 1996)|
| r_e_fact          | 1.0                  | -            | Equatorial flattening factor                |
| r_p_fact          | 0.9988               | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                        | Format   | Header | Notes                        |
|-------------|----------------------------------|----------|--------|------------------------------|
| Gravity     | GRGM1200l_data.txt               | shtools  | True   | Lemoine, F. G., et al. (2014), GRGM900C: A degree 900 lunar gravity model from GRAIL primary and extended mission data, Geophys. Res. Lett., doi:10.1002/2014GL060027, Goossens, S., et al. (2016), A Global Degree and Order 1200 Model of the Lunar Gravity Field using GRAIL Mission Data, Lunar and Planetary Science Conference, Houston, TX, Abstract #1484.|
| Topography  | MoonTopo2600p.shape              | shtools  | False  |M. A. Wieczorek, «Spherical harmonic model of the shape of Earth's Moon: MoonTopo2600p». Zenodo, 2015. doi: 10.5281/zenodo.3870924.|

- **Bouguer density:** 2900 kg/m³  
- **Crustal thickness filter (n_half):** 40

### Implemented Interior Models
#### 4 Layers
| Layers | Densities (kg/m³)         | Radii (km)              | Interface Types                | 
|--------|--------------------------|-------------------------|-------------------------------|
| Inner Core      | 0            | 0            | sphere|
| Outer Core      | 0            | 0            | sphere|
| Mantle    | 0            | 0           | sphere|
| Crust    | 0            | 0           | sphere|



---

## Ganymede 
### Bulk Parameters
| Parameter         | Value                | Unit         | Description                                 |
|-------------------|----------------------|--------------|---------------------------------------------|
| ref_radius        | 2631.2               | km           | Reference radius                            |
| GM_const          | 9.8780e+12           | m³/s²        | Gravitational constant                      |
| ref_mass          | 1.48e+23             | kg           | Reference mass                              |
| ref_rho           | 1942                 | kg/m³        | Mean density                                |
| ref_ang_vel       | 8.264e-07            | rad/s        | Angular velocity                            |
| ref_MoI           | 0.3115               | (I/MR²)      | Moment of inertia factor (Schubert, Anderson, Spohn, McKinnon, 2004) |                 |
| r_e_fact          | 1.0                  | -            | Equatorial flattening factor                |
| r_p_fact          | 1.0                  | -            | Polar flattening factor                     |

### Data Files
| Data Type   | File Path                        | Format   | Header | Notes                        |
|-------------|----------------------------------|----------|--------|------------------------------|
| Gravity     | None                             | shtools  | True   |                              |
| Topography  | None                             | shtools  | False  |                              |

- **Bouguer density:** 920 kg/m³  
- **Crustal thickness filter (n_half):** 25

### Implemented Interior Models
#### 7 Layers
| Layer           | Density (kg/m³) | Radius (km) | Interface Type |
|-----------------|-----------------|-------------|---------------|
| Core            | 8000            | 570         | sphere        |
| Mantle         | 3400            | 1820        | sphere        |
| Crust         | 3100            | 1870        | rng           |
| Ice VI         | 1320            | 2000        | sphere        |
| Ice V         | 1235            | 2280        | sphere        |
| Ocean         | 1100            | 2460        | sphere        |
| Ice I       | 920             | 2631.2      | surface(rng)       |



  

# Code Description

## Libraries
- **main_library.py**: Core library containing all main functions for model generation, analysis, spectrum computation, metrics evaluation, and utility routines used throughout the project. Each function is provided with documentation and help for the input variables
- **Planets_ConfigFiles.py**: Contains configuration classes for each supported planetary body (Mercury, Venus, Earth, Moon, Ganymede). Each class provides bulk parameters, data file paths, and interior structure models. See also the previous dataset section (Moon interior models is WIP).
- **requirements.txt**: Lists all Python package dependencies and their required versions for the project.
 
## Scripts
- **main.py**: Handles gravity and topography data in spherical harmonics expansion, evaluating also the power spectrum of the gravity field. It visualises projected maps and produces plots and data files.
  - Inputs:
    - `body`          = planetary body from which data is collected and studied  (Mercury, Earth, Venus, Moon)
    - `n_min`         = minimum degree of spherical harmonics expansion (n=0 is GM/R, n=1 is when centre of mass is not the same as the centre of coordinates, n=2 indicates the polar flattening and is usually the strongest)
    - `n_max`         = maximum degree of spherical harmonics expansion (keep an eye on the grav and topo file maximum degree)
    - `r`            = evaluation radius in meters [m] (usually = ref_radius)
    - `i_max`         = Bouguer Taylor series index (usually 7 works)
    - `proj_opt`      = projection type , default = ccrs.Mollweide() (see ccrs list)
    - `verbose_opt`   = verbose option to print on the terminal information about the progress and outputs

                
- **main_synthgen.py**: Generates and analyses synthetic gravity and topography models for planetary interiors. Allows for custom configuration of interior layers and parameters, and produces synthetic data, maps, and spectra.  
  - Inputs:
    - `body`: Planetary body to simulate (Mercury, Earth, Venus, Moon, Ganymede)
    - `n_layers`: Number of interior layers in the synthetic model
    - `n_min`, `n_max`: Minimum and maximum spherical harmonic degree for expansion
    - `r`: Evaluation radius in meters [m]
    - `i_max`: Bouguer Taylor series index
    - `mode`: Synthetic generation mode (`layers` or `interface`), distinguishing contributions for each layer or for each interface (density differences)
    - `save_opt`: Option to save results and figures
      - `None` = no saving;
      - `all` = save all layers/interface coefficients and spectra;
      - `total` =save just the final global synthetic coefficients.
    - `load_opt`: Option (`True`-`False`) to load existing coefficients (if present) 
    - `proj_opt`      = projection type , default = ccrs.Mollweide() (see ccrs list)
    - `verbose_opt`   = verbose option to print on the terminal information about the progress and outputs


- **main_synthgen_grid.py**: Random grid exploration of the user-input parameters space: it produces *N* models generated within the desired range around a pre-setted initial model (from `Planets_ConfigFiles.py`). The user must insert some input information and/or use default values ( +/- 200 km, +/- 200 kg/m^3 and n_{half}\in[3-100]).
  - Inputs:
    - same as `main_synthgen.py`"
    - User inputs:
      - n_counts = number of valid models to produce;
      - range = range of parameters to explore (see `InputRange` function)


- **main_synthgen_grid.py**: Random grid exploration of the user-input parameters space: it produces *N* models generated within the desired range around a pre-setted initial model (from `Planets_ConfigFiles.py`). The user must insert some input information and/or use default values ( +/- 200 km, +/- 200 kg/m^3 and n_{1/2} € [3-100]).
  - Inputs:
    - same as `main_synthgen.py`"
    - User inputs:
      - n_counts = number of valid models to produce;
      - range = range of parameters to explore (see `InputRange` function)
     
       
- **main_synthgen_grid_loading.py**: Analyse the models' results from the grid evaluations (keep the same main input parameter as `main_synthgen_grid.py` to analyse the same grid). Use must choose the metric list to calculate and then rank the models.  
  - Inputs:
    - `body`: Planetary body to simulate (Mercury, Earth, Venus, Moon, Ganymede)
    - `n_layers`: Number of interior layers in the synthetic model
    - `n_min`, `n_max`: Minimum and maximum spherical harmonic degree for expansion
    - `r`: Evaluation radius in meters [m]
    - `i_max`: Bouguer Taylor series index
    - `load_opt`: Option (`True`-`False`) to load existing analysis metrics (if present) 
    - `plot_opt`: Option to visualise results and figures
      - `all` = plot all of the models' metrics and histograms;
      - `top` = plot just the top `thresh` % of the models' metrics and histogram
    - `metrics_list` = statistical metrics to evaluate and rank for performance models measurements
      - `Delta_mean` = mean of the difference between Synthetic and Real maps;
      - `Delta_std` = standard deviation of the difference between Synthetic and Real maps;
      - `MAE` = Mean Absolute Errors;
      - `RMSE` =  Rott Mean Squared Errors;
      - `R^2` = Coefficient of Determination;
      - `PSNR` = Peak Signal-to-Noise Ratio;
      - `SSIM` = Structure Similarity Index Measure;
      - `NCC` = Normalized Cross-Correlation;
    - `threshold_arr` = array of thresholds to select the top % models
    



