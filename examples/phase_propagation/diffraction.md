# Diffraction
The code is at `diffraction.py` in this directory.

## Goal
Verify whether the framework can accurately model the diffraction phenomena for various parameters (e.g. wavelengths, pupil widths, and pupil types.)  

## Opical System
![Optical System](./figures/diffraction/opt_setting.png)  

The default configuration of the optical system follows as :

```python
Prop = Diffraction(
    pixel_size=[0.6, 0.6],
    pixel_num=[1000, 1000],
    lamb0=[0.4, 0.55, 0.7],
    refractive_index=1,
    paraxial=False,
    focal_length=10*1e3,
    NA=0.3,
    pupil_type=pupil_type,
    pupil_width=100,
    nyquist_spatial_bound=False
)
```

## Check the intermediate outputs of the code.
1. Field before applying the pupil function of the given aperture.
2. Field after applying the pupil function
3. Field after applying ASM (distance z = 10µm)
4. Detected intensity on the sensor.

### Circle Aperture
1. Field before applying the pupil function of the given aperture. - This is just constant because we set the source as normally incident plane wave.
2. Field after applying pupil function  
<img src="./figures/diffraction/400wvl_after_circ.png" width="300"><img src="./figures/diffraction/550wvl_after_circ.png" width="300"><img src="./figures/diffraction/700wvl_after_circ.png" width="300">

3. Field after applying ASM (distance z = 10µm)  
<img src="./figures/diffraction/400wvl_prop_circ.png" width="300"><img src="./figures/diffraction/550wvl_prop_circ.png" width="300"><img src="./figures/diffraction/700wvl_prop_circ.png" width="300">


### Square Aperture
1. Field before applying pupil function of the given aperture. - This is just constant because we set the source as normally incident plane wave.
2. Field after applying pupil function  
<img src="./figures/diffraction/400wvl_after_sqr.png" width="300"><img src="./figures/diffraction/550wvl_after_sqr.png" width="300"><img src="./figures/diffraction/700wvl_after_sqr.png" width="300">

3. Field after applying ASM (distance z = 10µm)  
<img src="./figures/diffraction/400wvl_prop_sqr.png" width="300"><img src="./figures/diffraction/550wvl_prop_sqr.png" width="300"><img src="./figures/diffraction/700wvl_prop_sqr.png" width="300">


## Check the diffraction pattern for various parameter settings (aperture type, wavelengths, pupil width)

### Circle Aperture
![Circle aperture](./figures/diffraction/circ_wvl_pupwidth.png)  

### Square Aperture
![Square aperture](./figures/diffraction/sqr_wvl_pupwidth.png)  


- As pupil width increases, the main lobe size decreases.
- As wavelength increases, the main lobe size increases.
- If the pupil width is too large, the diffraction pattern is split. - This has to be checked theoretically.