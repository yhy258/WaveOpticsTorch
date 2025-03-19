# WaveOpticsTorch


**WaveOpticsTorch** is a Python project based on Pytorch framework that allows to place various optical elements (lenses, slits, apertures, etc.) and perform wave optics propagation to simulate phenomena such as diffraction and compute PSF (Point Spread Function) with GPU.

## How to use?
1. Import necessary tools.  
```python
import torch
import systems.elements as elem
from systems.systems import OpticalSystem, Field
```  
    - systems.elements (elem) has several optical modules (e.g. propagation, lens, pupils, and so on..)
    - Field is the object that contains the grid parameter, wavelengths, and field.  
    - OpticalSystem class is used for the inheritance. With this class, we can automatically initialize some parameters given hyperparameters.  


2. Construct optical elements you need.  

```python
class Diffraction(OpticalSystem):
    def __init__(
        self,
        pixel_size=[1, 1],
        pixel_num=[1000, 1000],
        lamb0=[0.55, 1.05, 1.550],
        refractive_index=1,
        focal_length=19*1e3,
        NA=0.3,
        pupil_type='circle',
        pupil_width=20,
        nyquist_spatial_bound=True,
    ):
        super(Diffraction, self).__init__(
            pixel_size=pixel_size,
            pixel_num=pixel_num,
            lamb0=lamb0,
            refractive_index=refractive_index
        )
        self.pupil_type = pupil_type
        self.focal_length = focal_length
        self.source = elem.PlaneSource(
            amplitude=1.0,
            ref_idx=self.refractive_index,
            dir_factors=None, # center.
            power=1.0,
        )
        self.pupil_mask = elem.CirclePupil(self.x_grid, self.y_grid, pupil_width) if pupil_type=='circle' else elem.SquarePupil(self.x_grid, self.y_grid, pupil_width)
        self.prop = elem.ASMPropagation(
            z=focal_length,
            ref_idx=self.refractive_index,
            band_limited=True
        )
        
        self.sensor = elem.Sensor(shot_noise_modes=[], clip=[1e-20, 1e+9], channel_sum=False)    
```  
    - The elem's components are defined by inheriting from nn.Module. We can use these optical elements in Pytorch-friendly way.
    - OpticalSystem class also inherits from nn.Module.  


3. Define the forward function
```python
def forward(self):
    field = Field(lamb0=self.lamb0, x_grid=self.x_grid, y_grid=self.y_grid, fx_grid=self.fx_grid, fy_grid=self.fy_grid)
    src_field = self.source(field)
    pupiled_field = self.pupil_mask(src_field)  
    prop_field = self.prop(pupiled_field) # asm
    if isinstance(prop_field, list) or isinstance(prop_field, tuple): # SASPropagation.
        prop_field, pixel_size = prop_field
    out = self.sensor(prop_field)
    return out
```  

    - You have to first define the Field instance. In the Field instance, the internal components (e.g. field tensor and grid tensor) are updated during performing the given operations.  
    - The field is initially updated from the defined source!  

## Contributors
- Joonhyuk Seo: [@yhy258](https://github.com/yhy258)
- Chanik Kang: [@latteishorse](https://github.com/latteishorse)

## Contact
Feel free to reach me out through (joonhyuk.seow@gmail.com).  


## TODO List

- [X] Verificaiton of utilization of GPU devices
- [ ] Parallelization  
- [ ] Scalable ASM (SAS)  
- [ ] Various phase initialization methods.  
- [ ] Considering magnification.  
- [ ] Considering shifted locations of sources or fields.  
- [ ] Considering various directions of the input fields.  
- [ ] More various examples.  