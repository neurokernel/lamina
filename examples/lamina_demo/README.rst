Result Files
------------
All file names will be suffixed with a text specified in configuration
which by default is empty, unless it is stated otherwise.

*   grid_dima.h5/grid_dimb.h5: coordinates of screen grid 
    where the image is projected (not subject to a suffix)

*   retina_elev<id>.h5/retina_azim<id>.h5: spherical coordinates of ommatidia
    on retina, id is a numeric identifier of the retina

*   retina_dima<id>.h5/retina_dimb<id>.h5: coordinates of ommatidia
    on screen, id is a numeric identifier of the retina

*   intensities.h5: values of input on screen points


*   retina_input<id>.h5: inputs of retina, id
    is a numeric indentifier of the retina, in case there are more than 1
    (subject to a suffix)

*   retina_output<id>_gpot.h5: graded potential outputs of retina, id
    is a numeric indentifier of the retina, in case there are more than 1
    (subject to a suffix which will be appended before _gpot) 

*   lamina_output<id>_gpot.h5: graded potential outputs of lamina, id
    is a numeric indentifier of the lamina, in case there are more than 1
    (subject to a suffix which will be appended before _gpot)
