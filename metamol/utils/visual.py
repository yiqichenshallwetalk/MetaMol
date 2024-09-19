import os
import tempfile

__all__ = ["visual", "visual_py3Dmol"]

def visual(obj, backend='py3Dmol', params={}, inter=False, bonds=True, frameId=1, timestep=None):
    """ Visualize the system/molecule"""

    if backend == 'py3Dmol':
        visualize = visual_py3Dmol
    else:
        raise NotImplemented('Visualization with {} not implemented'.format(backend))
    
    if not inter:
        viewer = visualize(obj, params, bonds=bonds, frameId=frameId, timestep=timestep)
        viewer.show()
    else:
        from ipywidgets import interact, FloatSlider, Dropdown

        def inter_selector(sf, st):
            if st=='line':
                def line(lw):
                    return visualize(params={'surface':sf, 'style': st, 'linewidth': lw}, frame=frame)
                interact(line, lw=FloatSlider(min=0, max=10.0, step=1.0, 
                                            value=3.0, description='Linewidth'))

            elif st=='stick':
                def stick(radius):
                    return visualize(params={'surface':sf, 'style': st, 'radius': radius}, frame=frame)
                interact(stick, radius=FloatSlider(min=0.0, max=1.0, step=0.05,
                                                    value=0.2, description='Radius'))

            elif st=='sphere':
                def sphere(scale):
                    return visualize(params={'surface':sf, 'style': st, 'scale': scale}, frame=frame)
                interact(sphere, scale=FloatSlider(min=0.0, max=2.0, step=0.1,
                                                    value=1.0, description='Scale'))

            elif st=='ball and stick':
                def ball_stick(radius, scale):
                    return visualize(params={'surface':sf, 'style': st, 'radius': radius, 'scale': scale}, frame=frame)
                interact(ball_stick, 
                        radius=FloatSlider(min=0.0, max=1.0, step=0.05,
                                                    value=0.2, description='Radius'),
                        scale=FloatSlider(min=0.0, max=1.0, step=0.05,
                                                    value=0.3, description='Scale')
                        )

            elif st=='cartoon':
                def cartoon(thick):
                    return visualize(params={'surface':sf, 'style': st, 'thickness': thick}, frame=frame)
                interact(cartoon, thick=FloatSlider(min=0.0, max=3.0, step=0.2,
                                                    value=0.4, description='Thickness'))

        
        return interact(inter_selector, 
                        sf=Dropdown(
                            options=['No Surface', 'VDW', 'SAS', 'MS', 'SES'],
                            value='No Surface',
                            description='Surface'),
                        st=Dropdown(
                            options=['ball and stick', 'line', 'stick', 'sphere', 'cartoon'],
                            value='ball and stick',
                            description='Style'))

def visual_py3Dmol(obj, params, bonds=True, frameId=1, timestep=None):
    py3Dmol = __import__("py3Dmol")

    if 'size' in params:
        size = params['size']
        if not isinstance(size, tuple):
            raise TypeError("Size of py3Dmol plot must be a tuple")
        if len(size) != 2:
            raise ValueError("Size of py3Dmol plot must be of length 2")
    else:
        size = (600,300)
    
    if 'style' in params:
        style = params['style']

        if style.lower() == 'line':
            if 'linewidth' in params:
                lw = params['linewidth']
                if not isinstance(lw, (float, int)) or lw < 0:
                    raise TypeError("linewidth must be a positive number.")
            else:
                lw = 2
            st_dict = {'line': {'linewidth': lw}}

        elif style.lower() == 'stick':
            if 'radius' in params:
                radius = params['radius']
                if not isinstance(radius, (float, int)) or radius < 0:
                    raise TypeError("radius must be a positive number.")
            else:
                radius = 0.2
            st_dict = {'stick': {'radius': radius}}

        elif style.lower() == 'sphere':
            if 'scale' in params:
                scale = params['scale']
                if not isinstance(scale, (float, int)) or scale < 0:
                    raise TypeError("sacle must be a positive number.")
            else:
                scale = 1
            st_dict = {'sphere': {'scale': scale}}
        
        elif style.lower() == 'ball and stick':
            if 'radius' in params:
                radius = params['radius']
                if not isinstance(radius, (float, int)) or radius < 0:
                    raise TypeError("radius must be a positive number.")
            else:
                radius = 0.2
            
            if 'scale' in params:
                scale = params['scale']
                if not isinstance(scale, (float, int)) or scale < 0:
                    raise TypeError("sacle must be a positive number.")
            else:
                scale = 0.3
                
            st_dict = {
                'stick': {'radius': radius},
                'sphere': {'scale': scale}
            }                

        elif style.lower() == 'cartoon':
            if 'thickness' in params:
                tk = params['thickness']
                if not isinstance(tk, (float, int)) or tk < 0:
                    raise TypeError("thickness must be a positive number.")
            else:
                tk = 0.4
            st_dict = {'cartoon': {'thickness': tk}}

        else:
            raise ValueError('Visualization style {} not supported'.format(style))
    else:
        # Set default style to `ball and stick``
        st_dict = {
            'stick': {'radius': 0.2},
            'sphere': {'scale': 0.3}
            }

    if 'surface' in params:
        surface = params['surface']
        if surface=='No Surface': surface = None
        if surface:
            if not isinstance(surface, str):
                raise TypeError("The surface parameter must be a string type")

            if surface.upper() == 'VDW':
                surf_type = py3Dmol.VDW
            elif surface.upper() == 'SAS':
                surf_type = py3Dmol.SAS
            elif surface.upper() == 'MS':
                surf_type = py3Dmol.MS
            elif surface.upper() == 'SES':
                surf_type = py3Dmol.SES
            else:
                raise ValueError("Surface type {} not supported".format(surface))
        
            if 'opacity' in params:
                opacity = params['opacity']
                if not isinstance(opacity, (float, int)) or opacity<0.0 or opacity>1.0:
                    raise ValueError("opacity must be a number between 0.0 and 1.0")
            else:
                opacity = 0.5
    else:
        surface = None

    tmp_dir = tempfile.mkdtemp()
    obj.save(os.path.join(tmp_dir, "tmp.mol2"), frameId=frameId, timestep=timestep, bonds=bonds)

    viewer = py3Dmol.view(width=size[0], height=size[1])
    with open(os.path.join(tmp_dir, "tmp.mol2"), "r") as f:
        viewer.addModel(f.read(), "mol2", keepH=True)
    viewer.setStyle(st_dict)

    if surface:
        viewer.addSurface(surf_type, {'opacity': opacity})
    viewer.zoomTo()
    return viewer