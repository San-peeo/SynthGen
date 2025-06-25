from lib.lib_dep import *

def MapPlotting(parent=None, values=[], region=None, proj_opt=ccrs.Mollweide(), title='', cb_label='', cmap='jet', clim=None):

    """
    Usage
    ----------
    Plot a 2D map on a specified Cartopy projection with colorbar and gridlines.
    Designed for use with matplotlib and Cartopy axes.
    Allows for region selection and customization of title, colorbar label, and colormap.

    Parameters
    ----------
    parent      : list
                  [fig, ax] where fig is the matplotlib Figure and ax is the Cartopy GeoAxes.
    values      : numpy.ndarray or SHGrid
                  SHGrid or 2D array of values to plot (e.g., topography, gravity).
    region      : list, optional
                  [[lon_min, lon_max], [lat_min, lat_max]]; region to display on the map.
                  If None, the global map is shown.
    proj_opt    : cartopy.crs, optional
                  Cartopy projection for the plot (default: ccrs.Mollweide()).
    title       : str, optional
                  Title for the subplot.
    cblabel     : str, optional
                  Label for the colorbar.
    cmap        : str, optional
                  Colormap for the plot (default: 'jet').
    clim        : tuple, optional
                  Color limits for the plot (vmin, vmax). If None, the limits are determined

    Output
    ----------
    None
        The function draws the map on the provided axis, adds gridlines and a colorbar.
    """

    

    if region is None and isinstance(values, pysh.SHGrid):
        # Global map plotting through pyShtools routine
        if parent is not None:
            values.plot(ax=parent[1],colorbar='right',projection=proj_opt,title=title, cb_label=cb_label,cmap=cmap,cmap_limits=clim)
            return None
        else:
            fig,ax = values.plot(colorbar='right',projection=proj_opt,title=title, cb_label=cb_label,cmap=cmap,cmap_limits=clim)
            return fig,ax


    else:  
        # Region-specific map plotting using Cartopy (custom plot)
        if parent is None:
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=proj_opt)
            
        else:
            fig = parent[0]
            ax = parent[1]

        ax.set_global()


        # Check if values is a SHGrid or numpy array
        if isinstance(values, pysh.SHGrid):
            values = values.data     # Extract data from SHGrid to numpy array
        elif isinstance(values, np.ndarray):
            pass   # Ensure values is a numpy array
        else:
            raise TypeError("values must be a SHGrid or numpy.ndarray")


        # Create longitude and latitude arrays for masking and plotting
        nlat, nlon = values.shape
        lon = np.linspace(0, 360, nlon)
        lat = np.flip(np.linspace(-90,90, nlat))
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Apply region mask if region is specified
        if region is not None:
            mask = (
                (lon_grid >= region[0][0]) & (lon_grid <= region[0][1]) &
                (lat_grid >= region[1][0]) & (lat_grid <= region[1][1])
            )
            values_masked = np.where(mask, values, np.nan)
            vmin = np.nanmin(values_masked)
            vmax = np.nanmax(values_masked)
            ax.set_extent([region[0][0], region[0][1], region[1][0], region[1][1]])

        else:
            if clim is not None:
                values_masked = values
                vmin = np.nanmin(values)
                vmax = np.nanmax(values)
            else:
                values_masked = values
                vmin = clim[0]
                vmax = clim[1]

        # Plotting
        map1 = ax.pcolormesh(lon_grid, lat_grid, values, transform=ccrs.PlateCarree(),
                            cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

        # Region selection
        if region is not None:
            ax.set_extent([region[0][0], region[0][1], region[1][0], region[1][1]],crs=ccrs.PlateCarree())


        gl = ax.gridlines(draw_labels=True, dms=True,linewidth=0.75, color='gray', alpha=0.5, linestyle='--')

        ax.set_title(title)
        fig.colorbar(map1,ax=ax,label=cb_label, location='right', anchor=(0.5,0.5), shrink=0.7)


        return fig,ax


##########################################################################################################################
##########################################################################################################################
