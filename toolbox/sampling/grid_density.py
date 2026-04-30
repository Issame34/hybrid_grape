import jax


def sample_with_grid_density(key, sample_size, grid_pts, grid_ds, densities):
    """
    Sample points around a uniform multidimensional grid (given by "grid_pts")
        with a given "densities" associated with the grid points "grid_pts"
        The elementary cell size around the greed points is given by "grid_ds".
    
    Array shapes:
        grid_pts --> (N, M),               
        grid_ds --> M,
        densities --> N.
        Here N is the number of grid points and M is the grid dimension.
        
    Note that N and M have nothing to do with the number of the sampled points
        gived by "sample_size".
    """
    key1, key2 = jax.random.split(key)
    slots = jax.random.choice(key1,
                              grid_pts,
                              shape=(sample_size,),
                              p=densities) 
    
    crds_in_slot = jax.random.uniform(key2,
                                      shape=(sample_size, len(grid_ds)),
                                      minval=-0.5, maxval=0.5) * grid_ds 
    
    return slots + crds_in_slot