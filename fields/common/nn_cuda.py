from jax import numpy as jnp
import flax.linen as nn
from jaxtcnn import hashgrid_encode, HashGridMetadata

class TcnnMultiResolutionHashEncoding(nn.Module):
    table_size: int
    num_levels: int
    min_resolution: int
    max_resolution: int
    feature_dim: int

    def setup(self):
        self.levels = jnp.arange(self.num_levels)
        self.hash_offset = self.levels * self.table_size
        self.hash_offset = jnp.concatenate([
            self.hash_offset, jnp.array([self.hash_offset[-1] + self.table_size])
        ], axis=0)
        self.spatial_dim = 3
        if self.num_levels > 1:
            self.growth_factor = jnp.exp(
                (jnp.log(self.max_resolution) - jnp.log(self.min_resolution)) 
                / (self.num_levels - 1)
            )
        else:
            self.growth_factor = 1.0
        self.scalings = jnp.floor(self.min_resolution * self.growth_factor**self.levels)
        self.scalings = jnp.reshape(self.scalings, (self.scalings.shape[0], 1))
        absolute_table_size = self.table_size * self.num_levels
        self.hash_table = self.param(
            'hash_table', 
            nn.initializers.uniform(scale=10**-4), 
            (absolute_table_size, self.feature_dim,)
        )

    def __call__(self, x):
        _growth_factor = math.exp(
            (math.log(self.max_resolution) - math.log(self.min_resolution)) 
            / (self.num_levels - 1)
        )
        encoded_position = hashgrid_encode(
            desc=HashGridMetadata(
                L=int(self.num_levels),
                F=int(self.feature_dim),
                N_min=int(self.min_resolution),
                per_level_scale=_growth_factor
            ),
            offset_table_data=jnp.asarray(self.hash_offset, jnp.uint32),
            coords_rm=x.T,
            params=self.hash_table
        )
        return encoded_position.T
 
