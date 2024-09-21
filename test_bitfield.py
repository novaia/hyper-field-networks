import jax
import jax.numpy as jnp
import fp_tokenization as fpt

def main():
    a = jnp.array([0.123, 2.445, 7.92333], dtype=jnp.float32)
    print(a)
    b = fpt.to_bitfield(a)
    print(b)
    print(b.shape)

if __name__ == '__main__':
    main()
