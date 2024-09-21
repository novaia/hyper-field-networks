import jax
import jax.numpy as jnp
import fp_tokenization as fpt

def main():
    a = jnp.array([0.123, 2.445, 7.92333], dtype=jnp.float32)
    print(a)
    b = fpt.fp32_to_bitfield16(a)
    print(b)
    print(b.shape)
    c = fpt.bitfield16_to_fp32(b)
    print(c)

if __name__ == '__main__':
    main()
