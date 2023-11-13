import jax
import jax.numpy as jnp

def target_fn(x):
    return jnp.sin(x)

def main():
    hlo_comp = jax.xla_computation(target_fn)(jnp.ones((1,)))
    with open('hlo_comp.txt', 'w') as f:
        f.write(hlo_comp.as_hlo_text())
    print(hlo_comp.as_hlo_text())

if __name__ == '__main__':
    main()