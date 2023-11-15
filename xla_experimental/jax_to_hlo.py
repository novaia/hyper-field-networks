import jax
import jax.numpy as jnp

def target_fn(x):
    return x**2

def main():
    hlo_comp = jax.xla_computation(target_fn)(jnp.ones((1,)))
    hlo_comp_proto = str(hlo_comp.as_serialized_hlo_module_proto())
    hlo_comp_text = hlo_comp.as_hlo_text()
    with open('hlo_comp_proto.txt', 'w') as f:
        f.write(hlo_comp_proto)
    with open('hlo_comp_text.txt', 'w') as f:
        f.write(hlo_comp_text)
    
    print('Saved the following HLO computation:\n')
    print(hlo_comp.as_hlo_text())

if __name__ == '__main__':
    main()