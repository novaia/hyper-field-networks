from fp_conversion import cpp_ffi

def float_to_token(f:float):
    return cpp_ffi.float_to_token(f)

def token_to_float(t:int):
    return cpp_ffi.token_to_float(t)
