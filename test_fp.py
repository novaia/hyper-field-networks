import fp_conversion

def test(f):
    print('f', f)
    token = fp_conversion.float_to_token(f)
    print('token', token)
    f_hat = fp_conversion.token_to_float(token)
    print('f_hat', f_hat)

test(20)
test(-20)
