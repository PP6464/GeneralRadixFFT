import numpy as np

tau = 2 * np.pi

def bluestein_fft(x, custom_w=None):
    N = len(x)
    L = 2 ** np.log2(np.ceil(2 * N + 1))

    w = np.exp(-1j * tau / N)

    if custom_w is not None:
        w = custom_w

    newW = np.exp(np.log(w) * N/L)

    u = np.array([0] * L)
    v = np.array([0] * L)
    vStar = np.array([0] * N)

    for i in range(N):
        u[i] = x[i] * w ** (i * i / 2)
        v[i] = w ** (-i * i / 2)
        vStar[i] = w ** (i * i / 2)

        if i > 0:
            v[-i] = v[i]

    # The following FFT calls are radix-2 so should not have an infinite recursion problem
    fft_u = fft(u, newW)
    fft_v = fft(v, newW)
    conv_res = ifft(fft_u * fft_v, 1 / newW)

    res = np.array([0] * N)

    for i in range(N):
        res[i] = conv_res[i] * vStar[i]

    return res

def fft(x, custom_w=None):
    N = len(x)

    if N == 1:
        return x

    w = np.exp(-1j * tau / N)

    if custom_w is not None:
        w = custom_w

    # Find any factor that we can take out of N (to recursively compute the FFT)
    A = 1

    if N % 2 == 0:
        A = 2
    elif N % 3 == 0:
        A = 3
    elif N % 5 == 0:
        A = 5

    # Recursively do the computation required if you can meaningfully reduce the list
    if A != 1:
        x_chunks = []

        for k in range(A):
            x_chunks.append(x[k::A])

        fft_chunks = [fft(i) for i in x_chunks]

        res = [0] * N

        for c in range(A):
            W_c = np.exp(-1j * c * tau / A)

            for k in range(int(N/A)):
                fft_vals = [chunk[k] for chunk in fft_chunks]
                fft_vals_mul = [i[1] * W_c ** i[0] for i in enumerate(fft_vals)]
                res[k] = sum(fft_vals_mul)

        return res
    else:
        return bluestein_fft(x, w)

def ifft(x, custom_w):
    w = np.exp(1j * tau / len(x))

    if custom_w is not None:
        w = custom_w

    return fft(x, w)


print(fft([0, 1]))
