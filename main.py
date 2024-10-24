import numpy as np

tau = 2 * np.pi


def bluestein_fft(x, custom_w=None):
    N = len(x)
    L = int(2 ** np.log2(np.ceil(2 * N + 1)))

    w = np.exp(-1j * tau / N)

    if custom_w is not None:
        w = custom_w

    newW = np.exp(np.log(w) * N/L)

    u = np.array([0j] * L)
    v = np.array([0j] * L)
    vStar = np.array([0j] * N)

    for i in range(N):
        u[i] = x[i] * w ** (i * i / 2)
        v[i] = w ** (-i * i / 2)
        vStar[i] = w ** (i * i / 2)

        if i > 0:
            v[-i] = v[i]

    # The following FFT calls are radix-2 so should not have an infinite recursion problem
    fft_u = fft(u, newW)
    fft_v = fft(v, newW)
    conv_res = ifft(np.array(fft_u) * np.array(fft_v), 1 / newW)

    res = np.array([0j] * N)

    for i in range(N):
        res[i] = conv_res[i] * vStar[i]

    return res.tolist()


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

        fft_chunks = [fft(i, w ** A) for i in x_chunks]

        res = [0] * N

        for c in range(A):
            for k in range(int(N/A)):
                fft_vals = [chunk[k] for chunk in fft_chunks]
                fft_vals = [val * w ** (k * index) for index, val in enumerate(fft_vals)]
                fft_vals = [val * np.exp(-1j * tau * c * index / A) for index, val in enumerate(fft_vals)]
                res[k + c * int(N/A)] = sum(fft_vals)
        return res
    else:
        return bluestein_fft(x, w)


def ifft(x, custom_w=None):
    w = np.exp(1j * tau / len(x))

    if custom_w is not None:
        w = custom_w

    return [i/len(x) for i in fft(x, w)]


print("\n".join([str(i) for i in fft([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]))
