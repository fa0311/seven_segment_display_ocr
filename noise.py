import random

import numpy as np

# https://qiita.com/Cartelet/items/9fcf3890a9ac59e1fd1f


class PerlinNoise:
    def fade(self, t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def lerp(self, a, b, t):
        return a + self.fade(t) * (b - a)

    def perlin(self, r):
        ri = np.floor(r).astype(int)
        ri[0] -= ri[0].min()
        ri[1] -= ri[1].min()
        rf = np.array(r) % 1
        g = 2 * np.random.rand(ri[0].max() + 2, ri[1].max() + 2, 2) - 1
        e = np.array([[[[0, 0], [0, 1], [1, 0], [1, 1]]]])
        er = (np.array([rf]).transpose(2, 3, 0, 1) - e).reshape(
            r.shape[1], r.shape[2], 4, 1, 2
        )
        gr = (
            np.r_[
                "3,4,0",
                g[ri[0], ri[1]],
                g[ri[0], ri[1] + 1],
                g[ri[0] + 1, ri[1]],
                g[ri[0] + 1, ri[1] + 1],
            ]
            .transpose(0, 1, 3, 2)
            .reshape(r.shape[1], r.shape[2], 4, 2, 1)
        )
        p = (er @ gr).reshape(r.shape[1], r.shape[2], 4).transpose(2, 0, 1)

        return self.lerp(
            self.lerp(p[0], p[2], rf[0]), self.lerp(p[1], p[3], rf[0]), rf[1]
        )

    def generate(self, N=512, scale=1, count=1):
        y = np.zeros((N, N))
        for i in range(1, count + 1):
            x = np.linspace(0, scale, N)
            r = np.array(np.meshgrid(x, x))
            y += self.perlin(r)
        y = (y - y.min()) / (y.max() - y.min())
        return y


class NoiseImage:
    def __init__(self):
        self.perlin = PerlinNoise()

    def generate(self, N=512, count=1):
        base = np.random.rand(3) * 255
        for i in range(1, count + 1):
            scale = random.randint(1, 10 * (count + 1))
            y = self.perlin.generate(N=N, scale=scale)
            color2 = np.random.rand(3) * 255
            alpha = np.dstack([y, y, y])
            base = (1 - alpha) * base + alpha * color2
        return base
