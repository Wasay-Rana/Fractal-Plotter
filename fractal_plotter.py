import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

# Settings
width, height = 800, 450
offset = np.array([1.3 * width, height]) // 2

@ti.data_oriented
class FractalPlotter:
    def __init__(self):
        ti.init(arch=ti.cpu)
        # Taichi fields
        self.screen_field = ti.Vector.field(3, ti.uint32, (width, height))
        # Control settings
        self.velocity = 0.01
        self.zoom_factor, self.scale_factor = 2.2 / height, 0.993
        self.translation = ti.Vector([0.0, 0.0])
        self.max_iterations, self.max_iterations_limit = 30, 5500

    @ti.kernel
    def render_mandelbrot(self, max_iterations: ti.int32, zoom: ti.float32, dx: ti.float32, dy: ti.float32):
        for x, y in self.screen_field: # Parallelization loop
            c = ti.Vector([(x - offset[0]) * zoom - dx, (y - offset[1]) * zoom - dy])
            z = ti.Vector([0.0, 0.0])
            num_iterations = 0
            for i in range(max_iterations):
                z = ti.Vector([(z.x ** 2 - z.y ** 2 + c.x), (2 * z.x * z.y + c.y)])
                if z.dot(z) > 4:
                    break
                num_iterations += 1
            color = int(255 * num_iterations / max_iterations)
            self.screen_field[x, y] = ti.Vector([color, color, color])

    @ti.kernel
    def render_julia(self, max_iterations: ti.int32, zoom: ti.float32, dx: ti.float32, dy: ti.float32):
        for x, y in self.screen_field: # Parallelization loop
            z = ti.Vector([(x - offset[0]) * zoom - dx, (y - offset[1]) * zoom - dy])
            c = ti.Vector([-0.7, 0.27015])
            num_iterations = 0
            for i in range(max_iterations):
                z = ti.Vector([(z.x ** 2 - z.y ** 2 + c.x), (2 * z.x * z.y + c.y)])
                if z.dot(z) > 4:
                    break
                num_iterations += 1
            color = int(255 * num_iterations / max_iterations)
            self.screen_field[x, y] = ti.Vector([color, color, color])

    @ti.kernel
    def render_burning_ship(self, max_iterations: ti.int32, zoom: ti.float32, dx: ti.float32, dy: ti.float32):
        for x, y in self.screen_field: # Parallelization loop
            c = ti.Vector([(x - offset[0]) * zoom - dx, (y - offset[1]) * zoom - dy])
            z = ti.Vector([0.0, 0.0])
            num_iterations = 0
            for i in range(max_iterations):
                z = ti.Vector([abs(z.x), abs(z.y)]) ** 2 + c
                if z.dot(z) > 4:
                    break
                num_iterations += 1
            color = int(255 * num_iterations / max_iterations)
            self.screen_field[x, y] = ti.Vector([color, color, color])

    @ti.kernel
    def render_multibrot(self, max_iterations: ti.int32, zoom: ti.float32, dx: ti.float32, dy: ti.float32):
        for x, y in self.screen_field: # Parallelization loop
            c = ti.Vector([(x - offset[0]) * zoom - dx, (y - offset[1]) * zoom - dy])
            z = ti.Vector([0.0, 0.0])
            num_iterations = 0
            for i in range(max_iterations):
                z = ti.Vector([z.x ** 3 - 3 * z.x * z.y ** 2, 3 * z.x ** 2 * z.y - z.y ** 3]) + c
                if z.dot(z) > 4:
                    break
                num_iterations += 1
            color = int(255 * num_iterations / max_iterations)
            self.screen_field[x, y] = ti.Vector([color, color, color])

    def update_and_draw(self, render_function):
        render_function(self.max_iterations, self.zoom_factor, self.translation[0], self.translation[1])
        return self.screen_field.to_numpy()

    def run(self):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(self.update_and_draw(self.render_mandelbrot), cmap='inferno', extent=(-2.2, 1.2, -1.5, 1.5))
        plt.title('Mandelbrot Fractal')

        plt.subplot(2, 2, 2)
        plt.imshow(self.update_and_draw(self.render_julia), cmap='inferno', extent=(-2.2, 1.2, -1.5, 1.5))
        plt.title('Julia Fractal')

        plt.subplot(2, 2, 3)
        plt.imshow(self.update_and_draw(self.render_burning_ship), cmap='inferno', extent=(-2.2, 1.2, -1.5, 1.5))
        plt.title('Burning Ship Fractal')

        plt.subplot(2, 2, 4)
        plt.imshow(self.update_and_draw(self.render_multibrot), cmap='inferno', extent=(-2.2, 1.2, -1.5, 1.5))
        plt.title('Multibrot Fractal')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    fractal_plotter = FractalPlotter()
    fractal_plotter.run()
