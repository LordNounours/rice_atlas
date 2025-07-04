#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

py::array_t<uint8_t> suppress_neighbors(py::array_t<uint8_t> image, int threshold) {
    auto buf = image.request();
    if (buf.ndim != 2)
        throw std::runtime_error("L'image doit être 2D");

    ssize_t rows = buf.shape[0];
    ssize_t cols = buf.shape[1];
    uint8_t* input = static_cast<uint8_t*>(buf.ptr);

    // Image des voisins
    py::array_t<uint8_t> neighbors({rows, cols});
    uint8_t* output = static_cast<uint8_t*>(neighbors.request().ptr);

    // Initialisation explicite à zéro
    std::fill(output, output + rows * cols, 0);

    for (ssize_t i = 1; i < rows - 1; ++i) {
        for (ssize_t j = 1; j < cols - 1; ++j) {
            int idx = i * cols + j;
            if (input[idx] >= threshold) {
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        ssize_t ni = i + di;
                        ssize_t nj = j + dj;
                        output[ni * cols + nj] = 255;
                    }
                }
            }
        }
    }

    return neighbors;
}

PYBIND11_MODULE(denoise, m) {
    m.doc() = "Suppression de voisins (prétraitement)";
    m.def("suppress_neighbors", &suppress_neighbors, "Supprime les voisins actifs",
          py::arg("image"), py::arg("threshold") = 120);
}
