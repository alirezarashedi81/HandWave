#include <pybind11/pybind11.h>
#include <cmath>

namespace py = pybind11;

class ClickProcessor {
public:
    ClickProcessor(double alpha, double dead_zone_threshold) {
        this->alpha = alpha;
        this->dead_zone_threshold = dead_zone_threshold;
        this->prev_smooth_mx = -1.0;
        this->prev_smooth_my = -1.0;
        this->prev_raw_mx = -1.0;
        this->prev_raw_my = -1.0;
    }

    py::tuple process(int w, int h,
                      double thumb_x, double thumb_y,
                      double index_x, double index_y,
                      double middle_x, double middle_y,
                      double pinky_x, double pinky_y) {
        
        // Convert to pixel coords
        int tx = int(thumb_x * w);
        int ty = int(thumb_y * h);
        int ix = int(index_x * w);
        int iy = int(index_y * h);
        int mx = int(middle_x * w);
        int my = int(middle_y * h);
        int px = int(pinky_x * w);
        int py = int(pinky_y * h);

        // Dead zone check on RAW pixel coordinates
        bool dead = false;
        if (prev_raw_mx >= 0.0 && prev_raw_my >= 0.0) {
            double d = distance(mx, my, prev_raw_mx, prev_raw_my);
            dead = (d < dead_zone_threshold);
        }
        // Update previous raw coordinates
        prev_raw_mx = mx;
        prev_raw_my = my;

        // Smooth the middle finger (separate from dead zone)
        double smooth_mx = smooth(mx, prev_smooth_mx, alpha);
        double smooth_my = smooth(my, prev_smooth_my, alpha);
        prev_smooth_mx = smooth_mx;
        prev_smooth_my = smooth_my;

        // Distances
        double d1 = distance(tx, ty, ix, iy);
        double d2 = distance(tx, ty, px, py);

        return py::make_tuple(d1, d2, dead);
    }

    double get_smoothed_mx() const { return prev_smooth_mx; }
    double get_smoothed_my() const { return prev_smooth_my; }

    void reset() {
        prev_smooth_mx = -1.0;
        prev_smooth_my = -1.0;
        prev_raw_mx = -1.0;
        prev_raw_my = -1.0;
    }

private:
    double alpha;
    double dead_zone_threshold;
    double prev_smooth_mx;
    double prev_smooth_my;
    double prev_raw_mx;
    double prev_raw_my;

    double smooth(double current, double previous, double alpha) {
        if (previous < 0.0) return current;
        return alpha * current + (1.0 - alpha) * previous;
    }

    double distance(double x1, double y1, double x2, double y2) {
        double dx = x1 - x2;
        double dy = y1 - y2;
        return std::sqrt(dx * dx + dy * dy);
    }
};

PYBIND11_MODULE(click_utils, m) {
    py::class_<ClickProcessor>(m, "ClickProcessor")
        .def(py::init<double, double>(),
             py::arg("alpha") = 0.6,
             py::arg("dead_zone_threshold") = 0.0)
        .def("process", &ClickProcessor::process,
             py::arg("w"), py::arg("h"),
             py::arg("thumb_x"), py::arg("thumb_y"),
             py::arg("index_x"), py::arg("index_y"),
             py::arg("middle_x"), py::arg("middle_y"),
             py::arg("pinky_x"), py::arg("pinky_y"))
        .def("reset", &ClickProcessor::reset)
        .def_property_readonly("smoothed_mx", &ClickProcessor::get_smoothed_mx)
        .def_property_readonly("smoothed_my", &ClickProcessor::get_smoothed_my);
}
