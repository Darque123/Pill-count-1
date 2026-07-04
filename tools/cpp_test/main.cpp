// Off-device regression harness for the shipping C++ pipeline
// (PillCount/Detection/PillPipeline.cpp).
//
// Reads fixture images + a truth manifest produced by:
//     python3 tools/test_synthetic.py --dump tools/fixtures
// Build & run (needs a desktop OpenCV, e.g. `apt install libopencv-dev`):
//     g++ -std=c++17 -O2 tools/cpp_test/main.cpp \
//         PillCount/Detection/PillPipeline.cpp \
//         -IPillCount/Detection $(pkg-config --cflags --libs opencv4) \
//         -o /tmp/pill_test && /tmp/pill_test tools/fixtures

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "PillPipeline.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <fixtures-dir>\n";
        return 2;
    }
    const std::string dir = argv[1];
    std::ifstream manifest(dir + "/manifest.csv");
    if (!manifest) {
        std::cerr << "cannot open " << dir << "/manifest.csv\n";
        return 2;
    }

    pillcount::Params params;
    int failures = 0, total = 0;
    std::string line;
    while (std::getline(manifest, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string file, truthStr;
        std::getline(ss, file, ',');
        std::getline(ss, truthStr, ',');
        int truth = std::stoi(truthStr);

        cv::Mat img = cv::imread(dir + "/" + file, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "FAIL  " << file << ": cannot read image\n";
            ++failures;
            ++total;
            continue;
        }
        auto t0 = std::chrono::steady_clock::now();
        auto result = pillcount::detectPills(img, params);
        auto ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        int got = static_cast<int>(result.pills.size());
        bool ok = got == truth;
        failures += ok ? 0 : 1;
        ++total;
        std::printf("%s  %-28s truth=%3d got=%3d  (%.1f ms)\n",
                    ok ? "PASS" : "FAIL", file.c_str(), truth, got, ms);
    }
    std::printf("%d/%d passed\n", total - failures, total);
    return failures ? 1 : 0;
}
