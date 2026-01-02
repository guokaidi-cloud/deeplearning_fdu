#ifndef LIYIN_LABELS_HPP
#define LIYIN_LABELS_HPP

#include "opencv2/core/core.hpp"
#include <string>
#include <vector>

using namespace std;

class LiyinLabels {
public:
  LiyinLabels() {
    mLabels = {
        "Using Computer",        // 0
        "Listening Attentively", // 1
        "Taking Notes",          // 2
        "Using Phone",           // 3
        "Bowing the Head",       // 4
        "Lying on the Desk"      // 5
    };
  }

  string get_label(int i) {
    if (i >= 0 && i < (int)mLabels.size()) {
      return mLabels[i];
    }
    return "Unknown";
  }

  int num_classes() { return mLabels.size(); }

  cv::Scalar get_color(int i) {
    // 为每个类别定义固定颜色
    static vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),   // 0: 蓝色 - Using Computer
        cv::Scalar(0, 255, 0),   // 1: 绿色 - Listening Attentively
        cv::Scalar(0, 0, 255),   // 2: 红色 - Taking Notes
        cv::Scalar(255, 255, 0), // 3: 青色 - Using Phone
        cv::Scalar(255, 0, 255), // 4: 紫色 - Bowing the Head
        cv::Scalar(0, 255, 255)  // 5: 黄色 - Lying on the Desk
    };

    if (i >= 0 && i < (int)colors.size()) {
      return colors[i];
    }
    return cv::Scalar(128, 128, 128); // 灰色作为默认
  }

  cv::Scalar get_inverse_color(cv::Scalar color) {
    int blue = 255 - color[0];
    int green = 255 - color[1];
    int red = 255 - color[2];
    return cv::Scalar(blue, green, red);
  }

private:
  vector<string> mLabels;
};

#endif // LIYIN_LABELS_HPP
