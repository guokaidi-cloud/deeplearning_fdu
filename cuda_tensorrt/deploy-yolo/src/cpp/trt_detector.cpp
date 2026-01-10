#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"
#include "trt_logger.hpp"
#include "trt_model.hpp"
#include "utils.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <algorithm>
#include <string>

#include "liyin_labels.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc//imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "trt_detector.hpp"
#include "trt_preprocess.hpp"
#include "trt_postprocess.hpp"

using namespace std;
using namespace nvinfer1;

namespace model {

namespace detector {

float iou_calc(bbox bbox1, bbox bbox2) {
  auto inter_x0 = std::max(bbox1.x0, bbox2.x0);
  auto inter_y0 = std::max(bbox1.y0, bbox2.y0);
  auto inter_x1 = std::min(bbox1.x1, bbox2.x1);
  auto inter_y1 = std::min(bbox1.y1, bbox2.y1);

  // 修复：与 Python 一致，处理不相交的情况
  float inter_w = std::max(0.0f, inter_x1 - inter_x0);
  float inter_h = std::max(0.0f, inter_y1 - inter_y0);

  float inter_area = inter_w * inter_h;
  float union_area = (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0) +
                     (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0) - inter_area;

  // 防止除零
  if (union_area <= 0)
    return 0.0f;

  return inter_area / union_area;
}

void Detector::setup(void const *data, size_t size) {
  /*
   * detector setup需要做的事情
   *   创建engine, context
   *   设置bindings。这里需要注意，不同版本的yolo的输出binding可能还不一样
   *   分配memory空间。这里需要注意，不同版本的yolo的输出所需要的空间也还不一样
   */

  m_runtime = shared_ptr<IRuntime>(createInferRuntime(*m_logger),
                                   destroy_trt_ptr<IRuntime>);
  m_engine =
      shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(data, size),
                              destroy_trt_ptr<ICudaEngine>);
  m_context = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(),
                                            destroy_trt_ptr<IExecutionContext>);
  m_inputDims = m_context->getBindingDimensions(0);
  m_outputDims = m_context->getBindingDimensions(1);

  CUDA_CHECK(cudaStreamCreate(&m_stream));

  m_inputSize =
      m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
  m_imgArea = m_params->img.h * m_params->img.w;
  m_outputSize = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);

  // 这里对host和device上的memory一起分配空间
  CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
  CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], m_outputSize));
  CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
  CUDA_CHECK(cudaMalloc(&m_outputMemory[1], m_outputSize));

  // 创建m_bindings，之后再寻址就直接从这里找
  m_bindings[0] = m_inputMemory[1];
  m_bindings[1] = m_outputMemory[1];

  // 初始化 LetterBox scale
  m_letterbox_scale = 1.0f;

  // GPU 后处理内存分配
  m_max_boxes = 1000; // 最大检测框数量
  CUDA_CHECK(cudaMalloc(&m_d_decoded, m_max_boxes * 6 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&m_d_count, sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&m_h_decoded, m_max_boxes * 6 * sizeof(float)));
}

void Detector::reset_task() { m_bboxes.clear(); }

bool Detector::preprocess_cpu() {
  /*Preprocess -- yolo的预处理并没有mean和std，所以可以直接skip掉mean和std的计算
   */

  /*Preprocess -- 读取数据*/
  m_inputImage = cv::imread(m_imagePath);
  if (m_inputImage.data == nullptr) {
    LOGE("ERROR: Image file not founded! Program terminated");
    return false;
  }

  /*Preprocess -- 测速*/
  m_timer->start_cpu();

  /*Preprocess -- resize(默认是bilinear interpolation)*/
  cv::resize(m_inputImage, m_inputImage,
             cv::Size(m_params->img.w, m_params->img.h), 0, 0,
             cv::INTER_LINEAR);

  /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
  int index;
  int offset_ch0 = m_imgArea * 0;
  int offset_ch1 = m_imgArea * 1;
  int offset_ch2 = m_imgArea * 2;
  for (int i = 0; i < m_inputDims.d[2]; i++) {
    for (int j = 0; j < m_inputDims.d[3]; j++) {
      index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
      m_inputMemory[0][offset_ch2++] = m_inputImage.data[index + 0] / 255.0f;
      m_inputMemory[0][offset_ch1++] = m_inputImage.data[index + 1] / 255.0f;
      m_inputMemory[0][offset_ch0++] = m_inputImage.data[index + 2] / 255.0f;
    }
  }

  /*Preprocess -- 将host的数据移动到device上*/
  CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize,
                             cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

  m_timer->stop_cpu();
  m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");
  return true;
}

bool Detector::preprocess_gpu() {
  /*
   * LetterBox GPU 预处理（与 Python 版本完全一致）
   * 使用 CUDA kernel 完成：
   * 1. 计算 scale = min(tar_h / src_h, tar_w / src_w)
   * 2. GPU 上完成 resize + 填充 + BGR2RGB + HWC2CHW + normalize
   */

  /*Preprocess -- 读取数据*/
  m_inputImage = cv::imread(m_imagePath);
  if (m_inputImage.data == nullptr) {
    LOGE("ERROR: file not founded! Program terminated");
    return false;
  }

  /*Preprocess -- 测速*/
  m_timer->start_gpu();

  // 调用 GPU LetterBox 预处理（内部完成所有操作）
  preprocess::letterbox_resize_gpu(m_inputImage,
                                   m_inputMemory[1], // 直接写入 GPU 内存
                                   m_params->img.h, m_params->img.w,
                                   m_letterbox_scale // 输出 scale 供后处理使用
  );

  m_timer->stop_gpu();
  m_timer->duration_gpu("preprocess(GPU)");
  return true;
}

bool Detector::postprocess_cpu() {
  m_timer->start_cpu();

  /*Postprocess -- 将device上的数据移动到host上*/
  int output_size = m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);
  CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], output_size,
                             cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
  CUDA_CHECK(cudaStreamSynchronize(m_stream));

  /*Postprocess -- yolov8的postprocess需要做的事情*/
  /*
   * 1. 把bbox从输出tensor拿出来，并进行decode，把获取的bbox放入到m_bboxes中
   * 2. 把decode得到的m_bboxes根据nms threshold进行NMS处理
   * 3. 把最终得到的bbox绘制到原图中
   */

  float conf_threshold = 0.25; //用来过滤decode时的bboxes
  float nms_threshold = 0.45;  //用来过滤nms时的bboxes

  /*Postprocess -- 1. decode*/
  /*
   * 我们需要做的就是将[batch, bboxes, ch]转换为vector<bbox>
   * 几个步骤:
   * 1. 从每一个bbox中对应的ch中获取cx, cy, width, height
   * 2. 对每一个bbox中对应的ch中，找到最大的class label,
   * 可以使用std::max_element
   * 3. 将cx, cy, width, height转换为x0, y0, x1, y1
   * 4.
   * 因为图像是经过resize了的，所以需要根据resize的scale和shift进行坐标的转换(这里面可以根据preprocess中的到的affine
   * matrix来进行逆变换)
   * 5. 将转换好的x0, y0, x1,
   * y1，以及confidence和classness给存入到box中，并push到m_bboxes中，准备接下来的NMS处理
   */
  // 检测是否需要转置：如果 d[1] < d[2]，说明格式是 [1, channels, boxes]
  bool need_transpose = (m_outputDims.d[1] < m_outputDims.d[2]);

  int boxes_count, class_count, channels;
  if (need_transpose) {
    // 格式: [1, 4+classes, boxes] -> 需要转置
    boxes_count = m_outputDims.d[2];
    channels = m_outputDims.d[1];
    class_count = channels - 4;
    LOG("Output dims: [%d, %d, %d] (需要转置), boxes=%d, classes=%d",
        m_outputDims.d[0], m_outputDims.d[1], m_outputDims.d[2], boxes_count,
        class_count);
  } else {
    // 格式: [1, boxes, 4+classes] -> 正常
    boxes_count = m_outputDims.d[1];
    channels = m_outputDims.d[2];
    class_count = channels - 4;
    LOG("Output dims: [%d, %d, %d], boxes=%d, classes=%d", m_outputDims.d[0],
        m_outputDims.d[1], m_outputDims.d[2], boxes_count, class_count);
  }

  float cx, cy, w, h, obj, prob, conf;
  float x0, y0, x1, y1;
  int label;

  // 调试：查看前几个 box 的原始值和最大置信度
  float max_conf = 0;
  int max_idx = 0;
  for (int i = 0; i < boxes_count; i++) {
    float *t;
    float temp[128];
    if (need_transpose) {
      for (int c = 0; c < channels; c++) {
        temp[c] = m_outputMemory[0][c * boxes_count + i];
      }
      t = temp;
    } else {
      t = m_outputMemory[0] + i * channels;
    }
    for (int c = 0; c < class_count; c++) {
      if (t[4 + c] > max_conf) {
        max_conf = t[4 + c];
        max_idx = i;
      }
    }
  }
  LOG("Debug: max_conf=%.6f at box %d, threshold=%.2f", max_conf, max_idx,
      conf_threshold);

  // 打印最大置信度 box 的详细信息
  {
    float *t;
    float temp[128];
    if (need_transpose) {
      for (int c = 0; c < channels; c++) {
        temp[c] = m_outputMemory[0][c * boxes_count + max_idx];
      }
      t = temp;
    } else {
      t = m_outputMemory[0] + max_idx * channels;
    }
    LOG("Debug: best box: cx=%.2f, cy=%.2f, w=%.2f, h=%.2f, cls=[%.4f, %.4f, "
        "%.4f, %.4f, %.4f, %.4f]",
        t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9]);
  }

  for (int i = 0; i < boxes_count; i++) {
    // 根据是否需要转置来访问数据
    float temp_data[128]; // 临时存储（假设 channels <= 128）
    float *tensor;

    if (need_transpose) {
      // 转置访问：从 [channels, boxes] 读取第 i 个 box
      for (int c = 0; c < channels; c++) {
        temp_data[c] = m_outputMemory[0][c * boxes_count + i];
      }
      tensor = temp_data;
    } else {
      // 正常访问
      tensor = m_outputMemory[0] + i * channels;
    }

    label = max_element(tensor + 4, tensor + 4 + class_count) - (tensor + 4);
    conf = tensor[4 + label];
    if (conf < conf_threshold)
      continue;

    cx = tensor[0];
    cy = tensor[1];
    w = tensor[2];
    h = tensor[3];

    x0 = cx - w / 2;
    y0 = cy - h / 2;
    x1 = x0 + w;
    y1 = y0 + h;

    // LetterBox 坐标变换（与 Python 一致：简单除以 scale）
    x0 = x0 / m_letterbox_scale;
    y0 = y0 / m_letterbox_scale;
    x1 = x1 / m_letterbox_scale;
    y1 = y1 / m_letterbox_scale;

    bbox yolo_box(x0, y0, x1, y1, conf, label);
    m_bboxes.emplace_back(yolo_box);
  }
  LOG("the count of decoded bbox is %d", m_bboxes.size());

  /*Postprocess -- 2. NMS*/
  /*
   * 几个步骤:
   * 1. 做一个IoU计算的lambda函数
   * 2. 将m_bboxes中的所有数据，按照confidence从高到低进行排序
   * 3.
   * 最终希望是对于每一个class，我们都只有一个bbox，所以对同一个class的所有bboxes进行IoU比较，
   *    选取confidence最大。并与其他的同类bboxes的IoU的重叠率最大的同时IoU > IoU
   * threshold
   */

  vector<bbox> final_bboxes;
  final_bboxes.reserve(m_bboxes.size());
  std::sort(m_bboxes.begin(), m_bboxes.end(), [](bbox &box1, bbox &box2) {
    return box1.confidence > box2.confidence;
  });

  /*
   * nms在网上有很多实现方法，其中有一些是根据nms的值来动态改变final_bboex的大小(resize,
   * erease)
   * 这里需要注意的是，频繁的对vector的大小的更改的空间复杂度会比较大，所以尽量不要这么做
   * 可以通过给bbox设置skip计算的flg来调整。
   */
  for (int i = 0; i < m_bboxes.size(); i++) {
    if (m_bboxes[i].flg_remove)
      continue;

    final_bboxes.emplace_back(m_bboxes[i]);
    for (int j = i + 1; j < m_bboxes.size(); j++) {
      if (m_bboxes[j].flg_remove)
        continue;

      if (m_bboxes[i].label == m_bboxes[j].label) {
        if (iou_calc(m_bboxes[i], m_bboxes[j]) > nms_threshold)
          m_bboxes[j].flg_remove = true;
      }
    }
  }
  LOGD("the count of bbox after NMS is %d", final_bboxes.size());

  /*Postprocess -- draw_bbox*/
  /*
   * 几个步骤
   * 1. 通过label获取name
   * 2. 通过label获取color
   * 3. cv::rectangle
   * 4. cv::putText
   */
  string tag = "detect-" + getPrec(m_params->prec);
  m_outputPath = changePath(m_imagePath, "../result", ".png", tag);

  int font_face = 0;
  float font_scale = 0.001 * MIN(m_inputImage.cols, m_inputImage.rows);
  int font_thick = 2;
  int baseline;
  LiyinLabels labels;

  LOG("\tResult:");
  for (int i = 0; i < final_bboxes.size(); i++) {
    auto box = final_bboxes[i];
    auto name = labels.get_label(box.label);
    auto rec_color = labels.get_color(box.label);
    auto txt_color = labels.get_inverse_color(rec_color);
    auto txt = cv::format({"%s: %.2f%%"}, name.c_str(), box.confidence * 100);
    auto txt_size =
        cv::getTextSize(txt, font_face, font_scale, font_thick, &baseline);

    int txt_height = txt_size.height + baseline + 10;
    int txt_width = txt_size.width + 3;

    cv::Point txt_pos(round(box.x0), round(box.y0 - (txt_size.height -
                                                     baseline + font_thick)));
    cv::Rect txt_rec(round(box.x0 - font_thick), round(box.y0 - txt_height),
                     txt_width, txt_height);
    cv::Rect box_rec(round(box.x0), round(box.y0), round(box.x1 - box.x0),
                     round(box.y1 - box.y0));

    cv::rectangle(m_inputImage, box_rec, rec_color, 3);
    cv::rectangle(m_inputImage, txt_rec, rec_color, -1);
    cv::putText(m_inputImage, txt, txt_pos, font_face, font_scale, txt_color,
                font_thick, 16);

    LOG("%+20s detected. Confidence: %.2f%%. Cord: (x0, y0):(%6.2f, %6.2f), "
        "(x1, y1)(%6.2f, %6.2f)",
        name.c_str(), box.confidence * 100, box.x0, box.y0, box.x1, box.y1);
  }
  LOG("\tSummary:");
  LOG("\t\tDetected Objects: %d", final_bboxes.size());
  LOG("");

  m_timer->stop_cpu();
  m_timer->duration_cpu<timer::Timer::ms>("postprocess(CPU)");

  cv::imwrite(m_outputPath, m_inputImage);
  LOG("\tsave image to %s\n", m_outputPath.c_str());

  return true;
}

bool Detector::postprocess_gpu() {
  /*
   * GPU 后处理：
   * 1. GPU Decode：在 GPU 上并行解码 8400 个候选框
   * 2. 拷贝到 CPU：将解码结果从 GPU 传回 CPU
   * 3. CPU NMS：在 CPU 上执行非极大值抑制
   * 4. 绘制结果
   */
  m_timer->start_gpu();

  // Step 1: 检测输出格式
  bool need_transpose = (m_outputDims.d[1] < m_outputDims.d[2]);
  int boxes_count, class_count, channels;
  if (need_transpose) {
    boxes_count = m_outputDims.d[2];
    channels = m_outputDims.d[1];
    class_count = channels - 4;
  } else {
    boxes_count = m_outputDims.d[1];
    channels = m_outputDims.d[2];
    class_count = channels - 4;
  }

  float conf_threshold = 0.25f;
  float nms_threshold = 0.45f;

  // Step 2: GPU Decode - 在 GPU 上并行解码所有候选框
  postprocess::decode_yolo_gpu(m_outputMemory[1], // GPU 上的模型输出
                               m_d_decoded,       // GPU 上的解码结果
                               m_d_count,         // GPU 上的有效框计数
                               boxes_count, channels, class_count,
                               conf_threshold, m_letterbox_scale,
                               need_transpose, m_max_boxes, m_stream);

  // Step 3: 获取有效框数量
  int h_count = 0;
  CUDA_CHECK(cudaMemcpyAsync(&h_count, m_d_count, sizeof(int),
                             cudaMemcpyDeviceToHost, m_stream));
  CUDA_CHECK(cudaStreamSynchronize(m_stream));

  LOG("GPU Decode: %d boxes (conf > %.2f)", h_count, conf_threshold);

  // Step 4: 拷贝解码结果到 CPU
  if (h_count > 0) {
    h_count = min(h_count, m_max_boxes);
    CUDA_CHECK(cudaMemcpyAsync(m_h_decoded, m_d_decoded,
                               h_count * 6 * sizeof(float),
                               cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
  }

  // Step 5: 构建 bbox 列表
  m_bboxes.clear();
  for (int i = 0; i < h_count; i++) {
    float *box = m_h_decoded + i * 6;
    float x0 = box[0];
    float y0 = box[1];
    float x1 = box[2];
    float y1 = box[3];
    float conf = box[4];
    int label = (int)box[5];
    m_bboxes.emplace_back(x0, y0, x1, y1, conf, label);
  }

  m_timer->stop_gpu();
  m_timer->duration_gpu("postprocess GPU decode");

  // Step 6: CPU NMS（NMS 仍在 CPU 执行，因为有数据依赖）
  m_timer->start_cpu();

  vector<bbox> final_bboxes;
  final_bboxes.reserve(m_bboxes.size());
  std::sort(m_bboxes.begin(), m_bboxes.end(), [](bbox &box1, bbox &box2) {
    return box1.confidence > box2.confidence;
  });

  for (int i = 0; i < m_bboxes.size(); i++) {
    if (m_bboxes[i].flg_remove)
      continue;

    final_bboxes.emplace_back(m_bboxes[i]);
    for (int j = i + 1; j < m_bboxes.size(); j++) {
      if (m_bboxes[j].flg_remove)
        continue;

      if (m_bboxes[i].label == m_bboxes[j].label) {
        if (iou_calc(m_bboxes[i], m_bboxes[j]) > nms_threshold)
          m_bboxes[j].flg_remove = true;
      }
    }
  }
  LOGD("NMS: %d -> %d boxes", h_count, final_bboxes.size());

  // Step 7: 绘制结果
  string tag = "detect-" + getPrec(m_params->prec);
  m_outputPath = changePath(m_imagePath, "../result", ".png", tag);

  int font_face = 0;
  float font_scale = 0.001 * MIN(m_inputImage.cols, m_inputImage.rows);
  int font_thick = 2;
  int baseline;
  LiyinLabels labels;

  LOG("\tResult (GPU postprocess):");
  for (int i = 0; i < final_bboxes.size(); i++) {
    auto box = final_bboxes[i];
    auto name = labels.get_label(box.label);
    auto rec_color = labels.get_color(box.label);
    auto txt_color = labels.get_inverse_color(rec_color);
    auto txt = cv::format({"%s: %.2f%%"}, name.c_str(), box.confidence * 100);
    auto txt_size =
        cv::getTextSize(txt, font_face, font_scale, font_thick, &baseline);

    int txt_height = txt_size.height + baseline + 10;
    int txt_width = txt_size.width + 3;

    cv::Point txt_pos(round(box.x0), round(box.y0 - (txt_size.height -
                                                     baseline + font_thick)));
    cv::Rect txt_rec(round(box.x0 - font_thick), round(box.y0 - txt_height),
                     txt_width, txt_height);
    cv::Rect box_rec(round(box.x0), round(box.y0), round(box.x1 - box.x0),
                     round(box.y1 - box.y0));

    cv::rectangle(m_inputImage, box_rec, rec_color, 3);
    cv::rectangle(m_inputImage, txt_rec, rec_color, -1);
    cv::putText(m_inputImage, txt, txt_pos, font_face, font_scale, txt_color,
                font_thick, 16);

    LOG("%+20s detected. Confidence: %.2f%%. Cord: (x0, y0):(%6.2f, %6.2f), "
        "(x1, y1)(%6.2f, %6.2f)",
        name.c_str(), box.confidence * 100, box.x0, box.y0, box.x1, box.y1);
  }
  LOG("\tSummary:");
  LOG("\t\tDetected Objects: %d", final_bboxes.size());
  LOG("");

  m_timer->stop_cpu();
  m_timer->duration_cpu<timer::Timer::ms>("postprocess NMS+draw (CPU)");

  cv::imwrite(m_outputPath, m_inputImage);
  LOG("\tsave image to %s\n", m_outputPath.c_str());

  return true;
}

shared_ptr<Detector> make_detector(std::string onnx_path, logger::Level level,
                                   Params params) {
  return make_shared<Detector>(onnx_path, level, params);
}

}; // namespace detector
}; // namespace model
