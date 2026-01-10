#include "cuda_runtime_api.h"
#include "trt_postprocess.hpp"
#include <cstdio>
#include <cstdlib>

// CUDA 错误检查宏
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

namespace postprocess {

/**
 * YOLO Decode Kernel
 *
 * 每个线程处理一个候选框，并行解码所有框
 * 使用原子操作写入结果
 */
__global__ void decode_yolo_kernel(
    float *d_output, // 模型输出 [1, channels, boxes] 或 [1, boxes, channels]
    float *d_decoded, // 解码结果 [max_boxes, 6]: x0, y0, x1, y1, conf, label
    int *d_count,     // 有效 box 计数
    int boxes_count,  // 总候选框数量
    int channels,     // 通道数
    int class_count,  // 类别数
    float conf_threshold,  // 置信度阈值
    float letterbox_scale, // LetterBox 缩放比例
    bool need_transpose,   // 是否需要转置
    int max_boxes)         // 最大输出数量
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= boxes_count)
    return;

  // 获取当前 box 的数据
  float cx, cy, w, h;
  float max_class_score = -1.0f;
  int max_class_idx = 0;

  if (need_transpose) {
    // 格式: [channels, boxes] - 转置访问
    cx = d_output[0 * boxes_count + idx];
    cy = d_output[1 * boxes_count + idx];
    w = d_output[2 * boxes_count + idx];
    h = d_output[3 * boxes_count + idx];

    // 找最大类别分数
    for (int c = 0; c < class_count; c++) {
      float score = d_output[(4 + c) * boxes_count + idx];
      if (score > max_class_score) {
        max_class_score = score;
        max_class_idx = c;
      }
    }
  } else {
    // 格式: [boxes, channels] - 正常访问
    float *tensor = d_output + idx * channels;
    cx = tensor[0];
    cy = tensor[1];
    w = tensor[2];
    h = tensor[3];

    // 找最大类别分数
    for (int c = 0; c < class_count; c++) {
      float score = tensor[4 + c];
      if (score > max_class_score) {
        max_class_score = score;
        max_class_idx = c;
      }
    }
  }

  // 置信度过滤
  if (max_class_score < conf_threshold)
    return;

  // 坐标转换: center -> corner
  float x0 = cx - w / 2;
  float y0 = cy - h / 2;
  float x1 = cx + w / 2;
  float y1 = cy + h / 2;

  // LetterBox 逆变换
  x0 = x0 / letterbox_scale;
  y0 = y0 / letterbox_scale;
  x1 = x1 / letterbox_scale;
  y1 = y1 / letterbox_scale;

  // 原子操作获取写入位置
  int write_idx = atomicAdd(d_count, 1);
  if (write_idx >= max_boxes)
    return;

  // 写入结果 [x0, y0, x1, y1, conf, label]
  d_decoded[write_idx * 6 + 0] = x0;
  d_decoded[write_idx * 6 + 1] = y0;
  d_decoded[write_idx * 6 + 2] = x1;
  d_decoded[write_idx * 6 + 3] = y1;
  d_decoded[write_idx * 6 + 4] = max_class_score;
  d_decoded[write_idx * 6 + 5] = (float)max_class_idx;
}

void decode_yolo_gpu(float *d_output, float *d_decoded, int *d_count,
                     int boxes_count, int channels, int class_count,
                     float conf_threshold, float letterbox_scale,
                     bool need_transpose, int max_boxes, cudaStream_t stream) {
  // 重置计数器
  CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

  // 启动 kernel
  int threads_per_block = 256;
  int blocks = (boxes_count + threads_per_block - 1) / threads_per_block;

  decode_yolo_kernel<<<blocks, threads_per_block, 0, stream>>>(
      d_output, d_decoded, d_count, boxes_count, channels, class_count,
      conf_threshold, letterbox_scale, need_transpose, max_boxes);
}

} // namespace postprocess
