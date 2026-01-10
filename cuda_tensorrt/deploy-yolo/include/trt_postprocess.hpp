#ifndef __POSTPROCESS_HPP__
#define __POSTPROCESS_HPP__

#include "cuda_runtime_api.h"

// ============================================================================
// GPU 后处理命名空间
// ============================================================================
namespace postprocess {

/**
 * GPU Decode YOLO 输出
 *
 * @param d_output       模型输出 (GPU 内存)
 * @param d_decoded      解码结果 (GPU 内存) [max_boxes, 6]: x0,y0,x1,y1,conf,label
 * @param d_count        有效框计数 (GPU 内存)
 * @param boxes_count    候选框总数 (8400)
 * @param channels       通道数 (4 + num_classes)
 * @param class_count    类别数
 * @param conf_threshold 置信度阈值
 * @param letterbox_scale LetterBox 缩放比例
 * @param need_transpose 是否需要转置
 * @param max_boxes      最大输出框数
 * @param stream         CUDA 流
 */
void decode_yolo_gpu(float *d_output, float *d_decoded, int *d_count,
                     int boxes_count, int channels, int class_count,
                     float conf_threshold, float letterbox_scale,
                     bool need_transpose, int max_boxes, cudaStream_t stream);

} // namespace postprocess

#endif // __POSTPROCESS_HPP__

