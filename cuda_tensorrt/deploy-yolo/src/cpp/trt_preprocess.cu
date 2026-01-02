#include "cuda_runtime_api.h"
#include "stdio.h"
#include "trt_preprocess.hpp"
#include <iostream>

namespace preprocess {

TransInfo trans;
AffineMatrix affine_matrix;
LetterBoxInfo letterbox_info; // LetterBox 信息（全局变量）

void warpaffine_init(int srcH, int srcW, int tarH, int tarW) {
  trans.src_h = srcH;
  trans.src_w = srcW;
  trans.tar_h = tarH;
  trans.tar_w = tarW;
  affine_matrix.init(trans);
}

__host__ __device__ void affine_transformation(float trans_matrix[6], int src_x,
                                               int src_y, float *tar_x,
                                               float *tar_y) {
  *tar_x = trans_matrix[0] * src_x + trans_matrix[1] * src_y + trans_matrix[2];
  *tar_y = trans_matrix[3] * src_x + trans_matrix[4] * src_y + trans_matrix[5];
}

__global__ void nearest_BGR2RGB_nhwc2nchw_norm_kernel(
    float *tar, uint8_t *src, int tarW, int tarH, int srcW, int srcH,
    float scaled_w, float scaled_h, float *d_mean, float *d_std) {
  // nearest neighbour -- resized之后的图tar上的坐标
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // nearest neighbour -- 计算最近坐标
  int src_y = floor((float)y * scaled_h);
  int src_x = floor((float)x * scaled_w);

  if (src_x < 0 || src_y < 0 || src_x > srcW || src_y > srcH) {
    // nearest neighbour -- 对于越界的部分，不进行计算
  } else {
    // nearest neighbour -- 计算tar中对应坐标的索引
    int tarIdx = y * tarW + x;
    int tarArea = tarW * tarH;

    // nearest neighbour -- 计算src中最近邻坐标的索引
    int srcIdx = (src_y * srcW + src_x) * 3;

    // nearest neighbour -- 实现nearest beighbour的resize + BGR2RGB + nhwc2nchw
    // + norm
    tar[tarIdx + tarArea * 0] =
        (src[srcIdx + 2] / 255.0f - d_mean[2]) / d_std[2];
    tar[tarIdx + tarArea * 1] =
        (src[srcIdx + 1] / 255.0f - d_mean[1]) / d_std[1];
    tar[tarIdx + tarArea * 2] =
        (src[srcIdx + 0] / 255.0f - d_mean[0]) / d_std[0];
  }
}

__global__ void bilinear_BGR2RGB_nhwc2nchw_norm_kernel(
    float *tar, uint8_t *src, int tarW, int tarH, int srcW, int srcH,
    float scaled_w, float scaled_h, float *d_mean, float *d_std) {

  // bilinear interpolation -- resized之后的图tar上的坐标
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
  int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
  int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
  int src_y2 = src_y1 + 1;
  int src_x2 = src_x1 + 1;

  if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
    // bilinear interpolation -- 对于越界的坐标不进行计算
  } else {
    // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
    float th = ((y + 0.5) * scaled_h - 0.5) - src_y1;
    float tw = ((x + 0.5) * scaled_w - 0.5) - src_x1;

    // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
    float a1_1 = (1.0 - tw) * (1.0 - th); //右下
    float a1_2 = tw * (1.0 - th);         //左下
    float a2_1 = (1.0 - tw) * th;         //右上
    float a2_2 = tw * th;                 //左上

    // bilinear interpolation -- 计算4个坐标所对应的索引
    int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; //左上
    int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; //右上
    int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; //左下
    int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; //右下

    // bilinear interpolation -- 计算resized之后的图的索引
    int tarIdx = y * tarW + x;
    int tarArea = tarW * tarH;

    // bilinear interpolation -- 实现bilinear interpolation的resize + BGR2RGB +
    // NHWC2NCHW normalization 注意，这里tar和src进行遍历的方式是不一样的
    tar[tarIdx + tarArea * 0] =
        (round((a1_1 * src[srcIdx1_1 + 2] + a1_2 * src[srcIdx1_2 + 2] +
                a2_1 * src[srcIdx2_1 + 2] + a2_2 * src[srcIdx2_2 + 2])) /
             255.0f -
         d_mean[2]) /
        d_std[2];

    tar[tarIdx + tarArea * 1] =
        (round((a1_1 * src[srcIdx1_1 + 1] + a1_2 * src[srcIdx1_2 + 1] +
                a2_1 * src[srcIdx2_1 + 1] + a2_2 * src[srcIdx2_2 + 1])) /
             255.0f -
         d_mean[1]) /
        d_std[1];

    tar[tarIdx + tarArea * 2] =
        (round((a1_1 * src[srcIdx1_1 + 0] + a1_2 * src[srcIdx1_2 + 0] +
                a2_1 * src[srcIdx2_1 + 0] + a2_2 * src[srcIdx2_2 + 0])) /
             255.0f -
         d_mean[0]) /
        d_std[0];
  }
}

__global__ void bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel(
    float *tar, uint8_t *src, int tarW, int tarH, int srcW, int srcH,
    float scaled_w, float scaled_h, float *d_mean, float *d_std) {
  // resized之后的图tar上的坐标
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
  int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
  int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
  int src_y2 = src_y1 + 1;
  int src_x2 = src_x1 + 1;

  if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
    // bilinear interpolation -- 对于越界的坐标不进行计算
  } else {
    // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
    float th = (float)y * scaled_h - src_y1;
    float tw = (float)x * scaled_w - src_x1;

    // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
    float a1_1 = (1.0 - tw) * (1.0 - th); // 右下
    float a1_2 = tw * (1.0 - th);         // 左下
    float a2_1 = (1.0 - tw) * th;         // 右上
    float a2_2 = tw * th;                 // 左上

    // bilinear interpolation -- 计算4个坐标所对应的索引
    int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; // 左上
    int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; // 右上
    int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; // 左下
    int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; // 右下

    // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
    y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
    x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);

    // bilinear interpolation -- 计算resized之后的图的索引
    int tarIdx = (y * tarW + x) * 3;
    int tarArea = tarW * tarH;

    // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB + shift +
    // nhwc2nchw
    tar[tarIdx + tarArea * 0] =
        (round((a1_1 * src[srcIdx1_1 + 2] + a1_2 * src[srcIdx1_2 + 2] +
                a2_1 * src[srcIdx2_1 + 2] + a2_2 * src[srcIdx2_2 + 2])) /
             255.0f -
         d_mean[2]) /
        d_std[2];

    tar[tarIdx + tarArea * 1] =
        (round((a1_1 * src[srcIdx1_1 + 1] + a1_2 * src[srcIdx1_2 + 1] +
                a2_1 * src[srcIdx2_1 + 1] + a2_2 * src[srcIdx2_2 + 1])) /
             255.0f -
         d_mean[1]) /
        d_std[1];

    tar[tarIdx + tarArea * 2] =
        (round((a1_1 * src[srcIdx1_1 + 0] + a1_2 * src[srcIdx1_2 + 0] +
                a2_1 * src[srcIdx2_1 + 0] + a2_2 * src[srcIdx2_2 + 0])) /
             255.0f -
         d_mean[0]) /
        d_std[0];
  }
}

__global__ void nearest_BGR2RGB_nhwc2nchw_kernel(float *tar, uint8_t *src,
                                                 int tarW, int tarH, int srcW,
                                                 int srcH, float scaled_w,
                                                 float scaled_h) {
  // nearest neighbour -- resized之后的图tar上的坐标
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // nearest neighbour -- 计算最近坐标
  int src_y = floor((float)y * scaled_h);
  int src_x = floor((float)x * scaled_w);

  if (src_x < 0 || src_y < 0 || src_x > srcW || src_y > srcH) {
    // nearest neighbour -- 对于越界的部分，不进行计算
  } else {
    // nearest neighbour -- 计算tar中对应坐标的索引
    int tarIdx = y * tarW + x;
    int tarArea = tarW * tarH;

    // nearest neighbour -- 计算src中最近邻坐标的索引
    int srcIdx = (src_y * srcW + src_x) * 3;

    // nearest neighbour -- 实现nearest beighbour的resize + BGR2RGB + nhwc2nchw
    // + norm
    tar[tarIdx + tarArea * 0] = src[srcIdx + 2] / 255.0f;
    tar[tarIdx + tarArea * 1] = src[srcIdx + 1] / 255.0f;
    tar[tarIdx + tarArea * 2] = src[srcIdx + 0] / 255.0f;
  }
}

__global__ void bilinear_BGR2RGB_nhwc2nchw_kernel(float *tar, uint8_t *src,
                                                  int tarW, int tarH, int srcW,
                                                  int srcH, float scaled_w,
                                                  float scaled_h) {

  // bilinear interpolation -- resized之后的图tar上的坐标
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
  int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
  int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
  int src_y2 = src_y1 + 1;
  int src_x2 = src_x1 + 1;

  if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
    // bilinear interpolation -- 对于越界的坐标不进行计算
  } else {
    // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
    float th = ((y + 0.5) * scaled_h - 0.5) - src_y1;
    float tw = ((x + 0.5) * scaled_w - 0.5) - src_x1;

    // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
    float a1_1 = (1.0 - tw) * (1.0 - th); //右下
    float a1_2 = tw * (1.0 - th);         //左下
    float a2_1 = (1.0 - tw) * th;         //右上
    float a2_2 = tw * th;                 //左上

    // bilinear interpolation -- 计算4个坐标所对应的索引
    int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; //左上
    int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; //右上
    int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; //左下
    int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; //右下

    // bilinear interpolation -- 计算resized之后的图的索引
    int tarIdx = y * tarW + x;
    int tarArea = tarW * tarH;

    // bilinear interpolation -- 实现bilinear interpolation的resize + BGR2RGB +
    // NHWC2NCHW normalization 注意，这里tar和src进行遍历的方式是不一样的
    tar[tarIdx + tarArea * 0] =
        round((a1_1 * src[srcIdx1_1 + 2] + a1_2 * src[srcIdx1_2 + 2] +
               a2_1 * src[srcIdx2_1 + 2] + a2_2 * src[srcIdx2_2 + 2])) /
        255.0f;

    tar[tarIdx + tarArea * 1] =
        round((a1_1 * src[srcIdx1_1 + 1] + a1_2 * src[srcIdx1_2 + 1] +
               a2_1 * src[srcIdx2_1 + 1] + a2_2 * src[srcIdx2_2 + 1])) /
        255.0f;

    tar[tarIdx + tarArea * 2] =
        round((a1_1 * src[srcIdx1_1 + 0] + a1_2 * src[srcIdx1_2 + 0] +
               a2_1 * src[srcIdx2_1 + 0] + a2_2 * src[srcIdx2_2 + 0])) /
        255.0f;
  }
}

__global__ void
bilinear_BGR2RGB_nhwc2nchw_shift_kernel(float *tar, uint8_t *src, int tarW,
                                        int tarH, int srcW, int srcH,
                                        float scaled_w, float scaled_h) {
  // resized之后的图tar上的坐标
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
  int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
  int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
  int src_y2 = src_y1 + 1;
  int src_x2 = src_x1 + 1;

  if (src_y1 < 0 || src_x1 < 0 || src_y2 > srcH || src_x2 > srcW) {
    // bilinear interpolation -- 对于越界的坐标不进行计算
  } else {
    // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
    float th = (float)y * scaled_h - src_y1;
    float tw = (float)x * scaled_w - src_x1;

    // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
    float a1_1 = (1.0 - tw) * (1.0 - th); // 右下
    float a1_2 = tw * (1.0 - th);         // 左下
    float a2_1 = (1.0 - tw) * th;         // 右上
    float a2_2 = tw * th;                 // 左上

    // bilinear interpolation -- 计算4个坐标所对应的索引
    int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; // 左上
    int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; // 右上
    int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; // 左下
    int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; // 右下

    // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
    y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
    x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);

    // bilinear interpolation -- 计算resized之后的图的索引
    int tarIdx = y * tarW + x;
    int tarArea = tarW * tarH;

    // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB + shift +
    // nhwc2nchw
    tar[tarIdx + tarArea * 0] =
        round((a1_1 * src[srcIdx1_1 + 2] + a1_2 * src[srcIdx1_2 + 2] +
               a2_1 * src[srcIdx2_1 + 2] + a2_2 * src[srcIdx2_2 + 2])) /
        255.0f;

    tar[tarIdx + tarArea * 1] =
        round((a1_1 * src[srcIdx1_1 + 1] + a1_2 * src[srcIdx1_2 + 1] +
               a2_1 * src[srcIdx2_1 + 1] + a2_2 * src[srcIdx2_2 + 1])) /
        255.0f;

    tar[tarIdx + tarArea * 2] =
        round((a1_1 * src[srcIdx1_1 + 0] + a1_2 * src[srcIdx1_2 + 0] +
               a2_1 * src[srcIdx2_1 + 0] + a2_2 * src[srcIdx2_2 + 0])) /
        255.0f;
  }
}

__global__ void warpaffine_BGR2RGB_kernel(float *tar, uint8_t *src,
                                          TransInfo trans,
                                          AffineMatrix affine_matrix) {
  float src_x, src_y;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // 边界检查
  if (x >= trans.tar_w || y >= trans.tar_h)
    return;

  affine_transformation(affine_matrix.reverse, x + 0.5, y + 0.5, &src_x,
                        &src_y);

  int src_x1 = floor(src_x - 0.5);
  int src_y1 = floor(src_y - 0.5);
  int src_x2 = src_x1 + 1;
  int src_y2 = src_y1 + 1;

  int tarIdx = y * trans.tar_w + x;
  int tarArea = trans.tar_w * trans.tar_h;

  // 越界区域填充 114/255 ≈ 0.447（与 Python LetterBox 一致）
  if (src_y1 < 0 || src_x1 < 0 || src_y2 > trans.src_h ||
      src_x2 > trans.src_w) {
    float fill_value = 114.0f / 255.0f; // LetterBox 标准填充值
    tar[tarIdx + tarArea * 0] = fill_value;
    tar[tarIdx + tarArea * 1] = fill_value;
    tar[tarIdx + tarArea * 2] = fill_value;
  } else {
    float tw = src_x - src_x1;
    float th = src_y - src_y1;

    float a1_1 = (1.0 - tw) * (1.0 - th);
    float a1_2 = tw * (1.0 - th);
    float a2_1 = (1.0 - tw) * th;
    float a2_2 = tw * th;

    int srcIdx1_1 = (src_y1 * trans.src_w + src_x1) * 3;
    int srcIdx1_2 = (src_y1 * trans.src_w + src_x2) * 3;
    int srcIdx2_1 = (src_y2 * trans.src_w + src_x1) * 3;
    int srcIdx2_2 = (src_y2 * trans.src_w + src_x2) * 3;

    // tarIdx 和 tarArea 已在上面定义

    tar[tarIdx + tarArea * 0] =
        round((a1_1 * src[srcIdx1_1 + 2] + a1_2 * src[srcIdx1_2 + 2] +
               a2_1 * src[srcIdx2_1 + 2] + a2_2 * src[srcIdx2_2 + 2])) /
        255.0f;

    tar[tarIdx + tarArea * 1] =
        round((a1_1 * src[srcIdx1_1 + 1] + a1_2 * src[srcIdx1_2 + 1] +
               a2_1 * src[srcIdx2_1 + 1] + a2_2 * src[srcIdx2_2 + 1])) /
        255.0f;

    tar[tarIdx + tarArea * 2] =
        round((a1_1 * src[srcIdx1_1 + 0] + a1_2 * src[srcIdx1_2 + 0] +
               a2_1 * src[srcIdx2_1 + 0] + a2_2 * src[srcIdx2_2 + 0])) /
        255.0f;
  }
}

void resize_bilinear_gpu(float *d_tar, uint8_t *d_src, int tarW, int tarH,
                         int srcW, int srcH, float *d_mean, float *d_std,
                         tactics tac) {
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(tarW / 32 + 1, tarH / 32 + 1, 1);

  // scaled resize
  float scaled_h = (float)srcH / tarH;
  float scaled_w = (float)srcW / tarW;
  float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

  switch (tac) {
  case tactics::GPU_NEAREST:
    nearest_BGR2RGB_nhwc2nchw_norm_kernel<<<dimGrid, dimBlock>>>(
        d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h, d_mean,
        d_std);
    break;
  case tactics::GPU_NEAREST_CENTER:
    nearest_BGR2RGB_nhwc2nchw_norm_kernel<<<dimGrid, dimBlock>>>(
        d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale, d_mean, d_std);
    break;
  case tactics::GPU_BILINEAR:
    bilinear_BGR2RGB_nhwc2nchw_norm_kernel<<<dimGrid, dimBlock>>>(
        d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h, d_mean,
        d_std);
    break;
  case tactics::GPU_BILINEAR_CENTER:
    bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel<<<dimGrid, dimBlock>>>(
        d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale, d_mean, d_std);
    break;
  default:
    LOGE("ERROR: Wrong GPU resize tactics selected. Program terminated");
    exit(1);
  }
}

void resize_bilinear_gpu(float *d_tar, uint8_t *d_src, int tarW, int tarH,
                         int srcW, int srcH, tactics tac) {
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(tarW / 32 + 1, tarH / 32 + 1, 1);

  // scaled resize
  float scaled_h = (float)srcH / tarH;
  float scaled_w = (float)srcW / tarW;
  float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

  switch (tac) {
  case tactics::GPU_NEAREST:
    nearest_BGR2RGB_nhwc2nchw_kernel<<<dimGrid, dimBlock>>>(
        d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
    break;
  case tactics::GPU_NEAREST_CENTER:
    nearest_BGR2RGB_nhwc2nchw_kernel<<<dimGrid, dimBlock>>>(
        d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale);
    break;
  case tactics::GPU_BILINEAR:
    bilinear_BGR2RGB_nhwc2nchw_kernel<<<dimGrid, dimBlock>>>(
        d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
    break;
  case tactics::GPU_BILINEAR_CENTER:
    bilinear_BGR2RGB_nhwc2nchw_shift_kernel<<<dimGrid, dimBlock>>>(
        d_tar, d_src, tarW, tarH, srcW, srcH, scale, scale);
    break;
  case tactics::GPU_WARP_AFFINE:
    warpaffine_init(srcH, srcW, tarH, tarW);
    warpaffine_BGR2RGB_kernel<<<dimGrid, dimBlock>>>(d_tar, d_src, trans,
                                                     affine_matrix);
    break;
  default:
    LOGE("ERROR: Wrong GPU resize tactics selected. Program terminated");
    exit(1);
  }
}

/*
 * LetterBox GPU Kernel
 * 与 Python 版本完全一致：
 * 1. 计算 scale = min(tarH/srcH, tarW/srcW)
 * 2. resize 图像到 (new_w, new_h) = (srcW * scale, srcH * scale)
 * 3. 填充到 tarH x tarW，左上角对齐，填充值为 114/255.0
 * 4. BGR -> RGB, HWC -> CHW, normalize (/255.0)
 */
__global__ void letterbox_BGR2RGB_kernel(float *tar, uint8_t *src, int tarW,
                                         int tarH, int srcW, int srcH, int newW,
                                         int newH, float scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // 边界检查
  if (x >= tarW || y >= tarH)
    return;

  int tarIdx = y * tarW + x;
  int tarArea = tarW * tarH;

  // 如果在填充区域（超出 resize 后的图像范围），填充 114/255
  if (x >= newW || y >= newH) {
    float fill_value = 114.0f / 255.0f;
    tar[tarIdx + tarArea * 0] = fill_value;
    tar[tarIdx + tarArea * 1] = fill_value;
    tar[tarIdx + tarArea * 2] = fill_value;
    return;
  }

  // 计算在原图中的对应位置（使用双线性插值）
  float src_x_f = (x + 0.5f) / scale - 0.5f;
  float src_y_f = (y + 0.5f) / scale - 0.5f;

  int src_x1 = static_cast<int>(floorf(src_x_f));
  int src_y1 = static_cast<int>(floorf(src_y_f));
  int src_x2 = src_x1 + 1;
  int src_y2 = src_y1 + 1;

  // 边界保护
  src_x1 = max(0, min(src_x1, srcW - 1));
  src_y1 = max(0, min(src_y1, srcH - 1));
  src_x2 = max(0, min(src_x2, srcW - 1));
  src_y2 = max(0, min(src_y2, srcH - 1));

  float tw = src_x_f - floorf(src_x_f);
  float th = src_y_f - floorf(src_y_f);

  // 双线性插值权重
  float a1_1 = (1.0f - tw) * (1.0f - th); // 左上
  float a1_2 = tw * (1.0f - th);          // 右上
  float a2_1 = (1.0f - tw) * th;          // 左下
  float a2_2 = tw * th;                   // 右下

  // 计算原图中4个点的索引
  int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;
  int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;
  int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;
  int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;

  // 双线性插值 + BGR2RGB + normalize
  tar[tarIdx + tarArea * 0] =
      (a1_1 * src[srcIdx1_1 + 2] + a1_2 * src[srcIdx1_2 + 2] +
       a2_1 * src[srcIdx2_1 + 2] + a2_2 * src[srcIdx2_2 + 2]) /
      255.0f; // R

  tar[tarIdx + tarArea * 1] =
      (a1_1 * src[srcIdx1_1 + 1] + a1_2 * src[srcIdx1_2 + 1] +
       a2_1 * src[srcIdx2_1 + 1] + a2_2 * src[srcIdx2_2 + 1]) /
      255.0f; // G

  tar[tarIdx + tarArea * 2] =
      (a1_1 * src[srcIdx1_1 + 0] + a1_2 * src[srcIdx1_2 + 0] +
       a2_1 * src[srcIdx2_1 + 0] + a2_2 * src[srcIdx2_2 + 0]) /
      255.0f; // B
}

/*
 * LetterBox GPU 预处理函数
 * 与 Python 版本完全一致的实现
 */
void letterbox_resize_gpu(cv::Mat &h_src, float *d_tar, const int &tarH,
                          const int &tarW, float &out_scale) {
  int srcH = h_src.rows;
  int srcW = h_src.cols;

  // Step 1: 计算 scale（与 Python 一致）
  float scale = std::min((float)tarH / srcH, (float)tarW / srcW);
  int newH = static_cast<int>(srcH * scale);
  int newW = static_cast<int>(srcW * scale);

  // 保存 scale 供后处理使用
  out_scale = scale;
  letterbox_info.scale = scale;
  letterbox_info.new_w = newW;
  letterbox_info.new_h = newH;
  letterbox_info.pad_w = 0; // 左上角对齐，无偏移
  letterbox_info.pad_h = 0;

  // Step 2: 分配 GPU 内存并上传原图
  uint8_t *d_src = nullptr;
  size_t srcSize = srcH * srcW * 3 * sizeof(uint8_t);
  CUDA_CHECK(cudaMalloc(&d_src, srcSize));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data, srcSize, cudaMemcpyHostToDevice));

  // Step 3: 调用 GPU kernel
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((tarW + 31) / 32, (tarH + 31) / 32, 1);

  letterbox_BGR2RGB_kernel<<<dimGrid, dimBlock>>>(
      d_tar, d_src, tarW, tarH, srcW, srcH, newW, newH, scale);

  CUDA_CHECK(cudaDeviceSynchronize());

  // Step 4: 释放临时内存
  CUDA_CHECK(cudaFree(d_src));
}

} // namespace preprocess

// ============================================================================
// GPU 后处理（Decode YOLO 输出）
// ============================================================================
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
