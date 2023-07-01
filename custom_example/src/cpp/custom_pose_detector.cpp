#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "mmdeploy/pose_detector.h"
#include "mmdeploy/pipeline.h"
#include "mmdeploy/model.h"

int main(int argc,char *argv[]) {
  if (argc != 5) {
    fprintf(stderr, "usage:\n  pose_detection device_name model_path image_path test_batch\n");
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto image_path = argv[3];
  int test_batch ;
  sscanf_s(argv[4],"%d",&test_batch);
  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    fprintf(stderr, "failed to load image: %s\n", image_path);
    return 1;
  }

  mmdeploy_profiler_t profiler{};
  mmdeploy_profiler_create("profiler_data.txt", &profiler);

  mmdeploy_model_t model{};
  mmdeploy_model_create_by_path(model_path, &model);

  
  mmdeploy_context_t context{};
  mmdeploy_context_create_by_device(device_name, 0, &context);
  mmdeploy_context_add(context, MMDEPLOY_TYPE_PROFILER, nullptr, profiler);

  // mmdeploy_pipeline_t pipeline{};
  // mmdeploy_pipeline_create_from_model(model, context, &pipeline);

  mmdeploy_pose_detector_t pose_detector{};
  int status{};
  status = mmdeploy_pose_detector_create_v2(model, context, &pose_detector);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create pose_estimator, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};


  // mmdeploy_pose_detection_t *res{};
  // status = mmdeploy_pose_detector_apply(pose_detector, &mat, 1, &res);
  // if (status != MMDEPLOY_SUCCESS) {
  //   fprintf(stderr, "failed to apply pose estimator, code: %d\n", (int)status);
  //   return 1;
  // }
  // for (int i = 0; i < res->length; i++) {
  //   cv::circle(img, {(int)res->point[i].x, (int)res->point[i].y}, 1, {0, 255, 0}, 2);
  // }
  // cv::imwrite("output_pose.png", img);
  // mmdeploy_pose_detector_release_result(res, 1);


  fprintf(stderr, "Start inference benchmark\n");


  for (int i = 0; i < test_batch; i++) {
    mmdeploy_pose_detection_t *res{};
    status = mmdeploy_pose_detector_apply(pose_detector, &mat, 1, &res);
    mmdeploy_pose_detector_release_result(res, 1);
    if ((i+1)%100==0){
      fprintf(stderr, "[%d/%d]\n",i+1,test_batch);
    }
  }
  // if (status != MMDEPLOY_SUCCESS) {
  //   fprintf(stderr, "failed to apply pose estimator, code: %d\n", (int)status);
  //   return 1;
  // }

  // for (int i = 0; i < res->length; i++) {
  //   cv::circle(img, {(int)res->point[i].x, (int)res->point[i].y}, 1, {0, 255, 0}, 2);
  // }
  // cv::imwrite("output_pose.png", img);

  mmdeploy_pose_detector_destroy(pose_detector);
  mmdeploy_model_destroy(model);
  mmdeploy_profiler_destroy(profiler);
  // mmdeploy_pipeline_destroy(pipeline);
  mmdeploy_context_destroy(context);

  return 0;
}
