#ifdef __cplusplus
extern "C" {
#endif

int ROIPoolForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, int* argmax_data);


//int ROIPoolBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
//    const int height, const int width, const int channels, const int pooled_height,
//    const int pooled_width, const float* bottom_rois,
//    float* bottom_diff, const int* argmax_data);

#ifdef __cplusplus
}
#endif

