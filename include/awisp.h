#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AwispInstance AwispInstance;

typedef enum AwispParamType {
    AWISP_PARAM_F32 = 0,
    AWISP_PARAM_I32 = 1,
    AWISP_PARAM_U32 = 2,
    AWISP_PARAM_BOOL = 3,
    AWISP_PARAM_VEC2 = 4,
    AWISP_PARAM_VEC3 = 5,
    AWISP_PARAM_VEC4 = 6,
} AwispParamType;

typedef struct AwispParamDesc {
    const char* id;
    const char* label;
    AwispParamType param_type;
    uint32_t component_count;
    float min_value;
    float max_value;
    float step;
    float default_values[4];
    bool is_color;
    size_t option_count;
} AwispParamDesc;

size_t awisp_shader_count(void);
const char* awisp_shader_name(size_t index);
const char* awisp_shader_title(const char* shader_name);
size_t awisp_param_count(const char* asset_root, const char* shader_name);
bool awisp_param_desc(const char* asset_root, const char* shader_name, size_t index, AwispParamDesc* out_desc);
const char* awisp_param_option_label(const char* asset_root, const char* shader_name, size_t param_index, size_t option_index);
int32_t awisp_param_option_value(const char* asset_root, const char* shader_name, size_t param_index, size_t option_index);

AwispInstance* awisp_instance_open_embedded(
    const char* asset_root,
    const char* shader_name,
    const char* window_title,
    int32_t window_x,
    int32_t window_y,
    uint32_t window_width,
    uint32_t window_height);

bool awisp_instance_load_shader(AwispInstance* instance, const char* shader_name);
bool awisp_instance_set_visible(AwispInstance* instance, bool visible);
bool awisp_instance_set_title_bar_visible(AwispInstance* instance, bool visible);
bool awisp_instance_set_geometry(
    AwispInstance* instance,
    int32_t window_x,
    int32_t window_y,
    uint32_t window_width,
    uint32_t window_height);
bool awisp_instance_push_audio(
    AwispInstance* instance,
    const float* interleaved,
    size_t frames,
    size_t channels);
bool awisp_instance_load_image(AwispInstance* instance, const char* path);
bool awisp_instance_listen_udp(AwispInstance* instance, uint16_t port);
bool awisp_instance_set_param_f32(AwispInstance* instance, const char* id, const float* values, size_t value_count);
bool awisp_instance_set_param_bool(AwispInstance* instance, const char* id, bool value);
bool awisp_instance_set_param_i32(AwispInstance* instance, const char* id, int32_t value);
bool awisp_instance_set_param_u32(AwispInstance* instance, const char* id, uint32_t value);
void awisp_instance_free(AwispInstance* instance);
int32_t awisp_instance_status(const AwispInstance* instance);
const char* awisp_instance_error(const AwispInstance* instance);
const char* awisp_last_error(void);

#ifdef __cplusplus
}
#endif
