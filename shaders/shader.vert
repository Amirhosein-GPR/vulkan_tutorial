#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 project;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pcs;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTextureCoordinate;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTextureCoordinate;

void main() {
    gl_Position = ubo.project * ubo.view * pcs.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTextureCoordinate = inTextureCoordinate;
}
