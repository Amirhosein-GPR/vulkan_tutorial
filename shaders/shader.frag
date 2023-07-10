#version 450

layout(binding = 1) uniform sampler2D textureSampler;

layout(push_constant) uniform PushConstants {
    layout(offset = 64) float opacity;
} pcs;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTextureCoordinate;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(texture(textureSampler, fragTextureCoordinate).rgb, pcs.opacity);
}
