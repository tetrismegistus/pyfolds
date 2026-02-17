// shaders/resolve.compute.glsl
#version 430
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

uniform vec2  u_resolution;
uniform float u_alpha;     // EMA alpha
uniform float u_exposure;  // scales hits -> brightness

layout(r32ui,   binding = 0) uniform readonly  uimage2D hitsImg;
layout(rgba16f, binding = 1) uniform coherent image2D accumImg;

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= int(u_resolution.x) || gid.y >= int(u_resolution.y)) return;

    uint c = imageLoad(hitsImg, gid).r;

    // Convert hit count to intensity
    vec3 contrib = vec3(float(c)) * u_exposure;

    vec3 prev = imageLoad(accumImg, gid).rgb;

    // Exponential moving average
    vec3 accum = mix(prev, contrib, u_alpha);

    imageStore(accumImg, gid, vec4(accum, 1.0));
}
