// shaders/present.frag.glsl
#version 330

in vec2 v_uv;
layout (location = 0) out vec4 out_color;

// NOTE: r32ui textures are sampled as unsigned integer samplers
uniform usampler2D u_state;

// Cheap tonemap
vec3 tonemap(vec3 c) { return vec3(1.0) - exp(-c); }

void main() {
    // fetch uint count from R32UI
    uint count = texture(u_state, v_uv).r;

    // map to float intensity (tune these constants)
    float x = float(count) * 0.02;

    // simple “glowy” palette
    vec3 col = tonemap(vec3(x) * vec3(0.9, 0.6, 1.2));
    col = pow(clamp(col, 0.0, 1.0), vec3(0.6));

    out_color = vec4(col, 1.0);
}
