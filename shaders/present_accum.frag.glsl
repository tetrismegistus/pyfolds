#version 330
in vec2 v_uv;
layout (location = 0) out vec4 out_color;

uniform sampler2D u_accum;

// artistic controls
uniform float u_density_scale;   // overall darkness
uniform float u_density_gamma ;   // compress dynamic range

const vec3 PAPER = vec3(0.9373, 0.9294, 0.9098);
const vec3 INK   = vec3(0.6549, 0.7529, 0.8588);

void main() {

    // accumulated value from EMA buffer
    float d = texture(u_accum, v_uv).r;

    float density = d * u_density_scale;

    // convert density → number of "ink layers"
    float layers = pow(density, u_density_gamma);

    // MULTIPLY compositing
    vec3 color = PAPER * pow(INK, vec3(layers));

    

    out_color = vec4(color, 1.0);
}
