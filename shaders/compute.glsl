// shaders/compute.glsl
#version 430

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

#define hash_f_s(s)  ( float( hashi(uint(s)) ) / float( 0xffffffffU ) )
#define hash_f()     ( float( seed = hashi(seed) ) / float( 0xffffffffU ) )
#define hash_v2()    vec2(hash_f(),hash_f())
#define rot(a)       mat2(cos(a),-sin(a),sin(a),cos(a))

#define PI 3.1415926535897932384626433832795

uniform float fGlobalTime; // seconds
uniform vec2  v2Resolution; // pixels (w,h)

layout(r32ui, binding = 0) uniform coherent uimage2D computeTex; // "front"

uint seed;

uint hashi(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// ---- uniforms to match Processing mapping ----
uniform float u_x1;
uniform float u_x2;
uniform float u_y1;
uniform float u_y2;

// Processing: float margin = outputWidth * .95; (odd; see note)
// We'll interpret as a fraction of half-extent unless you confirm.
uniform float u_margin_frac = 0.05; // 5% inset (typical). Set to match your intent.

// ---- RNG helpers (you already have hash_f / seed / hashi) ----
float rand01() { return hash_f(); }

// Box-Muller: ~Gaussian(0,1). Good enough for jitter.
float randn() {
    float u1 = max(rand01(), 1e-7);
    float u2 = rand01();
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}
vec2 randn2() { return vec2(randn(), randn()); }

// ---- wrap: repeat into [min,max) ----
float wrapRepeat1(float x, float a, float b) {
    float w = b - a;
    return a + mod(x - a, w);
}
vec2 wrapRepeat(vec2 v, float x1, float x2, float y1, float y2) {
    return vec2(
        wrapRepeat1(v.x, x1, x2),
        wrapRepeat1(v.y, y1, y2)
    );
}

// ---- map like Processing's map() ----
float map01(float x, float a, float b) {
    return (x - a) / (b - a);
}

// domain->pixel with margin
vec2 domainToPixel(vec2 v, vec2 resolution) {
    float w = resolution.x;
    cloat h = resolution.y;
// shaders/resolve.compute.glsl — per-pixel resolve: hits (uint) -> accum (float) with EMA
#version 430
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

uniform vec2  u_resolution;
uniform float u_alpha;     // EMA alpha
uniform float u_exposure;  // scales hits -> brightness

layout(r32ui,  binding = 0) uniform readonly  uimage2D hitsImg;
layout(rgba16f, binding = 1) uniform coherent image2D accumImg;

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= int(u_resolution.x) || gid.y >= int(u_resolution.y)) return;

    uint c = imageLoad(hitsImg, gid).r;

    // Convert hit count to a sample intensity
    // (You’ll tune exposure and possibly apply log/sqrt here)
    vec3 sample = vec3(float(c)) * u_exposure;

    vec3 prev = imageLoad(accumImg, gid).rgb;

    // Exponential moving average (temporal accumulation)
    vec3 next = mix(prev, sample, u_alpha);

    imageStore(accumImg, gid, vec4(next, 1.0));
}

    // interpret margin as inset from each side
    float mx = u_margin_frac * w;
    float my = u_margin_frac * h;

    float tx = clamp(map01(v.x, u_x1, u_x2), 0.0, 1.0);
    float ty = clamp(map01(v.y, u_y1, u_y2), 0.0, 1.0);

    float px = mix(mx, w - mx, tx);
    float py = mix(my, h - my, ty);
    return vec2(px, py);
}

float getTheta(vec2 v) { return atan(v.x, v.y); }
float getR(vec2 p)     { return length(p); }
float rcp_safe(float x) { return 1.0 / max(abs(x), 1e-10); }

vec2 project_particle(vec3 p){
    p.xy /= p.z;
    return p.xy;
}


// ---- tunable parameters (set from host, or leave defaults if you hardcode) ----
uniform float pdj_a = 0.1;
uniform float pdj_b = 1.9;
uniform float pdj_c = -0.8;
uniform float pdj_d = -1.2;

uniform float popcorn_c = 0.1;
uniform float popcorn_f = 0.09;

uniform float waves_b = 0.3;
uniform float waves_c = 0.5;
uniform float waves_e = 1.0;
uniform float waves_f = 1.0;

uniform float rings_c = 0.9;

uniform float fan_c = 0.1;
uniform float fan_f = 0.4;

uniform float blob_high = 0.9;
uniform float blob_low  = 0.1;
uniform float blob_waves = 10.0;


vec2 v_sinusoidal(vec2 v, float amount) {
    return vec2(amount * sin(v.x), amount * sin(v.y));
}



vec2 v_linear(vec2 v, float amount) {
    return amount * v;
}

vec2 v_hyperbolic(vec2 v, float amount) {
    float r = getR(v) + 1.0e-10;
    float theta = getTheta(v);
    float x = amount * sin(theta) / r;
    float y = amount * cos(theta) * r;
    return vec2(x, y);
}

vec2 v_polar(vec2 v, float amount) {
    float theta = getTheta(v);
    float r = getR(v);
    return vec2(amount * (theta / PI),
                amount * (r - 1.0));
}

vec2 v_disc(vec2 v, float amount) {
    float theta = getTheta(v);
    float r = getR(v);
    float x = (theta / PI) * sin(PI * r);
    float y = (theta / PI) * cos(PI * r);
    return amount * vec2(x, y);
}

vec2 v_handkerchief(vec2 v, float amount) {
    float theta = getTheta(v);
    float r = getR(v);
    float x = r * sin(theta + r);
    float y =      cos(theta - r);
    return amount * vec2(x, y);
}

vec2 v_pdj(vec2 v, float amount) {
    return amount * vec2(
        sin(pdj_a * v.y) - cos(pdj_b * v.x),
        sin(pdj_c * v.x) - cos(pdj_d * v.y)
    );
}

vec2 v_rect(vec2 v, float amount) {
    // WARNING: divides by amount; amount must be non-zero
    float a = max(abs(amount), 1e-10);
    float x = a * (a * (2.0 * floor(v.x / a) + 1.0) - v.x);
    float y = a * (a * (2.0 * floor(v.y / a) + 1.0) - v.y);
    return vec2(x, y);
}

vec2 v_swirl(vec2 v, float amount) {
    float r = getR(v);
    float rr = r * r;
    float x = v.x * sin(rr) - v.y * cos(rr);
    float y = v.x * cos(rr) + v.y * sin(rr);
    return amount * vec2(x, y);
}

vec2 v_horseshoe(vec2 v, float amount) {
    float r = getR(v);
    float j = rcp_safe(r);
    float x = j * ((v.x - v.y) * (v.x + v.y));
    float y = j * (2.0 * v.x * v.y);
    return amount * vec2(x, y);
}

vec2 v_popcorn(vec2 v, float amount) {
    float x = v.x + popcorn_c * sin(tan(3.0 * v.y));
    float y = v.y + popcorn_f * sin(tan(3.0 * v.x));
    return amount * vec2(x, y);
}

vec2 v_julia(vec2 v, float amount, float rand01) {
    float r = amount * sqrt(getR(v));
    float theta = 0.5 * getTheta(v);
    float add = (rand01 < 0.5) ? 0.0 : PI;   // meaningful branch
    theta += add;
    return r * vec2(cos(theta), sin(theta));
}

vec2 v_sech(vec2 p, float amount) {
    float d = cos(2.0 * p.y) + cosh(2.0 * p.x);
    d = (abs(d) < 1e-10) ? 1e-10 : d;
    float k = amount * 2.0 / d;
    return vec2(k * cos(p.y) * cosh(p.x),
               -k * sin(p.y) * sinh(p.x));
}

vec2 v_spherical(vec2 p, float amount) {
    float r = getR(p);
    float inv = rcp_safe(r * r); // 1 / r^2
    return amount * p * inv;
}

vec2 v_heart(vec2 v, float amount) {
    float theta = getTheta(v);
    float r = getR(v);
    float x = r * sin(theta * r);
    float y = r * cos(theta * -r);
    return amount * vec2(x, y);
}

vec2 v_spiral(vec2 v, float amount) {
    float theta = getTheta(v);
    float r = getR(v);
    float invr = rcp_safe(r);
    float x = invr * cos(theta) + sin(r) * invr;
    float y = invr * sin(theta) - cos(r) * invr;
    return amount * vec2(x, y);
}

vec2 v_diamond(vec2 v, float amount) {
    float theta = getTheta(v);
    float r = getR(v);
    float x = sin(theta) * cos(r);
    float y = cos(theta) * sin(r);
    return amount * vec2(x, y);
}

vec2 v_ex(vec2 v, float amount) {
    float theta = getTheta(v);
    float r = getR(v);
    float p0 = sin(theta + r);
    float p1 = cos(theta - r);
    float x = r * (p0*p0*p0) + r * (p1*p1*p1);
    float y = r * (p0*p0*p0) - r * (p1*p1*p1);
    return amount * vec2(x, y);
}

vec2 v_bent(vec2 v, float amount) {
    vec2 p = v;
    if (v.x >= 0.0 && v.y >= 0.0) {
        p = vec2(v.x, v.y) * amount;
    } else if (v.x < 0.0 && v.y >= 0.0) {
        p = vec2(2.0 * v.x, v.y) * amount;
    } else if (v.x >= 0.0 && v.y < 0.0) {
        p = vec2(v.x, 0.5 * v.y) * amount;
    } else { // v.x < 0 && v.y < 0
        p = vec2(2.0 * v.x, 0.5 * v.y) * amount;
    }
    return p;
}

vec2 v_waves(vec2 v, float amount) {
    float x = v.x + waves_b * sin(v.y / max(waves_c*waves_c, 1e-10));
    float y = v.y + waves_e * sin(v.x / max(waves_f*waves_f, 1e-10));
    return amount * vec2(x, y);
}

vec2 v_fisheye(vec2 v, float amount) {
    float r = getR(v);
    float k = 2.0 / (r + 1.0);
    return amount * (k * v);
}

vec2 v_exponential(vec2 v, float amount) {
    float factor = exp(v.x - 1.0);
    float x = factor * cos(PI * v.y);
    float y = factor * sin(PI * v.y);
    return amount * vec2(x, y);
}

vec2 v_power(vec2 v, float amount) {
    float r = getR(v);
    float theta = getTheta(v);
    float powR = pow(r, sin(theta));
    return amount * vec2(cos(theta), sin(theta)) * powR;
}

vec2 v_cosine(vec2 v, float amount) {
    float x = cos(PI * v.x) * cosh(v.y);
    float y = -sin(PI * v.x) * sinh(v.y);
    return amount * vec2(x, y);
}

vec2 v_rings(vec2 v, float amount) {
    float r = getR(v);
    float theta = getTheta(v);
    float c2 = rings_c * rings_c;
    float factor = mod((r + c2), (2.0 * c2)) - c2 + r * (1.0 - c2);
    return amount * vec2(cos(theta), sin(theta)) * factor;
}

vec2 v_fan(vec2 v, float amount) {
    float t = PI * (fan_c * fan_c);
    float theta = getTheta(v);
    float r = getR(v);

    float x, y;
    if (mod(theta + fan_f, t) > t * 0.5) {
        x = r * cos(theta - t * 0.5);
        y =     sin(theta - t * 0.5);
    } else {
        x = r * cos(theta + t * 0.5);
        y =     sin(theta + t * 0.5);
    }

    return amount * vec2(x, y);
}

vec2 v_blob(vec2 v, float amount) {
    float theta = getTheta(v);
    float r = getR(v);
    float factor = blob_low + ((blob_high - blob_low) * 0.5) * (sin(blob_waves * theta) + 1.0);
    return amount * vec2(cos(theta), sin(theta)) * (r * factor);
}

void processParticle(vec3 p, float iters, vec2 resolution) {
    for (float i = 0.0; i < iters; i += 1.0) {

        // variation chain
        p.xy = v_blob(v_rect(p.xy, 0.5), 0.5);

        // Processing: currentMode.wrap(v)
        p.xy = wrapRepeat(p.xy, u_x1, u_x2, u_y1, u_y2);

        // Processing: v + 0.003 * randomGaussian()
        // (not perfectly correlated with your Processing RNG, but comparable magnitude/distribution)
        p.xy += 0.003 * randn2();

        // Processing: map(...) into pixels with margin
        vec2 pixf = domainToPixel(p.xy, resolution);
        ivec2 pix = ivec2(pixf);

        if (pix.x >= 0 && pix.x < int(resolution.x) &&
            pix.y >= 0 && pix.y < int(resolution.y)) {
            imageAtomicAdd(computeTex, pix, 1u);
        }
    }
}

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= int(v2Resolution.x) || gid.y >= int(v2Resolution.y)) return;

    // Seed RNG deterministically per invocation (as you had)
    int id = gid.x + gid.y * int(v2Resolution.x);
    seed = uint(id) + 1235125u;

    // IMPORTANT: start p.xy in a centered symmetric domain, not [0,1)
    // Choose something roughly comparable to your Processing "x,y" inputs.
    // If your Processing uses (x,y) from a grid in [x1,x2]x[y1,y2], this is the right idea.
    vec2 v0 = vec2(
        mix(u_x1, u_x2, rand01()),
        mix(u_y1, u_y2, rand01())
    );

    vec3 p = vec3(v0, 1.0); // z unused (keep stable)

    processParticle(p, 10.0, v2Resolution);
}
