// shaders/hits.compute.glsl — your attractor/variation splats atomically into r32ui hits
#version 430
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

#define PI 3.1415926535897932384626433832795

// Wrap mode IDs (match these in Python)
#define WRAP_NO_WRAP        0
#define WRAP_SINUSOIDAL     1
#define WRAP_SPHERICAL      2
#define WRAP_MOD            3

uniform int u_wrap_mode;   // 0..3


uniform float fGlobalTime;
uniform vec2  v2Resolution;

uniform float u_x1;
uniform float u_x2;
uniform float u_y1;
uniform float u_y2;
uniform float u_margin_frac;

layout(r32ui, binding = 0) uniform coherent uimage2D hitsImg;

uint seed;
uint hashi(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
#define hash_f() ( float( seed = hashi(seed) ) / float(0xffffffffU) )

float getTheta(vec2 v) { return atan(v.x, v.y); }
float getR(vec2 v) { return length(v); }
float rcp_safe(float x) { return 1.0 / max(abs(x), 1e-10); }

// ---- tunable parameters (set from host, or leave defaults if you hardcode) ----
uniform float pdj_a;
uniform float pdj_b;
uniform float pdj_c;
uniform float pdj_d;

uniform float popcorn_c;
uniform float popcorn_f;

uniform float waves_b;
uniform float waves_c;
uniform float waves_e;
uniform float waves_f;

uniform float rings_c;

uniform float fan_c;
uniform float fan_f;

uniform float blob_high;
uniform float blob_low;
uniform float blob_waves;


// --- helpers matching your Processing MIN/MAX ---
float domainW() { return u_x2 - u_x1; }
float domainH() { return u_y2 - u_y1; }

// Gaussian-ish jitter
float rand01() { return hash_f(); }
float randn() {
    float u1 = max(rand01(), 1e-7);
    float u2 = rand01();
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}
vec2 randn2() { return vec2(randn(), randn()); }

// Processing-style mod wrap into [min,max)
float modWrap1(float x, float a, float b) {
    float w = b - a;
    x = mod(x - a, w);
    if (x < 0.0) x += w;
    return x + a;
}
vec2 modWrap(vec2 v) {
    return vec2(modWrap1(v.x, u_x1, u_x2),
                modWrap1(v.y, u_y1, u_y2));
}

// --- transforms used by wraps ---
vec2 v_sinusoidal_wrap(vec2 v, float amplitude) {
    // Processing: sin(v.x)*amplitude, sin(v.y)*amplitude
    return vec2(sin(v.x), sin(v.y)) * amplitude;
}

vec2 v_spherical_wrap(vec2 p, float amount) {
    // Processing: r = 1/pow(getR(p),2)
    float r2 = dot(p, p);
    float inv = 1.0 / max(r2, 1e-12);   // avoid blowups at ~0
    return p * (amount * inv);
}

// --- main wrap dispatcher ---
vec2 wrap(vec2 v) {
    if (u_wrap_mode == WRAP_NO_WRAP) {
        return v;
    }

    if (u_wrap_mode == WRAP_MOD) {
        return modWrap(v);
    }

    // amplitude/amount = (MAX_X - MIN_X)/2
    float amp = 0.5 * domainW();

    if (u_wrap_mode == WRAP_SINUSOIDAL) {
        vec2 vv = v_sinusoidal_wrap(v, amp);
        return vv; // Processing does NOT mod-wrap after sinusoidal
    }

    // WRAP_SPHERICAL
    vec2 vv = v_spherical_wrap(v, amp);
    return modWrap(vv); // Processing DOES mod-wrap after spherical
}


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

vec2 v_julia(vec2 v, float amount) {
    float r = amount * sqrt(getR(v));
    float theta = 0.5 * getTheta(v);

    // draw from the same deterministic RNG stream as everything else
    theta += (rand01() < 0.5) ? 0.0 : PI;

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





float map01(float x, float a, float b) { return (x - a) / (b - a); }

vec2 domainToPixel(vec2 v) {
    float w = v2Resolution.x;
    float h = v2Resolution.y;

    float mx = u_margin_frac * w;
    float my = u_margin_frac * h;

    float tx = clamp(map01(v.x, u_x1, u_x2), 0.0, 1.0);
    float ty = clamp(map01(v.y, u_y1, u_y2), 0.0, 1.0);

    return vec2(mix(mx, w - mx, tx),
                mix(my, h - my, ty));
}

mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c,-s,s,c); }

void processOne(uint sid) {
    // seed per "particle"
    seed = sid;

    // start in domain like Processing would
    vec2 v = vec2(
        mix(u_x1, u_x2, rand01()),
        mix(u_y1, u_y2, rand01())
    );

    // n iterations like your Processing loop (set to 3 here)
    const int n = 3;
    for (int i = 0; i < n; i++) {

        float t = fGlobalTime;
        // gentle, slow rotation (period ~ 60–120s)
        v.xy = rot(0.5 * sin(t * 0.8)) * v.xy; 
        
        
        vec2 p1 = v_pdj(v, 1.0) * v_disc(v, 1.);
        v = v_julia(p1, 1.);
        
        v = wrap(v);
        v += 0.003 * randn2();

        ivec2 pix = ivec2(domainToPixel(v));
        if (pix.x >= 0 && pix.x < int(v2Resolution.x) &&
            pix.y >= 0 && pix.y < int(v2Resolution.y)) {
            imageAtomicAdd(hitsImg, pix, 1u);
        }
    }
}

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= int(v2Resolution.x) || gid.y >= int(v2Resolution.y)) return;

    // One deterministic stream per invocation. You can scale this up later by launching
    // fewer invocations and looping many particles per invocation.
    uint sid = uint(gid.x + gid.y * int(v2Resolution.x)) + 1235125u;
    processOne(sid);
}
