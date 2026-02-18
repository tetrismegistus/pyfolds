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
float log10(float x)
{
    return log(x) * 0.4342944819032518; // 1 / ln(10)
}
// ---- tunable parameters (set from host, or leave defaults if you hardcode) ----
uniform float pdj_a;
uniform float pdj_b;
uniform float pdj_c;
uniform float pdj_d;

uniform float dejong_a;
uniform float dejong_b;
uniform float dejong_c;
uniform float dejong_d;

uniform int n;

uniform float popcorn_c;
uniform float popcorn_f;

uniform float waves_b;
uniform float waves_c;
uniform float waves_e;
uniform float waves_f;

uniform float rings_c;
uniform float rings2_c;

uniform float fan_c;
uniform float fan_f;
uniform float fan2_x;
uniform float fan2_y;

uniform float blob_high;
uniform float blob_low;
uniform float blob_waves;

uniform float perspective_a;
uniform float perspective_d;

uniform float julian_power;   
uniform float julian_dist;    

uniform float juliascope_power;
uniform float juliascope_dist;    

uniform float radialblur_angle;

uniform float pie_slices;   
uniform float pie_rotation;  
uniform float pie_thickness;   

uniform float ngon_power;    
uniform float ngon_sides;  
uniform float ngon_corners;  
uniform float ngon_circle;  

uniform float curl_c1;
uniform float curl_c2;

uniform float rectangles_x;
uniform float rectangles_y;

uniform float arch_span;  

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

float rand01_open() {
    // maps uint in [0, 0xffffffff] -> float in (0,1)
    float u = hash_f();               // [0,1]
    return (u * (4294967295.0 - 2.0) + 1.0) / 4294967295.0; // (0,1)
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

vec2 v_rectangles(vec2 v, float amount) {
    float px = max(abs(rectangles_x), 1e-6);
    float py = max(abs(rectangles_y), 1e-6);

    float x = ( (2.0 * floor(v.x / px) + 1.0) * px ) - v.x;
    float y = ( (2.0 * floor(v.y / py) + 1.0) * py ) - v.y;

    return amount * vec2(x, y);
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

vec2 v_rings2(vec2 v, float amount) {
    float p = rings2_c * rings2_c;
    float r = getR(v);
    float theta = getTheta(v);
    float t = r - 2 * p * trunc((r + p)/(2 * p)) + r * (1.0 - p);
    return vec2(t * sin(theta), t * cos(theta)) * amount;
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

vec2 v_fan2(vec2 v, float amount) {
    float p1 = PI * (fan2_x * fan2_x);
    float p2 = fan2_y;

    float theta = getTheta(v);
    float r     = getR(v);

    if (abs(p1) < 1e-8) {
        return amount * v;
    }

    float t = theta + p2 - p1 * trunc((2.0 * theta * p2) / p1);
    float a = (t > 0.5 * p1) ? (theta - 0.5 * p1) : (theta + 0.5 * p1);
    return amount * (r * vec2(sin(a), cos(a)));
}

vec2 v_blob(vec2 v, float amount) {
    float theta = getTheta(v);
    float r = getR(v);
    float factor = blob_low + ((blob_high - blob_low) * 0.5) * (sin(blob_waves * theta) + 1.0);
    return amount * vec2(cos(theta), sin(theta)) * (r * factor);
}

vec2 v_eyefish(vec2 v, float amount) {
    float r = getR(v);
    float i = 2.0/(r+1);
    return vec2(i * v.x, i * v.y) * amount;
}

vec2 v_bubble(vec2 v, float amount) {
    float r = getR(v);
    float i = 4.0/((r * r) + 4.0);
    return i * vec2(v.x, v.y) * amount;
}

vec2 v_cylinder(vec2 v, float amount) {
    return amount * vec2(sin(v.x), sin(v.y));
}

vec2 v_perspective(vec2 v, float amount) {
    float i = perspective_d / (perspective_d - v.y * sin(perspective_a));
    return i * vec2(v.x, v.y * cos(perspective_a)) * amount;
}

vec2 v_noise(vec2 v, float amount) {
    float psi1 = rand01();           
    float psi2 = rand01();                 
    float a    = 6.28318530718 * psi2;   

    vec2 outv = psi1 * vec2(v.x * cos(a), v.y * sin(a));
    return amount * outv;
}

vec2 v_julian(vec2 v, float amount)
{
    float power = julian_power;
    float dist  = julian_dist;

    if (abs(power) < 1e-6)
        return amount * v;

    float r = getR(v);
    float theta = getTheta(v);
    float abs_p = abs(power);
    float branch = floor(abs_p * rand01());
    float t = (theta + 6.28318530718 * branch) / power;
    float radius = amount * pow(r, dist / power);

    return radius * vec2(cos(t), sin(t));
}

vec2 v_juliascope(vec2 v, float amount)
{
    float power = juliascope_power;
    float dist  = juliascope_dist;

    if (abs(power) < 1e-6)
        return amount * v;

    float r = getR(v);
    float theta = getTheta(v);

    float abs_p = abs(power);
    float branch = floor(abs_p * rand01());
    float lambda = (rand01() < 0.5) ? -1.0 : 1.0;
    float t = (lambda * theta + 6.28318530718 * branch) / power;
    float radius = amount * pow(r, dist / power);

    return radius * vec2(cos(t), sin(t));
}

vec2 v_blur(vec2 v, float amount)
{
    float psi1 = rand01();                 // radius in [0,1]
    float psi2 = rand01();                 // angle fraction in [0,1]
    float a    = 6.28318530718 * psi2;
    return amount * (psi1 * vec2(cos(a), sin(a)));
}

vec2 v_gaussian(vec2 v, float amount)
{
    float r =
        rand01() +
        rand01() +
        rand01() +
        rand01() - 2.0;

    float a = 6.28318530718 * rand01();
    return amount * (r * vec2(cos(a), sin(a)));
}

vec2 v_radialblur(vec2 v, float amount)
{
    float angle = radialblur_angle * 1.57079632679; // π/2

    // gaussian approx
    float g =
        rand01() +
        rand01() +
        rand01() +
        rand01() - 2.0;

    float t1 = amount * g;
    float t2 = getTheta(v) + t1 * sin(angle);
    float t3 = t1 * cos(angle) - 1.0;

    float r = getR(v);

    vec2 outv = vec2(
        r * cos(t2) + t3 * v.x,
        r * sin(t2) + t3 * v.y
    );

    return outv / amount;
}

vec2 v_pie(vec2 v, float amount)
{
    float slices    = max(1.0, floor(pie_slices));
    float rotation  = pie_rotation;
    float thickness = pie_thickness;

    float t1 = floor(rand01() * slices + 0.5);
    float t2 = rotation + (6.28318530718 / slices) * (t1 + rand01() * thickness);
    float r = rand01();

    return amount * (r * vec2(cos(t2), sin(t2)));
}

vec2 v_ngon(vec2 v, float amount)
{
    float power   = ngon_power;
    float sides   = max(1.0, floor(ngon_sides + 0.5));
    float corners = ngon_corners;
    float circle  = ngon_circle;

    float r = getR(v);
    if (r < 1e-6) return vec2(0.0);

    float phi = getTheta(v);

    float p2 = 6.28318530718 / sides;

    float t3 = phi - p2 * floor(phi / p2);
    float t4 = (t3 > p2 * 0.5) ? t3 : t3 - p2;

    float k = (corners * (1.0 / cos(t4) - 1.0) + circle) / pow(r, power);

    return amount * k * v;
}

vec2 v_curl(vec2 v, float amount) {
    float t1 = 1.0 + curl_c1 * v.x + curl_c2 * (v.x * v.x - v.y * v.y);
    float t2 = curl_c1 * v.y + 2.0 * curl_c2 * v.x * v.y ;
    float t3 = 1.0 / (t1 * t1 + t2 * t2);
    return t3 * vec2(v.x * t1 + v.y * t2, v.y * t1 - v.x * t2) * amount;
}

vec2 v_arch(vec2 v, float amount)
{
    float psi = rand01();

    float t = 3.141592653589793 * psi * arch_span;

    float s = sin(t);
    float c = cos(t);

    // keep asymptotes but avoid NaN
    c = (abs(c) < 1e-6) ? sign(c) * 1e-6 : c;

    vec2 outv = vec2(s, (s * s) / c);

    return amount * outv;
}

vec2 v_tangent(vec2 v, float amount) {
    return amount * vec2(sin(v.x)/cos(v.y), tan(v.y));
}

vec2 v_square(vec2 v, float amount) {
    float phi1 = rand01_open();
    float phi2 = rand01_open();
    return amount * vec2(phi1 - .5, phi2 - .5);
}

vec2 v_rays(vec2 v, float amount)
{
    float r2 = dot(v, v);
    if (r2 < 1e-10) return vec2(0.0);

    float psi = clamp(rand01(), 1e-7, 1.0 - 1e-7);

    float t = tan(3.141592653589793 * psi);

    vec2 outv = (t / r2) * vec2(cos(v.x), sin(v.y));

    return amount * outv;
}

vec2 v_blade(vec2 v, float amount)
{
    float psi = rand01(); 
    float r = getR(v);    

    float t = psi * r * amount;

    float c = cos(t);
    float s = sin(t);

    vec2 outv = v.x * vec2(c + s, c - s);
    return amount * outv;
}

vec2 v_secant(vec2 v, float amount)
{
    float r = length(v);

    float t = amount * r;
    float c = cos(t);

    // asymptotes are intentional but avoid NaN
    if (abs(c) < 1e-6) c = sign(c) * 1e-6;

    vec2 outv = vec2(
        v.x,
        1.0 / (amount * c)
    );

    return amount * outv;
}

vec2 v_twintian(vec2 v, float amount)
{
    float psi = rand01();
    float r = getR(v);

    float a = psi * r * amount;

    float s = sin(a);
    float s2 = s * s;

    // avoid log10(0)
    s2 = max(s2, 1e-12);

    float t = log10(s2) + cos(a);

    vec2 outv = v.x * vec2(
        t,
        t - 3.141592653589793 * s
    );

    return amount * outv;
}

vec2 v_cross(vec2 v, float amount)
{
    float d = v.x * v.x - v.y * v.y;
    d = max(abs(d), 1e-6);   // avoid division by zero

    vec2 outv = v / d;
    return amount * outv;
}


vec2 d_pdj(vec2 v, float amount) {
  float h = 0.1; // step
  float sqrth = sqrt(h);
  vec2 v1 = v_pdj(v, amount);
  vec2 v2 = v_pdj(vec2(v.x+h, v.y+h), amount);
  return vec2( (v2.x-v1.x)/sqrth, (v2.y-v1.y)/sqrth );
}

vec2 dejongsCurtains(vec2 v, float amount) {
    float x = sin(dejong_a * v.y) - cos(dejong_b * v.x);
    float y = sin(dejong_c * v.x) - cos(dejong_d * v.y);
    return vec2(x * amount, y * amount);
}


vec2 leviathan(vec2 v, float amount) {
     v = v_bent(v_horseshoe(v, 1.0), 1.0);
    for (int j = 0; j < 16; j++) v += v_julia(v, 1.5);
    return vec2(v.x * amount, v.y * amount); 
}

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
    for (int i = 0; i < n; i++) {
        float t = fGlobalTime;
        // gentle, slow rotation (period ~ 60–120s)
        v.xy = rot(.3 * sin(t * .3)) * v.xy; 
        
        vec2 p1 = v_cross(v_square(v, .5), .5);
        vec2 v = p1;
        v = wrap(v);
        v += 0.003 * randn2();

        ivec2 pix = ivec2(domainToPixel(v));
        if (pix.x >= 0 && pix.x < int(v2Resolution.x) &&
            pix.y >= 0 && pix.y < int(v2Resolution.y)) {
            imageAtomicAdd(hitsImg, pix, 1u);
        }
    }
}


void processTwo(uint sid) {
    // seed per "particle"
    seed = sid;

    // start in domain like Processing would
    vec2 v = vec2(
        mix(u_x1, u_x2, rand01()),
        mix(u_y1, u_y2, rand01())
    );

    // n iterations like your Processing loop (set to 3 here)
    for (int i = 0; i < n; i++) {
        float t = fGlobalTime;
        // gentle, slow rotation (period ~ 60–120s)
        v.xy = rot(.3 * sin(t * .3)) * v.xy; 
        
        
        vec2 p2 = leviathan(v, 1.0);
        vec2 v = v_disc(p2, 1.0);
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

    uint sid = uint(gid.x + gid.y * int(v2Resolution.x)) + 1235125u;
    processOne(sid);
}
