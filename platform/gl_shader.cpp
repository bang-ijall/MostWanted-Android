#include "gl_shader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace glshader {
using namespace gpu;

const char* kVertexSource = R"(#version 300 es
precision highp float;
layout(location = 0) in vec4 a_clip;
layout(location = 1) in vec4 a_v0;
layout(location = 2) in vec4 a_v1;
layout(location = 3) in vec4 a_fog;
layout(location = 4) in vec4 a_tc0;
layout(location = 5) in vec4 a_tc1;
layout(location = 6) in vec4 a_tc2;
layout(location = 7) in vec4 a_tc3;
uniform vec4 u_surf;  // 2 / width, 2 / height, 2 / depth units
out vec4 v_v0;
out vec4 v_v1;
out vec4 v_tc0;
out vec4 v_tc1;
out vec4 v_tc2;
out vec4 v_tc3;
out float v_fog;
void main() {
    // x and y in pixels from the top left, z in depth buffer units, all times w
    gl_Position = vec4(a_clip.x * u_surf.x - a_clip.w, a_clip.y * u_surf.y - a_clip.w,
                       0.99995 * (a_clip.z * u_surf.z - a_clip.w), a_clip.w);
    v_v0 = a_v0;
    v_v1 = a_v1;
    v_tc0 = a_tc0;
    v_tc1 = a_tc1;
    v_tc2 = a_tc2;
    v_tc3 = a_tc3;
    v_fog = a_fog.x;
}
)";

namespace {

std::string F(float v) {
    char b[48];
    std::snprintf(b, sizeof b, "%.9g", v);
    std::string s = b;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
    return s;
}
std::string Idx(int i) { return std::to_string(i); }

// GLSL for one fragment's shading
struct Gen {
    const PixelKey& k;
    const TexInfo* info;
    bool atest;
    std::ostringstream o;
    int uid = 0;

    Gen(const PixelKey& key, const TexInfo* i, bool a) : k(key), info(i), atest(a) {}

    // sample stage i at a vec2 expression; rect textures take texels
    std::string Tex(int i, const std::string& uv, bool norm) {
        if (!info[i].kind) return "vec4(0.0, 0.0, 0.0, 1.0)";
        if (info[i].kind == kTexCube) return "texture(u_s" + Idx(i) + ", vec3(" + uv + ", 1.0))";
        std::string c = uv;
        if (norm && info[i].rect) c = "(" + uv + ") / u_tsz[" + Idx(i) + "].xy";
        return "texture(u_s" + Idx(i) + ", " + c + ")";
    }
    // a cube along a direction; a plain texture takes x y as its coordinates
    std::string Dir(int i, const std::string& d) {
        if (!info[i].kind) return "vec4(0.0, 0.0, 0.0, 1.0)";
        if (info[i].kind == kTexCube) return "texture(u_s" + Idx(i) + ", " + d + ")";
        return "texture(u_s" + Idx(i) + ", (" + d + ").xy)";
    }

    // ---- register combiners

    std::string Var(uint32_t reg, int stage, bool clamp_sum, bool inv_v1, bool inv_r0) {
        uint32_t flags = k.count >> 8;
        switch (reg) {
            case 1: return "u_pc[" + Idx(stage == 8 ? 16 : ((flags & 0x10) ? stage * 2 : 0)) + "]";
            case 2: return "u_pc[" + Idx(stage == 8 ? 17 : ((flags & 0x100) ? stage * 2 + 1 : 1)) + "]";
            case 3: return "fogv";
            case 4: return "v0";
            case 5: return "v1";
            case 8: return "t0";
            case 9: return "t1";
            case 10: return "t2";
            case 11: return "t3";
            case 12: return "r0";
            case 13: return "r1";
            case 14: {
                std::string a = inv_v1 ? "(vec3(1.0) - v1.rgb)" : "v1.rgb", b = inv_r0 ? "(vec3(1.0) - r0.rgb)" : "r0.rgb";
                std::string s = "(" + a + " + " + b + ")";
                if (clamp_sum) s = "clamp(" + s + ", 0.0, 1.0)";
                return "vec4(" + s + ", 0.0)";
            }
            case 15: return "vec4(eE * eF, 0.0)";
            default: return "vec4(0.0)";
        }
    }

    // an input byte: register, alpha or blue channel, input mapping; a vec3, or a float for the alpha half
    std::string Input(uint32_t v, bool alpha, int stage, bool cs, bool i1, bool i0) {
        std::string x = Var(v & 0xF, stage, cs, i1, i0);
        bool a = v & 0x10;
        std::string c;
        if (alpha) c = a ? "(" + x + ").a" : "(" + x + ").b";
        else c = a ? "vec3((" + x + ").a)" : "(" + x + ").rgb";
        switch (v & 0xE0) {
            case 0x00: return "max(" + c + ", 0.0)";
            case 0x20: return "(1.0 - clamp(" + c + ", 0.0, 1.0))";
            case 0x40: return "(2.0 * max(" + c + ", 0.0) - 1.0)";
            case 0x60: return "(-2.0 * max(" + c + ", 0.0) + 1.0)";
            case 0x80: return "(max(" + c + ", 0.0) - 0.5)";
            case 0xA0: return "(-max(" + c + ", 0.0) + 0.5)";
            case 0xC0: return c;
            default: return "(-" + c + ")";
        }
    }

    static std::string OutMap(const std::string& v, uint32_t map) {
        switch (map) {
            case 0x08: return "(" + v + " - 0.5)";
            case 0x10: return "(" + v + " * 2.0)";
            case 0x18: return "((" + v + " - 0.5) * 2.0)";
            case 0x20: return "(" + v + " * 4.0)";
            case 0x30: return "(" + v + " / 2.0)";
            default: return v;
        }
    }

    static const char* Dest(uint32_t reg) {
        switch (reg) {
            case 4: return "v0";
            case 5: return "v1";
            case 8: return "t0";
            case 9: return "t1";
            case 10: return "t2";
            case 11: return "t3";
            case 12: return "r0";
            case 13: return "r1";
            default: return nullptr;
        }
    }

    // one half of a stage: locals are computed from the registers as they are, the writes are returned
    void Half(uint32_t inputs, uint32_t output, bool alpha, int stage, std::string& writes) {
        std::string p = "h" + Idx(uid++) + "_";
        const char* T = alpha ? "float" : "vec3";
        o << "    " << T << " " << p << "A = " << Input(inputs >> 24, alpha, stage, false, false, false) << ";\n";
        o << "    " << T << " " << p << "B = " << Input((inputs >> 16) & 0xFF, alpha, stage, false, false, false) << ";\n";
        o << "    " << T << " " << p << "C = " << Input((inputs >> 8) & 0xFF, alpha, stage, false, false, false) << ";\n";
        o << "    " << T << " " << p << "D = " << Input(inputs & 0xFF, alpha, stage, false, false, false) << ";\n";
        uint32_t fl = output >> 12, map = fl & 0x38;
        std::string A = p + "A", B = p + "B", C = p + "C", D = p + "D";
        std::string x, y;
        if (alpha) {
            x = A + " * " + B;
            y = C + " * " + D;
        } else {
            x = (fl & 2) ? "vec3(dot(" + A + ", " + B + "))" : A + " * " + B;
            y = (fl & 1) ? "vec3(dot(" + C + ", " + D + "))" : C + " * " + D;
        }
        o << "    " << T << " " << p << "x = " << x << ";\n";
        o << "    " << T << " " << p << "y = " << y << ";\n";
        std::string sum;
        if (fl & 4) {
            bool flag = (k.count >> 8) & 1;
            std::string pick = flag ? "(r0.a >= 0.5)" : "((int(r0.a * 255.0) & 1) == 1)";
            sum = "(" + pick + " ? " + p + "y : " + p + "x)";
        } else {
            sum = "(" + p + "x + " + p + "y)";
        }
        const char* lo = "-1.0";
        auto cl = [&](const std::string& v) { return "clamp(" + OutMap(v, map) + ", " + lo + ", 1.0)"; };
        o << "    " << T << " " << p << "ab = " << cl(p + "x") << ";\n";
        o << "    " << T << " " << p << "cd = " << cl(p + "y") << ";\n";
        o << "    " << T << " " << p << "ms = " << cl(sum) << ";\n";
        auto put = [&](uint32_t reg, const std::string& v, bool blue_to_alpha) {
            const char* d = Dest(reg);
            if (!d) return;
            if (alpha) {
                writes += std::string("    ") + d + ".a = " + v + ";\n";
                return;
            }
            writes += std::string("    ") + d + ".rgb = " + v + ";\n";
            if (blue_to_alpha) writes += std::string("    ") + d + ".a = " + v + ".b;\n";
        };
        put((output >> 4) & 0xF, p + "ab", fl & 0x80);
        put(output & 0xF, p + "cd", fl & 0x40);
        put((output >> 8) & 0xF, p + "ms", false);
    }

    // texture stage i by its mode
    void Fetch(int i) {
        uint32_t mode = (k.tex_modes >> (i * 5)) & 0x1F;
        int src = int(i == 1 ? 0 : i == 2 ? (k.input_texture >> 16) & 0xF : i == 3 ? (k.input_texture >> 20) & 0xF : 0) & 3;
        uint32_t map = i == 0 ? 0 : (k.dot_mapping >> ((i - 1) * 4)) & 0xF;
        std::string q = "v_tc" + Idx(i), t = "t" + Idx(i), ts = "t" + Idx(src), dp = "dp" + Idx(i);
        auto dpn = [&](int j) { return "dp" + Idx(j < 0 ? 0 : j); };
        auto dot = [&] { o << "    " << dp << " = dot(" << q << ".xyz, dotin(" << map << ", " << ts << "));\n"; };
        switch (mode) {
            case 0: o << "    " << t << " = vec4(0.0, 0.0, 0.0, 1.0);\n"; break;
            case 1:
            case 2:
                if (info[i].shadow) {
                    o << "    " << t << " = vec4(1.0);\n";
                } else if (mode == 1 && info[i].kind == kTexCube) {
                    o << "    { float w = " << q << ".w == 0.0 ? 1.0 : " << q << ".w;\n";
                    o << "      " << t << " = " << Dir(i, "vec3(" + q + ".xy / w, 1.0)") << "; }\n";
                } else {
                    o << "    { float w = " << q << ".w == 0.0 ? 1.0 : " << q << ".w;\n";
                    o << "      " << t << " = " << Tex(i, q + ".xy / w", true) << "; }\n";
                }
                break;
            case 3:
                if (info[i].kind == kTexCube) o << "    " << t << " = " << Dir(i, q + ".xyz") << ";\n";
                else o << "    " << t << " = " << Tex(i, q + ".xy", false) << ";\n";
                break;
            case 4: o << "    " << t << " = " << q << ";\n"; break;
            case 5:
                o << "    " << t << " = vec4(0.0);\n";
                for (int j = 0; j < 4; j++) {
                    bool ge = (k.compare_mode >> (i * 4 + j)) & 1;
                    o << "    if (" << q << "[" << j << "]" << (ge ? " >= 0.0" : " < 0.0") << ") discard;\n";
                }
                break;
            case 6:
            case 7: {  // bump env map: signed du dv through the 2x2 matrix, 7 scales by luminance
                std::string I = Idx(i);
                o << "    { float du = sbyte(" << ts << ".b, 0.0, 256.0, 127.0), dv = sbyte(" << ts << ".g, 0.0, 256.0, 127.0);\n";
                o << "      vec2 uv = vec2(" << q << ".x + (u_bm[" << I << "].x * du + u_bm[" << I << "].z * dv), " << q << ".y + (u_bm[" << I << "].y * du + u_bm[" << I << "].w * dv));\n";
                o << "      " << t << " = " << Tex(i, "uv", true) << ";\n";
                if (mode == 7) o << "      " << t << " *= " << ts << ".r * u_bl[" << I << "].x + u_bl[" << I << "].y;\n";
                o << "    }\n";
                break;
            }
            case 9:
                dot();
                o << "    " << t << " = " << Tex(i, "vec2(" + dpn(i - 1) + ", " + dp + ")", true) << ";\n";
                break;
            case 0xA:
            case 0x11: dot(); o << "    " << t << " = vec4(0.0);\n"; break;
            case 0xB: {
                dot();
                int s3 = int((k.input_texture >> 20) & 0xF) & 3;
                o << "    { float nx = dot(v_tc3.xyz, dotin(" << ((k.dot_mapping >> 8) & 0xF) << ", t" << s3 << "));\n";
                o << "      " << t << " = " << Dir(i, "vec3(" + dpn(i - 1) + ", " + dp + ", nx)") << "; }\n";
                break;
            }
            case 0xC: {
                dot();
                std::string e = "vec3(v_tc" + Idx(i - 2 < 0 ? 0 : i - 2) + ".w, v_tc" + Idx(i - 1 < 0 ? 0 : i - 1) + ".w, " + q + ".w)";
                o << "    { vec3 n = vec3(" << dpn(i - 2) << ", " << dpn(i - 1) << ", " << dp << "); vec3 eye = " << e << ";\n";
                o << "      vec3 r = 2.0 * n * dot(n, eye) / dot(n, n) - eye;\n";
                o << "      " << t << " = " << Dir(i, "r") << "; }\n";
                break;
            }
            case 0xD:
                dot();
                o << "    " << t << " = " << Tex(i, "vec2(" + dpn(i - 2) + ", " + dpn(i - 1) + ")", false) << ";\n";
                break;
            case 0xE:
                dot();
                o << "    " << t << " = " << Dir(i, "vec3(" + dpn(i - 2) + ", " + dpn(i - 1) + ", " + dp + ")") << ";\n";
                break;
            case 0xF: o << "    " << t << " = " << Tex(i, "vec2(" + ts + ".a, " + ts + ".r)", false) << ";\n"; break;
            case 0x10: o << "    " << t << " = " << Tex(i, "vec2(" + ts + ".g, " + ts + ".b)", false) << ";\n"; break;
            default: o << "    " << t << " = vec4(0.0);\n"; break;
        }
        if (mode && mode != 4 && mode != 5 && ((k.alphakill >> i) & 1)) o << "    if (" << t << ".a == 0.0) discard;\n";
    }

    void Programmable() {
        o << "    vec4 t0 = vec4(0.0), t1 = vec4(0.0), t2 = vec4(0.0), t3 = vec4(0.0);\n    float dp0 = 0.0, dp1 = 0.0, dp2 = 0.0, dp3 = 0.0;\n";
        for (int i = 0; i < 4; i++) Fetch(i);
        o << "    vec4 v0 = v_v0, v1 = v_v1;\n";
        o << "    vec4 r0 = vec4(0.0, 0.0, 0.0, " << ((k.tex_modes & 0x1F) ? "t0.a" : "1.0") << ");\n";
        o << "    vec4 r1 = vec4(0.0);\n";
        o << "    vec3 eE = vec3(0.0), eF = vec3(0.0);\n";
        uint32_t n = std::min<uint32_t>(k.count & 0xFF, 8);
        for (uint32_t st = 0; st < n; st++) {
            o << "    {\n";
            std::string writes;
            Half(k.rgb_in[st], k.rgb_out[st], false, int(st), writes);
            Half(k.a_in[st], k.a_out[st], true, int(st), writes);
            o << writes << "    }\n";
        }
        if (!k.final_abcd && !k.final_efg) {
            o << "    col = vec4(mix(fogv.rgb, r0.rgb, fogv.a), r0.a);\n";
            return;
        }
        uint32_t fl = k.final_efg & 0xFF;
        bool cs = fl & 0x80, i1 = fl & 0x40, i0 = fl & 0x20;
        o << "    {\n";
        o << "    vec3 E = " << Input(k.final_efg >> 24, false, 8, cs, i1, i0) << ";\n";
        o << "    vec3 F = " << Input((k.final_efg >> 16) & 0xFF, false, 8, cs, i1, i0) << ";\n";
        o << "    eE = E; eF = F;\n";
        o << "    vec3 A = " << Input(k.final_abcd >> 24, false, 8, cs, i1, i0) << ";\n";
        o << "    vec3 B = " << Input((k.final_abcd >> 16) & 0xFF, false, 8, cs, i1, i0) << ";\n";
        o << "    vec3 C = " << Input((k.final_abcd >> 8) & 0xFF, false, 8, cs, i1, i0) << ";\n";
        o << "    vec3 D = " << Input(k.final_abcd & 0xFF, false, 8, cs, i1, i0) << ";\n";
        o << "    float G = " << Input((k.final_efg >> 8) & 0xFF, true, 8, cs, i1, i0) << ";\n";
        o << "    col = vec4(D + C * (vec3(1.0) - A) + B * A, G);\n";
        o << "    }\n";
    }

    // ---- fixed function stages

    std::string Arg(uint32_t a) {
        std::string x;
        switch (a & 0xF) {
            case 0: x = "v0"; break;
            case 2: x = "tex"; break;
            case 3: x = "u_tf"; break;
            case 4: x = "v1"; break;
            case 5: x = "tmp"; break;
            default: x = "cur"; break;
        }
        if (a & 0x20) x = "vec4(" + x + ".a)";
        if (a & 0x10) x = "(vec4(1.0) - " + x + ")";
        return x;
    }

    // D3DTOP on rgb (vec3) or alpha (float)
    std::string Op(uint32_t op, const std::string& a1, const std::string& a2, const std::string& a0, bool alpha) {
        const char* s = alpha ? ".a" : ".rgb";
        std::string x = a1 + s, y = a2 + s, z = a0 + s;
        switch (op) {
            case 2: return x;
            case 3: return y;
            case 4: return "(" + x + " * " + y + ")";
            case 5: return "(" + x + " * " + y + " * 2.0)";
            case 6: return "(" + x + " * " + y + " * 4.0)";
            case 7: return "(" + x + " + " + y + ")";
            case 8: return "(" + x + " + " + y + " - 0.5)";
            case 9: return "((" + x + " + " + y + " - 0.5) * 2.0)";
            case 10: return "(" + x + " - " + y + ")";
            case 11: return "(" + x + " + " + y + " * (1.0 - " + x + "))";
            case 12: return "(" + y + " + (" + x + " - " + y + ") * v0.a)";
            case 13: return "(" + y + " + (" + x + " - " + y + ") * cur.a)";
            case 14: return "(" + y + " + (" + x + " - " + y + ") * tex.a)";
            case 15: return "(" + y + " + (" + x + " - " + y + ") * u_tf.a)";
            case 16: return "(" + x + " + " + y + " * (1.0 - tex.a))";
            case 18: return alpha ? x : "(" + x + " + " + y + " * " + a1 + ".a)";
            case 19: return alpha ? x : "(" + x + " * " + y + " + " + a1 + ".a)";
            case 20: return alpha ? x : "(" + x + " + " + y + " * (1.0 - " + a1 + ".a))";
            case 21: return alpha ? x : "((1.0 - " + x + ") * " + y + " + " + a1 + ".a)";
            case 22: {
                std::string d = "clamp(dot((" + a1 + ".rgb - 0.5) * 2.0, (" + a2 + ".rgb - 0.5) * 2.0), 0.0, 1.0)";
                return alpha ? d : "vec3(" + d + ")";
            }
            case 23: return "(" + z + " + " + x + " * " + y + ")";
            case 24: return "(" + y + " + (" + x + " - " + y + ") * " + z + ")";
            default: return alpha ? "cur.a" : "cur.rgb";
        }
    }

    void Stages() {
        o << "    vec4 v0 = v_v0, v1 = v_v1;\n    vec4 cur = v0, tmp = vec4(0.0), tex;\n";
        for (int i = 0; i < 4; i++) {
            const uint32_t* st = k.ff[i];
            if (st[0] <= 1) break;
            o << "    {\n";
            std::string q = "v_tc" + Idx(i);
            // the console's stages divide by q under a vertex program or D3DTTFF_PROJECTED
            std::string uv = ((k.proj >> i) & 1) ? "(" + q + ".xy / (" + q + ".w == 0.0 ? 1.0 : " + q + ".w))" : q + ".xy";
            if (info[i].kind == kTexCube) o << "    tex = " << Dir(i, q + ".xyz") << ";\n";
            else if (info[i].kind) o << "    tex = " << Tex(i, uv, true) << ";\n";
            else o << "    tex = vec4(1.0);\n";
            if ((k.alphakill >> i) & 1) o << "    if (tex.a == 0.0) discard;\n";
            o << "    vec4 c1 = " << Arg(st[1]) << ", c2 = " << Arg(st[2]) << ", c0 = " << Arg(st[3]) << ";\n";
            o << "    vec4 r;\n";
            o << "    r.rgb = clamp(" << Op(st[0], "c1", "c2", "c0", false) << ", 0.0, 1.0);\n";
            if (st[4] <= 1) {
                o << "    r.a = cur.a;\n";
            } else {
                o << "    vec4 b1 = " << Arg(st[5]) << ", b2 = " << Arg(st[6]) << ", b0 = " << Arg(st[7]) << ";\n";
                o << "    r.a = clamp(" << Op(st[4], "b1", "b2", "b0", true) << ", 0.0, 1.0);\n";
            }
            o << "    " << (st[8] == 5 ? "tmp" : "cur") << " = r;\n";
            o << "    }\n";
        }
        o << "    cur.rgb = clamp(cur.rgb + v1.rgb, 0.0, 1.0);\n";
        o << "    col = vec4(mix(fogv.rgb, cur.rgb, fogv.a), cur.a);\n";
    }

    std::string Run() {
        o << "#version 300 es\nprecision highp float;\nprecision highp int;\n";
        o << "in vec4 v_v0;\nin vec4 v_v1;\nin vec4 v_tc0;\nin vec4 v_tc1;\nin vec4 v_tc2;\nin vec4 v_tc3;\nin float v_fog;\n";
        o << "uniform vec4 u_pc[18];\nuniform vec4 u_tf;\nuniform vec4 u_fogc;\nuniform vec4 u_at;\n";
        o << "uniform vec4 u_bm[4];\nuniform vec4 u_bl[4];\nuniform vec4 u_tsz[4];\n";
        for (int i = 0; i < 4; i++)
            if (info[i].kind) o << "uniform " << (info[i].kind == kTexCube ? "samplerCube" : "sampler2D") << " u_s" << i << ";\n";
        o << "float sbyte(float u, float lo, float hi, float d) { float b = u * 255.0; return (b - (b >= 128.0 ? hi : lo)) / d; }\n";
        o << "vec3 dotin(int map, vec4 t) {\n"
             "    float lo, hi, d;\n"
             "    if (map == 1) { lo = 128.0; hi = 128.0; d = 127.0; }\n"
             "    else if (map == 2) { lo = -0.5; hi = 255.5; d = 127.5; }\n"
             "    else if (map == 3) { lo = 0.0; hi = 256.0; d = 127.0; }\n"
             "    else return t.rgb;\n"
             "    return vec3(sbyte(t.r, lo, hi, d), sbyte(t.g, lo, hi, d), sbyte(t.b, lo, hi, d));\n}\n";
        o << "layout(location = 0) out vec4 o_color;\n";
        o << "bool passes(int f, int a, int b) {\n"
             "    if (f == 0x200) return false;\n    if (f == 0x201) return a < b;\n    if (f == 0x202) return a == b;\n"
             "    if (f == 0x203) return a <= b;\n    if (f == 0x204) return a > b;\n    if (f == 0x205) return a != b;\n"
             "    if (f == 0x206) return a >= b;\n    return true;\n}\n";
        o << "void main() {\n";
        o << "    vec4 fogv = vec4(u_fogc.rgb, clamp(v_fog, 0.0, 1.0));\n";
        o << "    vec4 col;\n";
        if (k.programmable) Programmable();
        else Stages();
        o << "    col = clamp(col, 0.0, 1.0);\n";
        if (atest) o << "    if (!passes(int(u_at.x), int(floor(col.a * 255.0 + 0.5)), int(u_at.y))) discard;\n";
        o << "    o_color = col;\n}\n";
        return o.str();
    }
};

}  // namespace

std::string VertexProgramSource(const std::vector<uint32_t>& prog, uint32_t variants, uint32_t w1_mask) {
    std::ostringstream o;
    // 0 * anything is 0 on the console, infinity included
    o << "#version 300 es\nprecision highp float;\nprecision highp int;\n";
    std::string unpack;  // inputs read some other way, made vec4 at the top of main
    for (int r = 0; r < 16; r++) {
        uint32_t v = (variants >> (r * 2)) & 3;
        std::string R = std::to_string(r);
        if (v == 0 && (w1_mask >> r & 1)) {
            o << "layout(location = " << r << ") in vec4 in_" << R << ";\n";
            unpack += "  vec4 i" + R + " = vec4(in_" + R + ".xyz, 1.0);\n";
        } else if (v == 0) {
            o << "layout(location = " << r << ") in vec4 i" << R << ";\n";
        } else if (v == 1) {
            o << "layout(location = " << r << ") in vec4 in_" << R << ";\n";
            unpack += "  vec4 i" + R + " = in_" + R + ".zyxw;\n";
        } else if (v == 2) {
            o << "layout(location = " << r << ") in uint in_" << R << ";\n";
            unpack += "  vec4 i" + R + " = packed3(in_" + R + ");\n";
        } else {
            o << "layout(location = " << r << ") in vec4 in_" << R << ";\n";
            unpack += "  vec4 i" + R + " = vec4(in_" + R + ".x, in_" + R + ".y, 0.0, in_" + R + ".z);\n";
        }
    }
    o << "uniform vec4 u_c[192];\nuniform vec4 u_surf;\nuniform vec4 u_fogp;\nuniform vec4 u_vsmisc;\n"
         "out vec4 v_v0;\nout vec4 v_v1;\nout vec4 v_tc0;\nout vec4 v_tc1;\nout vec4 v_tc2;\nout vec4 v_tc3;\nout float v_fog;\n"
         "vec4 packed3(uint p) {\n"
         "  int x = int(p << 21u) >> 21, y = int(p << 10u) >> 21, z = int(p) >> 22;\n"
         "  return vec4(max(float(x) / 1023.0, -1.0), max(float(y) / 1023.0, -1.0), max(float(z) / 511.0, -1.0), 1.0);\n}\n"
         "vec4 mmul(vec4 a, vec4 b) {\n"
         "  bvec4 z = bvec4(vec4(equal(a, vec4(0.0))) + vec4(equal(b, vec4(0.0))));\n"
         "  return mix(a * b, vec4(0.0), z);\n}\n"
         "float magc(float v) { float m = clamp(abs(v), 5.42101e-20, 1.84467e+19); return v < 0.0 ? -m : m; }\n"
         "const float INF = 3.402823e+38;\n"
         "vec4 c(int i) { return u_c[clamp(i, 0, 191)]; }\n"
         "void main() {\n";
    o << unpack;
    o << "  int a0 = 0;\n"
         "  vec4 R0 = vec4(0.0), R1 = vec4(0.0), R2 = vec4(0.0), R3 = vec4(0.0), R4 = vec4(0.0), R5 = vec4(0.0), R6 = vec4(0.0),\n"
         "       R7 = vec4(0.0), R8 = vec4(0.0), R9 = vec4(0.0), R10 = vec4(0.0), R11 = vec4(0.0);\n"
         "  vec4 oPos = vec4(0.0), oD0 = vec4(0.0, 0.0, 0.0, 1.0), oD1 = vec4(0.0), oFog = vec4(0.0), oPts = vec4(1.0);\n"
         "  vec4 oB0 = vec4(0.0, 0.0, 0.0, 1.0), oB1 = vec4(0.0);\n"
         "  vec4 oT0 = vec4(0.0, 0.0, 0.0, 1.0), oT1 = vec4(0.0, 0.0, 0.0, 1.0), oT2 = vec4(0.0, 0.0, 0.0, 1.0), oT3 = vec4(0.0, 0.0, 0.0, 1.0);\n";
    auto field = [](const uint32_t* t, int sub, int bit, int len) { return (t[sub] >> bit) & ((1u << len) - 1); };
    auto temp = [](uint32_t r) { return r == 12 ? std::string("oPos") : "R" + std::to_string(r > 12 ? 0 : r); };
    static const char* kOutName[16] = {"oPos", nullptr, nullptr, "oD0", "oD1", "oFog", "oPts", "oB0", "oB1", "oT0", "oT1", "oT2", "oT3", nullptr, nullptr, nullptr};
    static const char kLane[4] = {'x', 'y', 'z', 'w'};
    auto masked = [&](const std::string& dst, const std::string& src, uint32_t xyzw) {
        std::string m;
        for (int i = 0; i < 4; i++)
            if (xyzw >> (3 - i) & 1) m += kLane[i];
        if (m.empty()) return std::string();
        return "    " + dst + "." + m + " = " + src + "." + m + ";\n";
    };
    for (size_t n = 0; n + 4 <= prog.size(); n += 4) {
        const uint32_t* k = &prog[n];
        uint32_t mac = field(k, 1, 21, 4), ilu = field(k, 1, 25, 3), r = field(k, 3, 20, 4), omask = field(k, 3, 12, 4);
        bool use_ilu = field(k, 3, 2, 1) == 1, out_en = field(k, 3, 11, 1) != 0, last = field(k, 3, 0, 1) != 0;
        uint32_t outr = field(k, 3, 3, 8) & 0xF, mask_ilu = field(k, 3, 16, 4);
        uint32_t mask_mac = ilu && r == 1 ? 0 : field(k, 3, 24, 4);
        if (mac || ilu) {
            std::string src[3];
            for (int which = 0; which < 3; which++) {
                uint32_t sw[4], reg, mux, neg;
                if (which == 0) {
                    neg = field(k, 1, 8, 1);
                    sw[0] = field(k, 1, 6, 2), sw[1] = field(k, 1, 4, 2), sw[2] = field(k, 1, 2, 2), sw[3] = field(k, 1, 0, 2);
                    reg = field(k, 2, 28, 4), mux = field(k, 2, 26, 2);
                } else if (which == 1) {
                    neg = field(k, 2, 25, 1);
                    sw[0] = field(k, 2, 23, 2), sw[1] = field(k, 2, 21, 2), sw[2] = field(k, 2, 19, 2), sw[3] = field(k, 2, 17, 2);
                    reg = field(k, 2, 13, 4), mux = field(k, 2, 11, 2);
                } else {
                    neg = field(k, 2, 10, 1);
                    sw[0] = field(k, 2, 8, 2), sw[1] = field(k, 2, 6, 2), sw[2] = field(k, 2, 4, 2), sw[3] = field(k, 2, 2, 2);
                    reg = field(k, 2, 0, 2) << 2 | field(k, 3, 30, 2), mux = field(k, 3, 28, 2);
                    if (ilu >= 2 && ilu <= 6) sw[1] = sw[2] = sw[3] = sw[0];
                }
                std::string base;
                if (mux == 1) {
                    base = temp(reg);
                } else if (mux == 2) {
                    base = "i" + std::to_string(field(k, 1, 9, 4));
                } else {
                    int cr = std::min(int(field(k, 1, 13, 8)), 191);
                    base = field(k, 3, 1, 1) ? "c(" + std::to_string(cr) + " + a0)" : "u_c[" + std::to_string(cr) + "]";
                }
                std::string swz = ".";
                for (int i = 0; i < 4; i++) swz += kLane[sw[i]];
                src[which] = std::string(neg ? "(-" : "(") + base + swz + ")";
            }
            const std::string &A = src[0], &B = src[1], &C = src[2];
            o << "  {\n    vec4 A = " << A << ", B = " << B << ", C = " << C << ";\n";
            if (mac) {
                o << "    vec4 mr = ";
                switch (mac) {
                    case 1: case 13: o << "A"; break;
                    case 2: o << "mmul(A, B)"; break;
                    case 3: o << "A + C"; break;
                    case 4: o << "mmul(A, B) + C"; break;
                    case 5: o << "vec4(dot(A.xyz, B.xyz))"; break;
                    case 6: o << "vec4(dot(A.xyz, B.xyz) + B.w)"; break;
                    case 7: o << "vec4(dot(A, B))"; break;
                    case 8: o << "vec4(1.0, A.y * B.y, A.z, B.w)"; break;
                    case 9: o << "min(A, B)"; break;
                    case 10: o << "max(A, B)"; break;
                    case 11: o << "vec4(lessThan(A, B))"; break;
                    case 12: o << "vec4(greaterThanEqual(A, B))"; break;
                    default: o << "vec4(0.0)"; break;
                }
                o << ";\n";
            }
            if (ilu) {
                o << "    vec4 ir;\n    { float x = C.x;\n";
                switch (ilu) {
                    case 1: o << "      ir = C;\n"; break;
                    case 2: o << "      ir = vec4(1.0 / x);\n"; break;
                    case 3: o << "      ir = vec4(magc(1.0 / x));\n"; break;
                    case 4: o << "      ir = vec4(x == 0.0 ? INF : inversesqrt(abs(x)));\n"; break;
                    case 5: o << "      float f = floor(x); ir = vec4(exp2(f), x - f, exp2(x), 1.0);\n"; break;
                    case 6: o << "      float m = abs(x); if (m == 0.0) ir = vec4(-INF, 1.0, -INF, 1.0);\n"
                                 "      else { float e = floor(log2(m)); ir = vec4(e, m / exp2(e), log2(m), 1.0); }\n"; break;
                    case 7: o << "      const float cap = 127.99609375; float p = clamp(C.w, -cap, cap), d = max(C.x, 0.0), sp = max(C.y, 0.0);\n"
                                 "      ir = vec4(1.0, d, d > 0.0 ? exp2(p * log2(sp)) : 0.0, 1.0);\n"; break;
                    default: o << "      ir = vec4(0.0);\n"; break;
                }
                o << "    }\n";
            }
            // the output register first, then the temporaries
            if (omask && out_en && ((use_ilu && ilu) || (!use_ilu && mac && mac != 13)) && kOutName[outr]) {
                const char* sv = use_ilu ? "ir" : "mr";
                if (outr == 5) {  // oFog takes one lane per mask bit, from x
                    int lanes = __builtin_popcount(omask);
                    std::string m(kLane, kLane + lanes);
                    o << "    oFog." << m << " = " << sv << "." << m << ";\n";
                } else {
                    o << masked(kOutName[outr], sv, omask);
                }
            }
            if (ilu) {
                uint32_t dst = mac ? 1 : r;
                if (dst <= 12) o << masked(temp(dst), "ir", mask_ilu);
            }
            if (mac == 13) o << "    a0 = int(floor(mr.x + 0.001));\n";
            else if (mac && r <= 12) o << masked(temp(r), "mr", mask_mac);
            o << "  }\n";
        }
        if (last) break;
    }
    o << "  float w = magc(oPos.w);\n"
         "  gl_Position = vec4(oPos.x * w * u_surf.x - w, oPos.y * w * u_surf.y - w, 0.99995 * (oPos.z * w * u_surf.z - w), w);\n"
         "  v_v0 = clamp(oD0, 0.0, 1.0);\n"
         "  v_v1 = u_vsmisc.x != 0.0 ? clamp(oD1, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);\n"
         "  v_tc0 = oT0; v_tc1 = oT1; v_tc2 = oT2; v_tc3 = oT3;\n"
         "  float f = oFog.x, fa;\n"
         "  if (u_fogp.w < 0.0) fa = 1.0;\n"
         "  else if (u_fogp.w == 0.0) fa = clamp(f, 0.0, 1.0);\n"
         "  else if (u_fogp.w == 1.0) fa = exp(-f * u_fogp.z);\n"
         "  else if (u_fogp.w == 2.0) fa = exp(-(f * u_fogp.z) * (f * u_fogp.z));\n"
         "  else fa = u_fogp.y == u_fogp.x ? 1.0 : (u_fogp.y - f) / (u_fogp.y - u_fogp.x);\n"
         "  v_fog = fa;\n"
         "}\n";
    return o.str();
}

std::string FragmentSource(const PixelKey& key, const TexInfo info[4], bool alpha_test) {
    Gen g(key, info, alpha_test);
    return g.Run();
}

uint64_t CacheHash(const PixelKey& k, const TexInfo info[4], bool alpha_test) {
    uint64_t h = 1469598103934665603ull;
    auto put = [&](uint32_t v) { h = (h ^ v) * 1099511628211ull; h ^= h >> 29; };
    put(k.programmable);
    put(alpha_test);
    for (int i = 0; i < 4; i++) put(uint32_t(info[i].kind) | uint32_t(info[i].rect) << 8 | uint32_t(info[i].shadow) << 16);
    put(k.alphakill);
    if (k.programmable) {
        put(k.count);
        for (int i = 0; i < 8; i++) {
            put(k.rgb_in[i]);
            put(k.rgb_out[i]);
            put(k.a_in[i]);
            put(k.a_out[i]);
        }
        put(k.final_abcd);
        put(k.final_efg);
        put(k.tex_modes);
        put(k.input_texture);
        put(k.dot_mapping);
        put(k.compare_mode);
    } else {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 9; j++) put(k.ff[i][j]);
        put(k.proj);
    }
    return h;
}

std::string CacheKey(const PixelKey& k, const TexInfo info[4], bool alpha_test) {
    std::string s;
    auto put = [&](uint32_t v) { s.append(reinterpret_cast<const char*>(&v), 4); };
    put(k.programmable);
    put(alpha_test);
    for (int i = 0; i < 4; i++) {
        put(info[i].kind);
        put(info[i].rect);
        put(info[i].shadow);
    }
    put(k.alphakill);
    if (k.programmable) {
        put(k.count);
        for (int i = 0; i < 8; i++) {
            put(k.rgb_in[i]);
            put(k.rgb_out[i]);
            put(k.a_in[i]);
            put(k.a_out[i]);
        }
        put(k.final_abcd);
        put(k.final_efg);
        put(k.tex_modes);
        put(k.input_texture);
        put(k.dot_mapping);
        put(k.compare_mode);
    } else {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 9; j++) put(k.ff[i][j]);
        put(k.proj);
    }
    return s;
}

}  // namespace glshader
