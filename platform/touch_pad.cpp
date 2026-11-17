#include "touch_pad.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef __ANDROID__
#include <jni.h>
#endif

namespace touchpad {
namespace {

// what the fingers hold, read by the game's threads
struct State {
    uint16_t buttons = 0;     // XINPUT digital bits
    uint8_t analog[8] = {};   // A B X Y black white LT RT
    int16_t lx = 0, ly = 0, rx = 0, ry = 0;
};
std::mutex g_lock;
State g_state;

std::atomic<bool> g_physical{false};     // a controller is plugged in on port 1
int g_w = 0, g_h = 0;                    // window size, from the last draw

enum Kind { kButton, kTrigger, kStick, kDpad };
struct Control {
    Kind kind;
    const char* label;
    float x, y, r;        // centre and radius (round), or centre and half size (rect: r across, rh down)
    float rh;
    int bit;              // digital bit, analog index, or stick 0/1
    SDL_Color colour;
};

constexpr uint16_t kUp = 0x01, kDown = 0x02, kLeft = 0x04, kRight = 0x08, kStart = 0x10, kBack = 0x20;
constexpr int kA = 0, kB = 1, kX = 2, kY = 3, kBlack = 4, kWhite = 5, kLT = 6, kRT = 7;

// the layout for a w x h window (landscape), in pixels; u is the short side
std::vector<Control> Layout(int w, int h) {
    const float u = float(std::min(w, h)), W = float(std::max(w, h));
    const float side = 0.035f * W;
    const SDL_Color grey{230, 230, 230, 255};
    std::vector<Control> c;
    // triggers and bumpers in the top corners
    const float tw = 0.13f * u, th = 0.065f * u;
    c.push_back({kTrigger, "LT", side + tw, 0.08f * u, tw, th, kLT, grey});
    c.push_back({kButton, "LB", side + tw, 0.08f * u + 2.3f * th, tw, th * 0.8f, kWhite, grey});
    c.push_back({kTrigger, "RT", W - side - tw, 0.08f * u, tw, th, kRT, grey});
    c.push_back({kButton, "RB", W - side - tw, 0.08f * u + 2.3f * th, tw, th * 0.8f, kBlack, grey});
    // Back and Start between them
    c.push_back({kButton, "BACK", W / 2 - 0.12f * u, 0.09f * u, 0.045f * u, 0, -kBack, grey});
    c.push_back({kButton, "START", W / 2 + 0.12f * u, 0.09f * u, 0.045f * u, 0, -kStart, grey});
    // left stick high, d-pad below and inward (an Xbox pad's left half)
    c.push_back({kStick, "", side + 0.17f * u, 0.52f * u, 0.15f * u, 0, 0, grey});
    c.push_back({kDpad, "", side + 0.47f * u, 0.80f * u, 0.12f * u, 0, 0, grey});
    // A B X Y high on the right, right stick below and inward
    const float bx = W - side - 0.17f * u, by = 0.52f * u, d = 0.105f * u, br = 0.062f * u;
    c.push_back({kButton, "A", bx, by + d, br, 0, kA, {107, 190, 69, 255}});
    c.push_back({kButton, "B", bx + d, by, br, 0, kB, {224, 62, 54, 255}});
    c.push_back({kButton, "X", bx - d, by, br, 0, kX, {58, 132, 230, 255}});
    c.push_back({kButton, "Y", bx, by - d, br, 0, kY, {245, 196, 40, 255}});
    c.push_back({kStick, "", W - side - 0.47f * u, 0.80f * u, 0.11f * u, 0, 1, grey});
    return c;
}

bool Rect(const Control& c) { return c.rh > 0; }

// how near a point is to a control: 0 inside, larger further out (a finger a little off still counts)
float Distance(const Control& c, float x, float y) {
    if (Rect(c)) {
        const float dx = std::max(std::fabs(x - c.x) - c.r, 0.f), dy = std::max(std::fabs(y - c.y) - c.rh, 0.f);
        return std::hypot(dx, dy) / c.rh;
    }
    const float reach = c.kind == kStick ? 1.6f : c.kind == kDpad ? 1.3f : 1.0f;
    return std::max(std::hypot(x - c.x, y - c.y) / (c.r * reach) - 1.f, 0.f) * 2.f;
}

struct Finger {
    int control = -1;   // index in the layout
    float x = 0, y = 0;
};
std::map<SDL_FingerID, Finger> g_fingers;

int Hit(const std::vector<Control>& lay, float x, float y) {
    int best = -1;
    float bd = 0.35f;
    for (size_t i = 0; i < lay.size(); i++) {
        const float d = Distance(lay[i], x, y);
        if (d < bd) { bd = d; best = int(i); }
    }
    return best;
}

// the fingers into the pad state
void Rebuild(const std::vector<Control>& lay) {
    State s;
    for (const auto& [id, f] : g_fingers) {
        if (f.control < 0) continue;
        const Control& c = lay[size_t(f.control)];
        switch (c.kind) {
        case kButton:
            if (c.bit < 0) s.buttons |= uint16_t(-c.bit);
            else s.analog[c.bit] = 255;
            break;
        case kTrigger:
            s.analog[c.bit] = 255;
            break;
        case kStick: {
            float dx = (f.x - c.x) / c.r, dy = (f.y - c.y) / c.r;
            const float len = std::hypot(dx, dy);
            if (len > 1) { dx /= len; dy /= len; }
            const int16_t sx = int16_t(std::lround(dx * 32767)), sy = int16_t(std::lround(-dy * 32767));
            if (c.bit == 0) { s.lx = sx; s.ly = sy; }
            else { s.rx = sx; s.ry = sy; }
            break;
        }
        case kDpad: {
            const float dx = f.x - c.x, dy = f.y - c.y;
            if (std::hypot(dx, dy) < c.r * 0.2f) break;
            // eight ways: a diagonal holds both
            const float a = std::atan2(-dy, dx);
            const float pi = 3.14159265f;
            auto near = [&](float centre) { float d = std::fabs(std::remainder(a - centre, 2 * pi)); return d < pi * 3 / 8; };
            if (near(0)) s.buttons |= kRight;
            if (near(pi / 2)) s.buttons |= kUp;
            if (near(pi)) s.buttons |= kLeft;
            if (near(-pi / 2)) s.buttons |= kDown;
            break;
        }
        }
    }
    std::lock_guard lock(g_lock);
    g_state = s;
}

// shown only while no controller is plugged in
bool Visible() { return !g_physical.load(); }

// the overlay's shapes go into one batch, drawn with one call; text and rects flush it first to keep the order
std::vector<SDL_Vertex> g_batch;
void Flush(SDL_Renderer* r) {
    if (g_batch.empty()) return;
    SDL_RenderGeometry(r, nullptr, g_batch.data(), int(g_batch.size()), nullptr, 0);
    g_batch.clear();
}

// the unit circle in 40 steps, worked out once
constexpr int kSteps = 40;
const float* UnitCircle() {
    static float t[2 * (kSteps + 1)];
    static bool made = false;
    if (!made) {
        made = true;
        for (int i = 0; i <= kSteps; i++) {
            const float a = 2 * 3.14159265f * float(i) / kSteps;
            t[2 * i] = std::cos(a);
            t[2 * i + 1] = std::sin(a);
        }
    }
    return t;
}

// a filled circle (or ring) as a triangle fan
void Circle(SDL_Renderer*, float x, float y, float rad, SDL_Color c, float inner = 0) {
    const float* u = UnitCircle();
    const SDL_FColor fc{c.r / 255.f, c.g / 255.f, c.b / 255.f, c.a / 255.f};
    for (int i = 0; i < kSteps; i++) {
        const float c0 = u[2 * i], s0 = u[2 * i + 1], c1 = u[2 * i + 2], s1 = u[2 * i + 3];
        const SDL_FPoint o0{x + c0 * rad, y + s0 * rad}, o1{x + c1 * rad, y + s1 * rad};
        if (inner > 0) {
            const SDL_FPoint i0{x + c0 * inner, y + s0 * inner}, i1{x + c1 * inner, y + s1 * inner};
            g_batch.insert(g_batch.end(), {{o0, fc, {}}, {o1, fc, {}}, {i0, fc, {}}, {i0, fc, {}}, {o1, fc, {}}, {i1, fc, {}}});
        } else {
            g_batch.insert(g_batch.end(), {{{x, y}, fc, {}}, {o0, fc, {}}, {o1, fc, {}}});
        }
    }
}

// a filled outline that every ray from (x, y) crosses once, as one triangle fan (no overlap: an even alpha)
void Fan(SDL_Renderer*, float x, float y, const std::vector<SDL_FPoint>& rim, SDL_Color c) {
    const SDL_FColor fc{c.r / 255.f, c.g / 255.f, c.b / 255.f, c.a / 255.f};
    for (size_t i = 0; i < rim.size(); i++)
        g_batch.insert(g_batch.end(), {{{x, y}, fc, {}}, {rim[i], fc, {}}, {rim[(i + 1) % rim.size()], fc, {}}});
}

void RoundRect(SDL_Renderer* r, float x, float y, float hw, float hh, SDL_Color c) {
    const float rad = hh, pi = 3.14159265f;
    std::vector<SDL_FPoint> rim;
    for (int i = 0; i <= 16; i++) {  // right end, top to bottom
        const float a = -pi / 2 + pi * float(i) / 16;
        rim.push_back({x + hw - rad + std::cos(a) * rad, y + std::sin(a) * rad});
    }
    for (int i = 0; i <= 16; i++) {  // left end, bottom to top
        const float a = pi / 2 + pi * float(i) / 16;
        rim.push_back({x - hw + rad + std::cos(a) * rad, y + std::sin(a) * rad});
    }
    Fan(r, x, y, rim, c);
}

// a plus sign: arms of half width a reaching out to l
void Plus(SDL_Renderer* r, float x, float y, float a, float l, SDL_Color c) {
    Fan(r, x, y, {{x - a, y - l}, {x + a, y - l}, {x + a, y - a}, {x + l, y - a}, {x + l, y + a}, {x + a, y + a},
                  {x + a, y + l}, {x - a, y + l}, {x - a, y + a}, {x - l, y + a}, {x - l, y - a}, {x - a, y - a}}, c);
}

// the labels in the system's font, each a texture made on the window thread when first drawn
struct Label {
    int w = 0, h = 0;
    std::vector<uint8_t> alpha;
    SDL_Texture* tex = nullptr;
};
std::mutex g_label_lock;
std::map<std::string, Label> g_labels;

void Text(SDL_Renderer* r, float x, float y, const char* s, float size, SDL_Color c) {
    if (!*s) return;
    Flush(r);
    {
        std::lock_guard lock(g_label_lock);
        auto it = g_labels.find(s);
        if (it != g_labels.end() && it->second.w > 0) {
            Label& l = it->second;
            if (!l.tex) {
                std::vector<uint32_t> px(size_t(l.w) * l.h);
                for (size_t i = 0; i < px.size(); i++) px[i] = 0x00FFFFFFu | uint32_t(l.alpha[i]) << 24;
                l.tex = SDL_CreateTexture(r, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, l.w, l.h);
                if (l.tex) {
                    SDL_UpdateTexture(l.tex, nullptr, px.data(), l.w * 4);
                    SDL_SetTextureBlendMode(l.tex, SDL_BLENDMODE_BLEND);
                    SDL_SetTextureScaleMode(l.tex, SDL_SCALEMODE_LINEAR);
                }
            }
            if (l.tex) {
                // size is the height of a capital letter
                const float h = size * 0.85f, w = h * float(l.w) / float(l.h);
                const SDL_FRect dst{x - w / 2, y - h / 2, w, h};
                SDL_SetTextureColorMod(l.tex, c.r, c.g, c.b);
                SDL_SetTextureAlphaMod(l.tex, c.a);
                SDL_RenderTexture(r, l.tex, nullptr, &dst);
                return;
            }
        }
    }
    const float k = size / 8.f;
    SDL_SetRenderScale(r, k, k);
    SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a);
    SDL_RenderDebugText(r, (x - 4.f * k * float(SDL_strlen(s))) / k, (y - 4.f * k) / k, s);
    SDL_SetRenderScale(r, 1, 1);
}

SDL_Color Alpha(SDL_Color c, uint8_t a) { c.a = a; return c; }

}  // namespace

void SetLabel(const char* text, int w, int h, const uint8_t* alpha) {
    if (!text || w <= 0 || h <= 0 || !alpha) return;
    std::lock_guard lock(g_label_lock);
    Label& l = g_labels[text];
    l.w = w;
    l.h = h;
    l.alpha.assign(alpha, alpha + size_t(w) * h);
    l.tex = nullptr;  // made again from these on the next draw (an old one is left to the renderer)
}

void Event(const SDL_Event& e) {
    // the renderer lost its textures (back from the background): the labels are made again when drawn
    if (e.type == SDL_EVENT_RENDER_DEVICE_RESET) {
        std::lock_guard lock(g_label_lock);
        for (auto& [k, l] : g_labels) l.tex = nullptr;
    }
    if (e.type != SDL_EVENT_FINGER_DOWN && e.type != SDL_EVENT_FINGER_MOTION && e.type != SDL_EVENT_FINGER_UP &&
        e.type != SDL_EVENT_FINGER_CANCELED)
        return;
    if (g_w <= 0 || g_h <= 0) return;
    const auto lay = Layout(g_w, g_h);
    const float x = e.tfinger.x * float(g_w), y = e.tfinger.y * float(g_h);
    const SDL_FingerID id = e.tfinger.fingerID;
    if (e.type == SDL_EVENT_FINGER_UP || e.type == SDL_EVENT_FINGER_CANCELED) {
        g_fingers.erase(id);
    } else if (e.type == SDL_EVENT_FINGER_DOWN) {
        g_fingers[id] = {Hit(lay, x, y), x, y};
    } else {
        Finger& f = g_fingers[id];
        f.x = x;
        f.y = y;
        // a finger on a stick keeps it; one on buttons slides from button to button
        if (f.control < 0 || lay[size_t(f.control)].kind != kStick) f.control = Hit(lay, x, y);
    }
    Rebuild(lay);
}

void Draw(SDL_Renderer* r, int w, int h) {
    g_w = w;
    g_h = h;
    if (!Visible() || w <= 0 || h <= 0) return;
    const auto lay = Layout(w, h);
    State s;
    {
        std::lock_guard lock(g_lock);
        s = g_state;
    }
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    const float u = float(std::min(w, h));
    const SDL_Color base{20, 20, 20, 110}, edge{255, 255, 255, 70};
    for (const Control& c : lay) {
        switch (c.kind) {
        case kButton:
        case kTrigger: {
            const bool down = c.bit < 0 ? (s.buttons & uint16_t(-c.bit)) != 0 : s.analog[c.bit] != 0;
            if (Rect(c)) {
                RoundRect(r, c.x, c.y, c.r, c.rh, down ? SDL_Color{255, 255, 255, 120} : base);
                Text(r, c.x, c.y, c.label, 0.035f * u, Alpha(c.colour, 230));
            } else {
                Circle(r, c.x, c.y, c.r + 2, edge, c.r);  // a ring, not under the fill
                Circle(r, c.x, c.y, c.r, down ? Alpha(c.colour, 200) : base);
                if (c.kind == kButton && c.bit >= 0) Circle(r, c.x, c.y, c.r, Alpha(c.colour, 170), c.r * 0.86f);
                const bool small = SDL_strlen(c.label) > 1;
                Text(r, c.x, c.y, c.label, (small ? 0.021f : 0.05f) * u, down ? SDL_Color{255, 255, 255, 255} : Alpha(c.colour, 235));
            }
            break;
        }
        case kStick: {
            Circle(r, c.x, c.y, c.r, base);
            Circle(r, c.x, c.y, c.r, edge, c.r - 2);
            const float sx = (c.bit == 0 ? s.lx : s.rx) / 32767.f, sy = (c.bit == 0 ? s.ly : s.ry) / 32767.f;
            const float kx = c.x + sx * c.r * 0.6f, ky = c.y - sy * c.r * 0.6f;
            Circle(r, kx, ky, c.r * 0.45f, {60, 60, 60, 170});
            Circle(r, kx, ky, c.r * 0.45f, {255, 255, 255, 110}, c.r * 0.45f - 3);
            break;
        }
        case kDpad: {
            const float a = c.r * 0.34f, l = c.r;
            Plus(r, c.x, c.y, a, l, base);
            struct { uint16_t bit; float dx, dy; } arms[4] = {{kUp, 0, -1}, {kDown, 0, 1}, {kLeft, -1, 0}, {kRight, 1, 0}};
            for (auto& m : arms) {
                const float cx = c.x + m.dx * c.r * 0.62f, cy = c.y + m.dy * c.r * 0.62f;
                if (s.buttons & m.bit) {
                    Flush(r);
                    SDL_SetRenderDrawColor(r, 255, 255, 255, 120);
                    const SDL_FRect box{cx - a + 1, cy - a + 1, 2 * a - 2, 2 * a - 2};
                    SDL_RenderFillRect(r, &box);
                }
                // an arrow pointing out along the arm
                const float t = 0.42f * a, px = -m.dy, py = m.dx;
                Fan(r, cx, cy, {{cx + m.dx * t * 1.1f, cy + m.dy * t * 1.1f}, {cx - m.dx * t * 0.7f + px * t, cy - m.dy * t * 0.7f + py * t},
                                {cx - m.dx * t * 0.7f - px * t, cy - m.dy * t * 0.7f - py * t}}, {235, 235, 235, 230});
            }
            break;
        }
        }
    }
    Flush(r);
}

void Merge(int port, input::Pad& p) {
    if (port != 0) return;
    g_physical = p.connected;
    State s;
    {
        std::lock_guard lock(g_lock);
        s = g_state;
    }
    if (!Visible()) return;
    p.connected = true;
    p.buttons |= s.buttons;
    for (int i = 0; i < 8; i++) p.analog[i] = std::max(p.analog[i], s.analog[i]);
    auto pick = [](int16_t a, int16_t b) { return std::abs(b) > std::abs(a) ? b : a; };
    p.lx = pick(p.lx, s.lx);
    p.ly = pick(p.ly, s.ly);
    p.rx = pick(p.rx, s.rx);
    p.ry = pick(p.ry, s.ry);
}

}  // namespace touchpad

#ifdef __ANDROID__
// MainActivity.nativeLabel(text, w, h, alpha): one label drawn by Android's text renderer
extern "C" JNIEXPORT void JNICALL Java_com_jall_ea_nfsmw_MainActivity_nativeLabel(JNIEnv* env, jclass, jstring text, jint w, jint h,
                                                                                 jbyteArray alpha) {
    if (!text || !alpha || env->GetArrayLength(alpha) < w * h) return;
    const char* t = env->GetStringUTFChars(text, nullptr);
    jbyte* a = env->GetByteArrayElements(alpha, nullptr);
    touchpad::SetLabel(t, w, h, reinterpret_cast<const uint8_t*>(a));
    env->ReleaseByteArrayElements(alpha, a, JNI_ABORT);
    env->ReleaseStringUTFChars(text, t);
}
#endif
