// the host side on SDL3 (Android, and Linux for testing): paths, settings defaults, window loop
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#ifdef __ANDROID__
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <android/hardware_buffer.h>
#include <unistd.h>
#include <android/choreographer.h>
#include <android/looper.h>
#include <android/native_window.h>
#include <dlfcn.h>
#endif
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include "app/app.h"
#include "core/input.h"
#include "gpu/backend.h"
#include "frame.h"
#include "touch_pad.h"
#include "power.h"
#include "image_symbols.h"
#include "kernel/kernel.h"

#include "ref/reference_renderer.h"

namespace app {

Settings& Get() { static Settings s; return s; }
void Load() {}
void Save() {}
void ApplyDisplay() {}
Mode Desktop() { return {1280, 720}; }
Mode Output() { return Desktop(); }
float RenderScale(uint32_t) { return 1.0f; }
bool Widescreen() { return true; }

std::filesystem::path ExeDir() {
    static const std::filesystem::path dir = [] {
#ifdef __ANDROID__
        if (const char* p = SDL_GetAndroidInternalStoragePath()) return std::filesystem::path(p);
#endif
        return std::filesystem::current_path();
    }();
    return dir;
}
int Instance() { return 0; }
void* Window() { return nullptr; }
bool OpenSplash() { return false; }
void SplashFirstFrame() {}
void CloseSplash() {}
void OpenGameWindow(bool) {}
bool StartHandover() { return false; }
bool HandingOver() { return false; }
void FinishHandover() {}
std::recursive_mutex& UiLock() { static std::recursive_mutex m; return m; }

}  // namespace app
extern int xbox_vblank_hz;
namespace app {
// the window's presents, one per refresh of the display (vsync): the game's clock follows them
namespace {
std::mutex g_refresh_lock;
std::condition_variable g_refresh_cv;
uint64_t g_refreshes = 0;
}  // namespace
void DisplayRefreshed() {
    {
        std::lock_guard lock(g_refresh_lock);
        g_refreshes++;
    }
    g_refresh_cv.notify_all();
}
#ifdef __ANDROID__
// the display's vsync from Android's Choreographer, on a thread of its own (the window's presents wait for the GPU's
// frame first, so a clock run off them waited for the game's own frame and fell to 80 fps)
void StartVsync() {
    static bool started = false;
    if (started) return;
    started = true;
    std::thread([] {
        using Post64 = void (*)(AChoreographer*, void (*)(int64_t, void*), void*);
        using Post = void (*)(AChoreographer*, void (*)(long, void*), void*);
        static auto post64 = reinterpret_cast<Post64>(dlsym(RTLD_DEFAULT, "AChoreographer_postFrameCallback64"));
        static auto post = reinterpret_cast<Post>(dlsym(RTLD_DEFAULT, "AChoreographer_postFrameCallback"));
        ALooper_prepare(0);
        static AChoreographer* ch = AChoreographer_getInstance();
        if (!ch || (!post64 && !post)) return;
        struct Cb {
            static void on64(int64_t, void*) {
                DisplayRefreshed();
                post64(ch, on64, nullptr);
            }
            static void on(long, void*) {
                DisplayRefreshed();
                post(ch, on, nullptr);
            }
        };
        if (post64) post64(ch, Cb::on64, nullptr);
        else post(ch, Cb::on, nullptr);
        for (;;) ALooper_pollOnce(-1, nullptr, nullptr, nullptr);
    }).detach();
}
#endif
bool WaitRefresh(uint64_t& seen, int64_t timeout_us) {
    std::unique_lock lock(g_refresh_lock);
    if (!g_refresh_cv.wait_for(lock, std::chrono::microseconds(timeout_us), [&] { return g_refreshes != seen; })) return false;
    seen = g_refreshes;
    return true;
}
// at most one frame per refresh of the display (120 on a 120 Hz panel). The frame clock (d3d8 VerticalBlank) runs off
// the same refreshes, two blanks each, so a title that waits for its blanks is already in step and this returns at
// once; a timer of its own here, out of phase with the blanks, cost a frame now and then (95-105 fps, not 120)
void Pace() {
    using clock = std::chrono::steady_clock;
    static clock::time_point next = clock::now();
    // the game thread's work this frame (from the end of the last pace to now) for Android's performance hints
    static clock::time_point worked_from{};
    static bool registered = false;
    if (!registered) {
        registered = true;
        power::RegisterThread(power::kGame);
    }
    if (worked_from.time_since_epoch().count())
        power::GameFrame(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - worked_from).count());
    const int64_t period_us = 1000000 / std::clamp(xbox_vblank_hz / 2, 30, 240);
    static uint64_t paced = 0;
    if (WaitRefresh(paced, 2 * period_us)) {
        next = clock::now();
        worked_from = next;
        return;
    }
    next += std::chrono::microseconds(period_us);
    auto now = clock::now();
    if (next > now) std::this_thread::sleep_until(next);
    else next = now;
    worked_from = clock::now();
}
}  // namespace app

extern int xbox_vblank_hz;

namespace {
std::string GameDir(int argc, char** argv) {
    // the app hands its folder for the game files as the first argument (MainActivity: Android/data/<package>/files/game)
    if (argc > 1) return argv[1];
#ifdef __ANDROID__
    return "/storage/emulated/0/Android/data/com.jall.ea.nfsmw/files/game";
#else
    return "./game";
#endif
}

// The saves (the console's hard disk): beside the game's files in the app's own folder on the phone's storage
// (Android/data/<package>/files/savedata), where a player can see and back them up. Saves an earlier version kept in
// the app's internal storage are moved there once.
std::filesystem::path SaveDir(const std::string& game) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(game).parent_path() / "savedata";
    std::error_code ec;
    const fs::path old = app::ExeDir() / "hdd";
    if (fs::is_directory(old, ec) && (!fs::exists(dir, ec) || fs::is_empty(dir, ec))) {
        fs::create_directories(dir, ec);
        fs::copy(old, dir, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
        if (!ec) fs::remove_all(old, ec);
    }
    fs::create_directories(dir, ec);
    return dir;
}

void Message(SDL_Window* win, SDL_Renderer* r, const std::string& l1, const std::string& l2) {
    bool run = true;
    while (run) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) if (e.type == SDL_EVENT_QUIT) run = false;
        SDL_SetRenderDrawColor(r, 90, 0, 0, 255);
        SDL_RenderClear(r);
        SDL_SetRenderScale(r, 2.0f, 2.0f);
        SDL_SetRenderDrawColor(r, 255, 255, 255, 255);
        SDL_RenderDebugText(r, 20, 40, l1.c_str());
        SDL_RenderDebugText(r, 20, 70, l2.c_str());
        SDL_SetRenderScale(r, 1.0f, 1.0f);
        SDL_RenderPresent(r);
        SDL_Delay(100);
    }
    (void)win;
}
}  // namespace

namespace {
// fixed picture settings: the GPU draws at the screen's own resolution, the game runs at 120 frames
// a second (the panel's highest rate where that is lower)
constexpr int kFrameHz = 120;
constexpr float kResHeight = 0.f;  // lines drawn: 0 for the screen's own
int g_panel_hz = 60;
// NFS Most Wanted waits two blanks of the frame clock per frame (30 fps on the console's 60 Hz): the clock runs at
// twice the display's rate, so a frame can come every refresh
int FrameHz() { return 2 * std::min(kFrameHz, std::max(g_panel_hz, 30)); }
// the render scale for a picture of pw x ph window pixels: the screen's own pixels (or kResHeight lines at most)
void SetRenderScale(float pw, float ph) {
    const float share = ph > 0 && kResHeight > 0 ? std::min(kResHeight / ph, 1.f) : 1.f;
    platform::RenderScaleX() = std::max(pw * share / 640.f, 0.5f);
    platform::RenderScaleY() = std::max(ph * share / 480.f, 0.5f);
}
// The refresh rate the window really gets, from the presents (vsync): the shortest time between two in a second,
// snapped to the panel rates. The game's frame clock follows it both ways (the system lowers the rate when the phone is
// hot or saving power, and raises it again)
void TrackRefresh() {
    static Uint64 last = 0, window_start = 0, shortest = ~0ull;
    const Uint64 now = SDL_GetTicksNS();
    if (last) shortest = std::min(shortest, now - last);
    last = now;
    if (!window_start) window_start = now;
    if (now - window_start < 1000000000ull) return;
    window_start = now;
    if (shortest > 0 && shortest < 50000000ull) {
        const double hz = 1e9 / double(shortest);
        static const int kRates[] = {60, 90, 120, 144};
        int best = 60;
        for (int r : kRates)
            if (hz >= r * 0.93) best = r;
        if (best != g_panel_hz) {
            g_panel_hz = best;
        }
    }
    shortest = ~0ull;
}
// the highest refresh rate the panel offers for this window, and asked of Android for the
// window (it keeps apps at 60 Hz unless asked). Right after start the mode list can still say 60 only and the native
// window is not there yet, so the window loop asks again for its first seconds
// the refresh rate asked of Android for the window: the panel's highest (it keeps apps at 60 Hz unless asked)
void AskRate(SDL_Window* win) {
#ifdef __ANDROID__
    static float asked = 0;
    static void* asked_window = nullptr;  // a window made again (the app back from the background) is asked again
    using SetFrameRate = int32_t (*)(ANativeWindow*, float, int8_t);
    static auto set_rate = reinterpret_cast<SetFrameRate>(dlsym(RTLD_DEFAULT, "ANativeWindow_setFrameRate"));
    auto* aw = static_cast<ANativeWindow*>(
        SDL_GetPointerProperty(SDL_GetWindowProperties(win), SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr));
    // always the rate the game is made for: asking for the rate seen (60 Hz when the phone started the app at 60)
    // kept the window at 60 Hz for good
    const float want = float(kFrameHz);
    if (set_rate && aw && (want != asked || aw != asked_window)) {
        asked = want;
        asked_window = aw;
    }
#else
    (void)win;
#endif
}
void ProbePanel(SDL_Window* win) {
    float best = 0;
    if (SDL_DisplayID d = SDL_GetDisplayForWindow(win)) {
        int n = 0;
        if (SDL_DisplayMode** modes = SDL_GetFullscreenDisplayModes(d, &n)) {
            for (int i = 0; i < n; i++) best = std::max(best, modes[i]->refresh_rate);
            SDL_free(modes);
        }
        if (const SDL_DisplayMode* m = SDL_GetCurrentDisplayMode(d)) best = std::max(best, m->refresh_rate);
    }
    const int hz = std::max(int(best + 0.5f), 60);
    // only a first guess (the presents tell the rate from then on)
    static bool guessed = false;
    if (!guessed && hz > g_panel_hz) {
        guessed = true;
        g_panel_hz = hz;
    }
    AskRate(win);
}
}  // namespace

int main(int argc, char** argv) {
    SDL_SetHint(SDL_HINT_ORIENTATIONS, "LandscapeLeft LandscapeRight");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMEPAD)) return 1;
    // errors and crash reports beside the game folder (Android/data/<package>/files/logs), where a player can reach them
    SDL_Window* win = SDL_CreateWindow("NFS Most Wanted", 1280, 720, SDL_WINDOW_FULLSCREEN | SDL_WINDOW_RESIZABLE);
#ifdef __ANDROID__
    SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengles2");  // GL ES, so the renderer's frames can be shared with it
#endif
    SDL_Renderer* r = win ? SDL_CreateRenderer(win, nullptr) : nullptr;
    if (!r) return 1;
    SDL_SetRenderVSync(r, 1);
#ifdef __ANDROID__
    // the window's GL context, for the renderer's to share: its frames then reach the screen as textures
    // the renderer's frames come as images on the GPU when the window draws with GL ES and EGL has what that takes
    auto p_client_buffer = reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(eglGetProcAddress("eglGetNativeClientBufferANDROID"));
    auto p_create_image = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(eglGetProcAddress("eglCreateImageKHR"));
    auto p_destroy_image = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(eglGetProcAddress("eglDestroyImageKHR"));
    auto p_image_target = reinterpret_cast<PFNGLEGLIMAGETARGETTEXTURE2DOESPROC>(eglGetProcAddress("glEGLImageTargetTexture2DOES"));
    auto p_create_sync = reinterpret_cast<PFNEGLCREATESYNCKHRPROC>(eglGetProcAddress("eglCreateSyncKHR"));
    auto p_wait_sync = reinterpret_cast<PFNEGLWAITSYNCKHRPROC>(eglGetProcAddress("eglWaitSyncKHR"));
    auto p_destroy_sync = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(eglGetProcAddress("eglDestroySyncKHR"));
    auto p_dup_fd = reinterpret_cast<PFNEGLDUPNATIVEFENCEFDANDROIDPROC>(eglGetProcAddress("eglDupNativeFenceFDANDROID"));
    EGLDisplay egl_dpy = eglGetCurrentDisplay();
    if (egl_dpy == EGL_NO_DISPLAY) egl_dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    // Android native fence files: a GPU wait for one (taken over), one for what this context sent so far
    auto wait_fd = [&](int fd) {
        if (fd < 0) return;
        const EGLint attr[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, fd, EGL_NONE};
        EGLSyncKHR sync = p_create_sync(egl_dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, attr);
        if (sync == EGL_NO_SYNC_KHR) { close(fd); return; }
        p_wait_sync(egl_dpy, sync, 0);
        p_destroy_sync(egl_dpy, sync);
    };
    auto fence_fd = [&]() -> int {
        const EGLint attr[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, EGL_NO_NATIVE_FENCE_FD_ANDROID, EGL_NONE};
        EGLSyncKHR sync = p_create_sync(egl_dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, attr);
        if (sync == EGL_NO_SYNC_KHR) return -1;
        glFlush();
        const int fd = p_dup_fd(egl_dpy, sync);
        p_destroy_sync(egl_dpy, sync);
        return fd;
    };
    {
        const char* name = SDL_GetRendererName(r);
        const bool ok = name && std::string(name) == "opengles2" && p_client_buffer && p_create_image && p_destroy_image &&
                        p_image_target && p_create_sync && p_wait_sync && p_destroy_sync && p_dup_fd;
        platform::GetPresentChain().window_ready = ok;
    }
#endif

    std::string game = GameDir(argc, argv);
    // the renderer keeps its compiled shader programs in the app's own folder (<files>/glcache)
    reference::g_dump_dir = (app::ExeDir() / "shots").string();
    std::error_code ec;
    std::filesystem::create_directories(game, ec);
    std::string xbe_path = game + "/" + IMAGE_XBE;
    std::ifstream f(xbe_path, std::ios::binary);
    if (!f) {
        Message(win, r, "Game files not found.", "Import the game's ISO from the app's first screen.");
        return 1;
    }
    std::vector<uint8_t> xbe((std::istreambuf_iterator<char>(f)), {});

    // the title's frame clock: 120 Hz (the panel's highest rate where that is lower)
    ProbePanel(win);
    xbox_vblank_hz = FrameHz();
    // the GPU draws the game at the screen's resolution, the game's camera at the screen's own shape
    {
        int ww = 0, wh = 0;
        SDL_GetWindowSizeInPixels(win, &ww, &wh);
        SetRenderScale(float(std::max(ww, wh)), float(std::min(ww, wh)));
        if (ww > 0 && wh > 0) platform::ScreenAspect() = float(std::min(ww, wh)) / float(std::max(ww, wh));
    }
    app::Load();
    input::Init();
    if (app::kGame.init) app::kGame.init();
    if (app::kGame.window) app::kGame.window(nullptr);
#ifdef __ANDROID__
    app::StartVsync();  // the game's frame clock follows the display's vsync
#endif
    if (!kernel::Boot(std::move(xbe), IMAGE_XBE, game, SaveDir(game).string())) {
        Message(win, r, "The game failed to start.", "Import the game's ISO again from the app's first screen.");
        return 1;
    }

    SDL_Texture* tex = nullptr;
    int tw = 0, th = 0;
    constexpr int kBuffers = platform::PresentChain::kBuffers;
    SDL_Texture* ptex[kBuffers] = {};  // the frames the renderer hands over on the GPU (platform::PresentChain)
#ifdef __ANDROID__
    AHardwareBuffer* pbuf[kBuffers] = {};
    EGLImageKHR pimg[kBuffers] = {};
#endif
    uint64_t seen = 0;
    bool run = true;
    while (run) {
        SDL_Event e;
        bool remake = false;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) run = false;
            // back from the background: Android may have taken the GL context away meanwhile (after a while out of the
            // app), and with it the textures the frames are shown through, which then showed black with the sound
            // still going. They are made again then, whether lost or not
            if (e.type == SDL_EVENT_RENDER_DEVICE_RESET || e.type == SDL_EVENT_RENDER_TARGETS_RESET ||
                e.type == SDL_EVENT_DID_ENTER_FOREGROUND || e.type == SDL_EVENT_WINDOW_RESTORED)
                remake = true;
            touchpad::Event(e);
        }
        if (remake) {
            if (tex) SDL_DestroyTexture(tex);
            tex = nullptr;
            seen = 0;
#ifdef __ANDROID__
            auto& pc = platform::GetPresentChain();
            std::lock_guard lock(pc.lock);
            pc.w = pc.h = 0;  // the buffers and their textures are made again below
#endif
        }
        int ww, wh;
        SDL_GetRenderOutputSize(r, &ww, &wh);
        {
            auto& fr = platform::GetFrame();
            std::lock_guard lock(fr.lock);
            if (fr.serial != seen && fr.w) {
                if (!tex || tw != int(fr.w) || th != int(fr.h)) {
                    if (tex) SDL_DestroyTexture(tex);
                    tw = int(fr.w); th = int(fr.h);
                    tex = SDL_CreateTexture(r, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, tw, th);
                    SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_LINEAR);
                    SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_NONE);  // the frame's alpha is not opacity
                }
                SDL_UpdateTexture(tex, nullptr, fr.rgba.data(), tw * 4);
                seen = fr.serial;
            }
        }
        // the renderer's latest frame, handed over on the GPU: textures of its size, the copy waited for there
        int shown_k = -1;
#ifdef __ANDROID__
        {
            auto& pc = platform::GetPresentChain();
            int wait = -1;
            bool took = false;
            if (pc.active) {
                std::lock_guard lock(pc.lock);
                if (pc.want_w && (pc.w != pc.want_w || pc.h != pc.want_h)) {
                    // three hardware buffers of the frame's size: the renderer draws into them, SDL shows them
                    for (int i = 0; i < kBuffers; i++) {
                        for (int* f : {&pc.write_fd[i], &pc.read_fd[i]})
                            if (*f >= 0) { close(*f); *f = -1; }
                        if (ptex[i]) SDL_DestroyTexture(ptex[i]);
                        if (pimg[i]) p_destroy_image(egl_dpy, pimg[i]);
                        if (pbuf[i]) AHardwareBuffer_release(pbuf[i]);
                        ptex[i] = nullptr;
                        pimg[i] = nullptr;
                        pbuf[i] = nullptr;
                        pc.buffer[i] = nullptr;
                        AHardwareBuffer_Desc desc{};
                        desc.width = pc.want_w;
                        desc.height = pc.want_h;
                        desc.layers = 1;
                        desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
                        desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;
                        if (AHardwareBuffer_allocate(&desc, &pbuf[i]) != 0) { pbuf[i] = nullptr; break; }
                        const EGLint attr[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};
                        pimg[i] = p_create_image(egl_dpy, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID, p_client_buffer(pbuf[i]), attr);
                        if (pimg[i] == EGL_NO_IMAGE_KHR) { pimg[i] = nullptr; break; }
                        ptex[i] = SDL_CreateTexture(r, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, int(pc.want_w), int(pc.want_h));
                        SDL_SetTextureScaleMode(ptex[i], SDL_SCALEMODE_LINEAR);
                        SDL_SetTextureBlendMode(ptex[i], SDL_BLENDMODE_NONE);
                        const GLuint id = GLuint(SDL_GetNumberProperty(SDL_GetTextureProperties(ptex[i]), SDL_PROP_TEXTURE_OPENGLES2_TEXTURE_NUMBER, 0));
                        glActiveTexture(GL_TEXTURE0);
                        glBindTexture(GL_TEXTURE_2D, id);
                        p_image_target(GL_TEXTURE_2D, GLeglImageOES(pimg[i]));
                        pc.buffer[i] = pbuf[i];
                    }
                    bool ok = true;
                    for (int i = 0; i < kBuffers; i++) ok = ok && pc.buffer[i];
                    pc.w = ok ? pc.want_w : 0;
                    pc.h = ok ? pc.want_h : 0;
                    pc.latest = pc.in_use = -1;
                    pc.queued = 0;
                    if (!ok) {
                        pc.active = false;  // the renderer goes back to the memory way
                    } else {
                    }
                }
                if (const int next = pc.Pop(); next >= 0) {
                    wait = pc.write_fd[next];
                    pc.write_fd[next] = -1;
                    pc.in_use = next;
                    took = true;
                }
                shown_k = pc.w ? pc.in_use : -1;
            }
            if (took) pc.taken.notify_all();
            wait_fd(wait);
        }
#endif
        // the whole screen, the game's camera at the screen's shape (src/game/hooks.cpp)
        const bool crop = false;
        const float shape = 0.f;
        if (ww > 0 && wh > 0) platform::ScreenAspect() = float(std::min(ww, wh)) / float(std::max(ww, wh));
        {
            static Uint64 last_probe = 0;
            if (SDL_GetTicks() - last_probe > 1000) {
                last_probe = SDL_GetTicks();
                if (SDL_GetTicks() < 30000) ProbePanel(win);
                else AskRate(win);
            }
        }
        power::Update(FrameHz() / 2);
        xbox_vblank_hz = FrameHz();
        // the render scale for the picture's size on the screen (the window can still be turning when the game starts)
        if (ww > 0 && wh > 0) SetRenderScale(float(std::max(ww, wh)), float(std::min(ww, wh)));
        SDL_Texture* show = shown_k >= 0 ? ptex[shown_k] : tex;
        if (shown_k >= 0) {
            tw = int(platform::GetPresentChain().w);
            th = int(platform::GetPresentChain().h);
        }
        // the picture covers the whole window: a clear first would only be drawn over (a full screen fill each frame)
        if (!show || crop || shape > 0) {
            SDL_SetRenderDrawColor(r, 0, 0, 0, 255);
            SDL_RenderClear(r);
        }
        if (show) {
            SDL_FRect src{0, 0, float(tw), float(th)};
            float dw = float(ww), dh = float(wh);
            if (crop && ww > 0 && wh > 0) {
                const float screen = float(ww) / float(wh);
                if (screen > shape) { src.h = th * shape / screen; src.y = (th - src.h) / 2; }
                else { src.w = tw * screen / shape; src.x = (tw - src.w) / 2; }
            } else if (shape > 0) {
                dw = dh * shape;
                if (dw > ww) { dw = float(ww); dh = dw / shape; }
            }
            SDL_FRect dst{(ww - dw) / 2, (wh - dh) / 2, dw, dh};
            SDL_RenderTexture(r, show, &src, &dst);
            platform::PicturePlacement() = {src.x / tw, src.y / th, src.w / tw, src.h / th, dst.x, dst.y, dst.w, dst.h};
        }
        touchpad::Draw(r, ww, wh);  // the on-screen controller over the picture
        if (SDL_RenderPresent(r)) TrackRefresh();
#ifdef __ANDROID__
        // the shown texture is free again for the renderer once this draw of it is done
        if (shown_k >= 0) {
            auto& pc = platform::GetPresentChain();
            const int done = fence_fd();
            std::lock_guard lock(pc.lock);
            if (pc.read_fd[shown_k] >= 0) close(pc.read_fd[shown_k]);
            pc.read_fd[shown_k] = done;
        }
#endif
    }
    kernel::Shutdown(0);
}
