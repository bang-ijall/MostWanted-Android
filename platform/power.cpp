#include "power.h"

#include <algorithm>
#include <atomic>
#include <type_traits>

#ifdef __ANDROID__
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace power {
namespace {
std::atomic<int> g_tid[2] = {0, 0};
std::atomic<int64_t> g_work_ns{0};    // the game thread's last frame
std::atomic<int64_t> g_render_ns{0};  // the render thread's last frame
std::atomic<uint64_t> g_work_serial{0};

#ifdef __ANDROID__
// Android's performance hints (API 33), looked up at run time as the app starts from Android 8
struct Api {
    void* (*manager)() = nullptr;
    void* (*create)(void*, const int32_t*, size_t, int64_t) = nullptr;
    int (*update_target)(void*, int64_t) = nullptr;
    int (*report)(void*, int64_t) = nullptr;
};
const Api& Load() {
    static const Api a = [] {
        Api x;
        if (void* lib = dlopen("libandroid.so", RTLD_NOW)) {
            auto get = [&](auto& fn, const char* name) { fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name)); };
            get(x.manager, "APerformanceHint_getManager");
            get(x.create, "APerformanceHint_createSession");
            get(x.update_target, "APerformanceHint_updateTargetWorkDuration");
            get(x.report, "APerformanceHint_reportActualWorkDuration");
        }
        return x;
    }();
    return a;
}
#endif
}  // namespace

void RegisterThread(Thread which) {
#ifdef __ANDROID__
    g_tid[which] = int(gettid());
#else
    (void)which;
#endif
}

void GameFrame(int64_t work_ns) {
    g_work_ns = work_ns;
    g_work_serial++;
}

void RenderFrame(int64_t work_ns) { g_render_ns = work_ns; }

namespace {
// the work time aimed at: 80% of the frame, so a heavier frame than the last still makes the refresh (the system
// otherwise sets the clocks for the frame's whole time, and any bump missed it)
int64_t Target(int frame_hz) { return 800000000ll / frame_hz; }
}  // namespace

void Update(int frame_hz) {
#ifdef __ANDROID__
    // off: on this phone (Snapdragon) a hint session held the clocks of the game's and renderer's threads at a third
    // of their top (1.0-1.5 GHz) while the renderer was 90% busy, 70-80 fps; the scheduler's own governor raises them
    (void)frame_hz;
    if (constexpr bool kHints = false; !kHints) return;
    const Api& a = Load();
    if (!a.manager || !a.create || !a.update_target || !a.report) return;
    // one session for the game's and the renderer's threads once both are known: the system raises the clocks only as
    // far as the frames need, rather than to the top (cooler at the same frame rate)
    static void* session = nullptr;
    static bool tried = false;
    static int target_hz = 0;
    if (!session && !tried && g_tid[0] && g_tid[1] && frame_hz > 0) {
        tried = true;
        const int32_t tids[2] = {g_tid[0], g_tid[1]};
        if (void* m = a.manager()) session = a.create(m, tids, 2, Target(frame_hz));
        target_hz = frame_hz;
    }
    if (!session) return;
    if (frame_hz > 0 && frame_hz != target_hz) {
        target_hz = frame_hz;
        a.update_target(session, Target(frame_hz));
    }
    static uint64_t reported = 0;
    const uint64_t serial = g_work_serial.load();
    if (serial != reported) {
        reported = serial;
        // the busier of the two threads: the session's clocks serve both, and reporting the game thread alone (often
        // the lighter one in a race) had the system lower them under the render thread (1.2 of 3.2 GHz, 100-110 fps)
        if (const int64_t ns = std::max(g_work_ns.load(), g_render_ns.load()); ns > 0) a.report(session, ns);
    }
#else
    (void)frame_hz;
#endif
}

}  // namespace power
