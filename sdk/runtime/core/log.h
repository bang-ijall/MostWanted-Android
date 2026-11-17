#pragma once
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <string>

namespace logx {

// a release build writes Report and Error lines only
enum class Level { Debug, Info, Warn, Error, Report };

// release: dir/prefix.log, carried on when append; debug: dir/prefix_NNN.log per run, newest 30 kept
void Init(const std::string& dir, const std::string& prefix, bool append = false);
void Write(Level lv, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
void Flush();
const std::string& Path();

// release builds write errors and reports only (no info lines formatted or written while the game runs)
#if defined(NDEBUG) && !defined(KEEP_LOGS)
inline constexpr bool kVerbose = false;
#else
inline constexpr bool kVerbose = true;
#endif

}  // namespace logx

#if defined(NDEBUG) && !defined(KEEP_LOGS) && defined(__ANDROID__)
// the Android release writes no log at all
#define LOGE(...) ((void)0)
#else
#define LOGE(...) ::logx::Write(::logx::Level::Error, __VA_ARGS__)
#endif
#if defined(NDEBUG) && !defined(KEEP_LOGS)
// a release build keeps only errors: the rest compile to nothing (no call, no formatting, no writing)
#define LOGD(...) ((void)0)
#define LOGI(...) ((void)0)
#define LOGW(...) ((void)0)
#define LOGR(...) ((void)0)
#define LOG_FIRST(n, ...) ((void)0)
#else
#define LOGD(...) ::logx::Write(::logx::Level::Debug, __VA_ARGS__)
#define LOGI(...) ::logx::Write(::logx::Level::Info, __VA_ARGS__)
#define LOGW(...) ::logx::Write(::logx::Level::Warn, __VA_ARGS__)
#define LOGR(...) ::logx::Write(::logx::Level::Report, __VA_ARGS__)
// logs only the first n times a call site is reached
#define LOG_FIRST(n, ...)                                  \
    do {                                                   \
        static std::atomic<int> _c{0};                     \
        if (_c.fetch_add(1, std::memory_order_relaxed) < (n)) LOGI(__VA_ARGS__); \
    } while (0)
#endif
