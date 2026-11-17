// The Direct3D 12 backend on a thread of its own. The title's render thread records each call with what the console
// takes at call time (state changes, resource headers, indices, rects) and goes on; vertex, texture and surface memory
// are read when the draw runs, as the console's GPU reads them after the call returns. The XDK's fences, busy checks
// and locks wait for that.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/diag.h"
#include "core/log.h"
#include "gpu/backend.h"
#include "gpu/simple_dirty.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"

namespace gpu {

namespace {

// commands handed over at once, so the render thread keeps close behind and a wait is short
constexpr uint32_t kBatchCommands = 64;

// resource headers the render thread reads: common, data, lock, format, size
constexpr uint32_t kHeaderBytes = 20, kHeaderSlot = 32;
constexpr uint32_t kHeaderRing = 1u << 20;  // guest memory for header copies; a copy is renewed after half a ring
constexpr uint32_t kPaletteRing = 2u << 20;  // contiguous guest memory for palette colour copies (their Data is physical)

enum class WaitFor : uint8_t { ReadBack, Resource, Fence, Flush, Copy, Frame, Count };

enum class Op : uint8_t { State, Draw, Clear, Copy, Present, Sync, RecordLate, Flush, Gamma, ClearReport, GetReport };

struct Cmd {
    Op op;
    uint32_t a = 0, b = 0, c = 0, d = 0;
    float f = 0, g = 0;
    uint32_t data = 0, size = 0;  // bytes in Batch::bytes
    int32_t vs = -1, ps = -1;     // index in Batch::vs / ps: the shader changed
    Surface s1, s2;
};

struct Batch {
    uint64_t id = 0;
    std::vector<Cmd> cmds;
    std::vector<uint8_t> bytes;
    std::vector<std::shared_ptr<VertexShader>> vs;
    std::vector<std::shared_ptr<PixelShader>> ps;
};

// the plain data in State, around its shader pointers and the resource headers (sent as copies); of the transforms only
// view, projection, texture and world are drawn with
struct Range {
    size_t off, len;
};
size_t At(const State& s, const void* p) { return size_t(static_cast<const uint8_t*>(p) - reinterpret_cast<const uint8_t*>(&s)); }
std::vector<Range> StateRanges() {
    static const State s;
    return {
        {At(s, &s.rs), At(s, &s.textures) - At(s, &s.rs)},  // simple goes by SendSimple
        {At(s, &s.index_buffer), At(s, &s.vs) - At(s, &s.index_buffer)},
        {At(s, &s.psc), At(s, &s.rt) - At(s, &s.psc)},  // vsc goes by SendConstants
        {At(s, &s.scissor_count), At(s, &s.transforms) - At(s, &s.scissor_count)},
        {At(s, &s.transforms[2]), 2 * sizeof(s.transforms[0])},
        {At(s, &s.transforms[16]), 4 * sizeof(s.transforms[0])},
        {At(s, &s.transforms[256]), sizeof(s.transforms[0])},
        {At(s, &s.material), sizeof(State) - At(s, &s.material)},
    };
}

// equal bytes, 16 at a time: most of the state is compared on every draw
bool Same(const uint8_t* a, const uint8_t* b, size_t n) { return !std::memcmp(a, b, n); }


class Threaded final : public Backend {
public:
    explicit Threaded(std::unique_ptr<Backend> inner) : inner_(std::move(inner)), ranges_(StateRanges()) {
        ring_ = mem::AllocSystem(kHeaderRing, 64);
        pal_ring_ = mem::AllocContiguous(kPaletteRing, 0, 0xFFFFFFFF, 64);
        cur_ = NewBatch();
        std::thread([this] { Run(); }).detach();
    }
    ~Threaded() override {
        {
            std::lock_guard lock(m_);
            stop_ = true;
        }
        cv_.notify_all();
        std::unique_lock lock(m_);
        // at exit the worker may be gone mid-batch: then the inner backend is left, not freed under it
        if (!done_cv_.wait_for(lock, std::chrono::milliseconds(500), [&] { return exited_; })) (void)inner_.release();
    }

    bool Init(uint32_t width, uint32_t height) override { return inner_->Init(width, height); }

    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override {
        SendState(s);
        Cmd c{Op::Draw};
        c.a = prim;
        c.b = start;
        c.c = count;
        if (indices) {
            c.d = 1;
            c.data = Put(indices, size_t(count) * 2);
        }
        cur_->cmds.push_back(c);
        // the draw's resources, each looked up only when its slot holds another one than at the batch's last draw
        const uint64_t id = cur_->id;
        auto use = [&](uint32_t slot, uint32_t header) {
            UsedSlot& u = used_slots_[slot];
            if (u.header == header && u.batch == id) return;
            u.header = header;
            u.batch = id;
            Use(header);
        };
        for (uint32_t i = 0; i < 4; i++) use(i, s.textures[i]);
        for (uint32_t i = 0; i < 4; i++) use(4 + i, s.palettes[i]);
        for (uint32_t i = 0; i < 16; i++) use(8 + i, s.streams[i].vb);
        use(24, s.rt.header);
        use(25, s.zs.header);
        if (s.rt.header) {
            const uint32_t data = mem::VirtualFromPhysical(s.rt.data);
            if (data != last_written_) {
                written_.insert(data);
                last_written_ = data;
            }
        }
        if (cur_->cmds.size() >= kBatchCommands) Submit();
    }

    void Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z, uint32_t stencil) override {
        SendState(s);
        Cmd c{Op::Clear};
        c.a = count;
        c.b = flags;
        c.c = color;
        c.d = stencil;
        c.f = z;
        if (count && rects) {
            c.data = Put(rects, size_t(count) * 16);
            c.size = count * 16;
        }
        cur_->cmds.push_back(c);
        Use(s.rt.header);
        Use(s.zs.header);
        if (flags & kClearTarget) Written(s.rt);
    }

    // rects and points are taken now; the copy itself, CPU or GPU, runs in order like the console's
    void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects, uint32_t dst_points,
              float src_kx, float src_ky) override {
        Cmd c{Op::Copy};
        c.a = count;
        c.f = src_kx;
        c.g = src_ky;
        c.b = count && src_rects ? Scratch(kernel::HostPtr(src_rects), count * 16) : 0;
        c.c = count && dst_points ? Scratch(kernel::HostPtr(dst_points), count * 8) : 0;
        c.s1 = src;
        c.s1.header = Header(src.header);
        c.s2 = dst;
        c.s2.header = Header(dst.header);
        cur_->cmds.push_back(c);
        Use(src.header);
        Use(dst.header);
        Written(dst);
    }

    void Present(const Surface& front) override {
        Cmd c{Op::Present};
        c.s1 = front;
        c.s1.header = Header(front.header);
        cur_->cmds.push_back(c);
        Use(front.header);
        uint64_t id = Submit();
        // the title records one frame while the one before it draws, on the GPU too: a GPU bound frame rate then paces
        // the game like a frame rate cap, and the frame times the game measures match the frames shown
        Wait(last_present_, WaitFor::Frame, nullptr);
        if (presents_) inner_->WaitGpuFrame(presents_ - 1);
        presents_++;
        all_constants_ = true;
        last_present_ = id;
    }

    void Sync(uint32_t data, uint32_t) override { ReadBack(data, false); }

    void SyncLate(uint32_t data) override {
        if (!inner_->DeliverLate(data)) {
            written_.insert(data);
            ReadBack(data, true);
        }
        Cmd c{Op::RecordLate};
        c.a = data;
        cur_->cmds.push_back(c);
    }

    void Flush() override {
        cur_->cmds.push_back(Cmd{Op::Flush});
        Wait(Submit(), WaitFor::Flush, "flush");
    }

    void SetGamma(const uint8_t* ramp) override {
        Cmd c{Op::Gamma};
        c.data = Put(ramp, 768);
        cur_->cmds.push_back(c);
    }

    void ClearReport() override { cur_->cmds.push_back(Cmd{Op::ClearReport}); }
    void GetReport(uint32_t report) override {
        Cmd c{Op::GetReport};
        c.a = report;
        cur_->cmds.push_back(c);
    }

    uint32_t Fence() override { return uint32_t(cur_->id); }
    bool FencePending(uint32_t fence) override { return fence > done_.load(); }
    void WaitFence(uint32_t fence) override {
        // NFS Most Wanted fences its whole frame and blocks on it right before it presents, which kept the title and
        // the render thread taking turns (each idle half the frame). Its buffers are locked through the library,
        // whose busy checks still wait for the draws that read them, so the frame before is enough here
        if (g_fence_ahead.load(std::memory_order_relaxed) && last_present_ && fence > last_present_) fence = uint32_t(last_present_);
        Wait(fence, WaitFor::Fence, "fence");
    }
    bool Busy(uint32_t resource) override {
        auto it = last_use_.find(resource);
        return it != last_use_.end() && it->second > done_.load();
    }
    void WaitResource(uint32_t resource) override {
        auto it = last_use_.find(resource);
        if (it != last_use_.end()) Wait(it->second, WaitFor::Resource, "resource");
    }

private:
    Batch* NewBatch() {
        Batch* b;
        {
            std::lock_guard lock(m_);
            if (free_.empty()) {
                b = new Batch;
            } else {
                b = free_.back();
                free_.pop_back();
            }
        }
        b->id = ++next_id_;
        return b;
    }

    uint32_t Put(const void* p, size_t n) {
        size_t at = (cur_->bytes.size() + 3) & ~size_t(3);
        cur_->bytes.resize(at + n);
        std::memcpy(cur_->bytes.data() + at, p, n);
        return uint32_t(at);
    }

    void ReadBack(uint32_t data, bool late) {
        // only surfaces the GPU wrote since the last read-back hold anything newer than guest memory
        if (data == last_written_) last_written_ = ~0u;
        if (!written_.erase(data)) return;
        Cmd c{Op::Sync};
        c.a = data;
        cur_->cmds.push_back(c);
        Wait(Submit(), WaitFor::ReadBack, "read-back");
    }

    // bytes the render thread reads through a guest address, in the header ring
    uint32_t Scratch(const void* p, uint32_t n) {
        uint32_t size = (n + kHeaderSlot - 1) & ~(kHeaderSlot - 1);
        if (ring_pos_ % kHeaderRing + size > kHeaderRing) ring_pos_ += kHeaderRing - ring_pos_ % kHeaderRing;
        uint32_t at = ring_ + uint32_t(ring_pos_ % kHeaderRing);
        ring_pos_ += size;
        std::memcpy(kernel::HostPtr(at), p, n);
        return at;
    }

    // A header as it is now. The console's push buffer takes a texture's format and address, and a surface's or vertex
    // buffer's address, when they are set; the title points the same header elsewhere before the GPU gets there.
    uint32_t Header(uint32_t header) {
        if (!header || !ring_) return header;
        // vertex buffer, index buffer and palette headers are 12 bytes: reading 20 can run off the end of the heap
        const uint32_t type = kernel::R32(header) & 0x70000;  // 0 vertex, 0x10000 index, 0x30000 palette
        uint8_t now[kHeaderBytes] = {};
        std::memcpy(now, kernel::HostPtr(header), type < 0x20000 || type == 0x30000 ? 12 : kHeaderBytes);
        // a palette's colours are copied as well, and the copy's header points at them: the title fills the same
        // palette memory again for the next frames while this one is still to be drawn (NFS Most Wanted's finish line
        // flare smoke was decoded with whatever the memory held by then: near black, and no red smoke)
        // A copy is made only when the colours differ from this palette's last copy, which is reused while it is
        // well inside the ring: copying on every state change wrapped the ring within a frame, and draws still queued
        // read other palettes' colours (the sky flickered to yellow noise)
        if (type == 0x30000 && pal_ring_) {
            uint32_t data;
            std::memcpy(&data, now + 4, 4);
            // compared once a frame: the same palette is set for hundreds of draws, and hashing its colours for each
            // was the biggest part of recording a draw (a change between frames is what the copies are for)
            PaletteCopy& pc = pal_copies_[header];
            if (pc.made && pc.frame == presents_ && pc.data == data && pal_written_ - pc.made <= kPaletteRing / 2 &&
                ring_pos_ - pc.header_made <= kHeaderRing / 2)
                return pc.header;
            const uint32_t bytes = (256u >> (kernel::R32(header) >> 30)) * 4;
            const uint8_t* src = static_cast<const uint8_t*>(kernel::HostPtr(mem::VirtualFromPhysical(data)));
            uint64_t h = 1469598103934665603ull ^ data;
            for (uint32_t i = 0; i < bytes; i += 8) {
                uint64_t w;
                std::memcpy(&w, src + i, 8);
                h = (h ^ w) * 1099511628211ull;
            }
            pc.frame = presents_;
            pc.data = data;
            if (!pc.made || pc.hash != h || pal_written_ - pc.made > kPaletteRing / 2 || ring_pos_ - pc.header_made > kHeaderRing / 2) {
                if (pal_pos_ + bytes > kPaletteRing) pal_pos_ = 0;
                const uint32_t at = pal_ring_ + pal_pos_;
                const uint32_t step = (bytes + 63) & ~63u;
                pal_pos_ += step;
                pal_written_ += step;
                std::memcpy(kernel::HostPtr(at), src, bytes);
                std::memcpy(now + 4, &(pc.phys = mem::PhysicalFromVirtual(at)), 4);
                pc.hash = h;
                pc.made = pal_written_;
                pc.header = Scratch(now, kHeaderBytes);
                pc.header_made = ring_pos_;
            }
            return pc.header;
        }
        Recent<HeaderCopy>& r = recent_headers_[(header >> 3) & 63];
        HeaderCopy& c = r.key == header ? *r.value : headers_[header];
        r = {header, &c};
        if (c.made && !std::memcmp(c.bytes, now, kHeaderBytes) && ring_pos_ - c.made < kHeaderRing / 2) return c.at;
        c.at = Scratch(now, kHeaderBytes);
        c.made = ring_pos_;
        std::memcpy(c.bytes, now, kHeaderBytes);
        return c.at;
    }

    // a field of the render thread's State, when it differs from what was sent
    void Delta(const void* field, const void* value, size_t n) {
        uint8_t* sent = reinterpret_cast<uint8_t*>(&sent_) + At(sent_, field);
        if (Same(sent, static_cast<const uint8_t*>(value), n)) return;
        std::memcpy(sent, value, n);
        uint32_t head[2] = {uint32_t(At(sent_, field)), uint32_t(n)};
        size_t at = cur_->bytes.size();
        cur_->bytes.resize(at + 8 + n);
        std::memcpy(cur_->bytes.data() + at, head, 8);
        std::memcpy(cur_->bytes.data() + at + 8, value, n);
    }

    void DeltaRange(const Range& r, const uint8_t* now) {
        // most ranges are as sent: one compare of the whole range, the pieces only when something changed
        if (Same(reinterpret_cast<const uint8_t*>(&sent_) + r.off, now + r.off, r.len)) return;
        for (size_t off = r.off, end = r.off + r.len; off < end; off += 64)
            Delta(reinterpret_cast<const uint8_t*>(&sent_) + off, now + off, std::min<size_t>(64, end - off));
    }

    // simple whole, or only the pieces the title marked when its Direct3D layer marks every write
    void SendSimple(const State& s) {
        static_assert(sizeof(State::simple) == sizeof(g_simple_dirty) * 8 * kSimplePiece * sizeof(uint32_t));
        if (!g_simple_tracked) {
            DeltaRange({At(s, &s.simple), sizeof(s.simple)}, reinterpret_cast<const uint8_t*>(&s));
            return;
        }
        for (size_t w = 0; w < g_simple_dirty.size(); w++) {
            for (uint64_t bits = g_simple_dirty[w]; bits; bits &= bits - 1) {
                size_t first = (w * 64 + size_t(std::countr_zero(bits))) * kSimplePiece;
                Delta(&sent_.simple[first], &s.simple[first], kSimplePiece * sizeof(uint32_t));
            }
            g_simple_dirty[w] = 0;
        }
    }

    // the vertex constants the title wrote since the last draw; all of them once a frame, for a write not marked
    void SendConstants(const State& s, const uint8_t* now) {
        if (all_constants_) {
            all_constants_ = false;
            g_vsc_dirty = {};
            DeltaRange({At(s, &s.vsc), sizeof(s.vsc)}, now);
            return;
        }
        for (size_t w = 0; w < g_vsc_dirty.size(); w++) {
            for (uint64_t bits = g_vsc_dirty[w]; bits; bits &= bits - 1) {
                const size_t r = w * 64 + size_t(std::countr_zero(bits));
                Delta(&sent_.vsc[r], &s.vsc[r], sizeof(s.vsc[r]));
            }
            g_vsc_dirty[w] = 0;
        }
    }

    // what changed since the last command, in 64 byte pieces
    void SendState(const State& s) {
        const uint8_t* now = reinterpret_cast<const uint8_t*>(&s);
        Cmd c{Op::State};
        size_t first = cur_->bytes.size();
        SendSimple(s);
        SendConstants(s, now);
        for (const Range& r : ranges_) DeltaRange(r, now);
        std::array<uint32_t, 4> textures;
        for (size_t i = 0; i < 4; i++) textures[i] = Header(s.textures[i]);
        Delta(&sent_.textures, &textures, sizeof(textures));
        std::array<uint32_t, 4> palettes;
        for (size_t i = 0; i < 4; i++) palettes[i] = Header(s.palettes[i]);
        Delta(&sent_.palettes, &palettes, sizeof(palettes));
        auto streams = s.streams;
        for (auto& st : streams) st.vb = Header(st.vb);
        Delta(&sent_.streams, &streams, sizeof(streams));
        Surface rt = s.rt, zs = s.zs;
        rt.header = Header(rt.header);
        zs.header = Header(zs.header);
        Delta(&sent_.rt, &rt, sizeof(rt));
        Delta(&sent_.zs, &zs, sizeof(zs));
        if (s.vs.get() != sent_.vs.get()) {
            sent_.vs = s.vs;
            c.vs = int32_t(cur_->vs.size());
            cur_->vs.push_back(s.vs);
        }
        if (s.ps.get() != sent_.ps.get()) {
            sent_.ps = s.ps;
            c.ps = int32_t(cur_->ps.size());
            cur_->ps.push_back(s.ps);
        }
        if (cur_->bytes.size() == first && c.vs < 0 && c.ps < 0) return;
        c.data = uint32_t(first);
        c.size = uint32_t(cur_->bytes.size() - first);
        cur_->cmds.push_back(c);
    }

    // resources by guest header: the batch that last read them
    void Use(uint32_t header) {
        if (!header) return;
        Recent<uint64_t>& r = recent_uses_[(header >> 3) & 63];
        if (r.key != header) r = {header, &last_use_[header]};
        *r.value = cur_->id;
    }
    void Written(const Surface& s) {
        if (!s.header) return;
        const uint32_t data = mem::VirtualFromPhysical(s.data);
        written_.insert(data);
        last_written_ = data;
    }

    uint64_t Submit() {
        uint64_t id = cur_->id;
        {
            std::lock_guard lock(m_);
            queue_.push_back(cur_);
        }
        cv_.notify_one();
        cur_ = NewBatch();
        return id;
    }

    void Wait(uint64_t id, WaitFor kind, const char* why) {
        if (!id || done_.load() >= id) return;
        if (id >= cur_->id) id = Submit();
        {
            std::unique_lock lock(m_);
            done_cv_.wait(lock, [&] { return done_.load() >= id; });
        }
    }

    void Run() {
        using SetDescription = HRESULT(WINAPI*)(HANDLE, PCWSTR);
        if (auto set = reinterpret_cast<SetDescription>(
                reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription"))))
            set(GetCurrentThread(), L"render");
        for (;;) {
            Batch* b;
            {
                std::unique_lock lock(m_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) break;
                b = queue_.front();
                queue_.pop_front();
            }
            const auto busy_from = std::chrono::steady_clock::now();
            for (const Cmd& c : b->cmds) {
                // a present waits for the GPU and the window inside: its time is not counted as work
                if (c.op == Op::Present) {
                    g_render_busy_ns.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - busy_from).count(),
                        std::memory_order_relaxed);
                    Execute(*b, c);
                    continue;
                }
                Execute(*b, c);
            }
            if (b->cmds.empty() || b->cmds.back().op != Op::Present)
                g_render_busy_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - busy_from).count(),
                    std::memory_order_relaxed);
            uint64_t id = b->id;
            b->cmds.clear();
            b->bytes.clear();
            b->vs.clear();
            b->ps.clear();
            {
                std::lock_guard lock(m_);
                free_.push_back(b);
                done_.store(id);
            }
            done_cv_.notify_all();
        }
        std::lock_guard lock(m_);
        exited_ = true;
        done_cv_.notify_all();
    }

    void Execute(const Batch& b, const Cmd& c) {
        const uint8_t* data = b.bytes.data() + c.data;
        switch (c.op) {
            case Op::State: {
                uint8_t* s = reinterpret_cast<uint8_t*>(&state_);
                static const uint32_t vsc_lo = uint32_t(At(state_, &state_.vsc)), vsc_hi = vsc_lo + uint32_t(sizeof(state_.vsc));
                for (const uint8_t* p = data; p < data + c.size;) {
                    uint32_t head[2];
                    std::memcpy(head, p, 8);
                    std::memcpy(s + head[0], p + 8, head[1]);
                    // a change to the vertex constants: the renderer uploads them again (and only then)
                    if (head[0] < vsc_hi && head[0] + head[1] > vsc_lo) state_.vsc_serial++;
                    p += 8 + head[1];
                }
                if (c.vs >= 0) state_.vs = b.vs[size_t(c.vs)];
                if (c.ps >= 0) state_.ps = b.ps[size_t(c.ps)];
                break;
            }
            case Op::Draw: inner_->Draw(state_, c.a, c.b, c.c, c.d ? data : nullptr); break;
            case Op::Clear: inner_->Clear(state_, c.a, c.size ? data : nullptr, c.b, c.c, c.f, c.d); break;
            case Op::Copy: inner_->Copy(c.s1, c.s2, c.a, c.b, c.c, c.f, c.g); break;
            case Op::Present: inner_->Present(c.s1); break;
            case Op::Sync: inner_->Sync(c.a, 0); break;
            case Op::RecordLate: inner_->RecordLate(c.a); break;
            case Op::Flush: inner_->Flush(); break;
            case Op::Gamma: inner_->SetGamma(data); break;
            case Op::ClearReport: inner_->ClearReport(); break;
            case Op::GetReport: inner_->GetReport(c.a); break;
        }
    }

    std::unique_ptr<Backend> inner_;
    std::vector<Range> ranges_;

    // the title's thread
    struct HeaderCopy {
        uint32_t at = 0;
        uint64_t made = 0;  // ring_pos_ after it was written
        uint8_t bytes[kHeaderBytes];
    };
    std::unordered_map<uint32_t, HeaderCopy> headers_;
    // the last few lookups, in front of the maps (their elements stay put)
    template <typename T>
    struct Recent {
        uint32_t key = 0;
        T* value = nullptr;
    };
    std::array<Recent<HeaderCopy>, 64> recent_headers_{};
    std::array<Recent<uint64_t>, 64> recent_uses_{};
    uint32_t ring_ = 0;
    uint32_t pal_ring_ = 0, pal_pos_ = 0;
    uint64_t pal_written_ = 0;
    struct PaletteCopy { uint64_t hash = 0, made = 0, header_made = 0, frame = ~0ull; uint32_t phys = 0, header = 0, data = 0; };
    std::unordered_map<uint32_t, PaletteCopy> pal_copies_;
    uint64_t ring_pos_ = 0;
    Batch* cur_ = nullptr;
    State sent_;  // the state as the render thread will have it
    std::unordered_map<uint32_t, uint64_t> last_use_;
    std::unordered_set<uint32_t> written_;  // surfaces drawn since their last read-back, by data address
    uint32_t last_written_ = ~0u;           // the last one put in written_ (and still there)
    struct UsedSlot {
        uint32_t header = 0;
        uint64_t batch = 0;
    };
    std::array<UsedSlot, 26> used_slots_{};  // Draw's resources by slot: what the batch's last draw marked
    uint64_t next_id_ = 0, last_present_ = 0;
    uint64_t presents_ = 0;
    bool all_constants_ = true;  // compare every vertex constant at the next draw (once a frame)

    // the render thread
    State state_;

    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    std::deque<Batch*> queue_;
    std::vector<Batch*> free_;
    std::atomic<uint64_t> done_{0};
    bool stop_ = false, exited_ = false;
};

}  // namespace

std::unique_ptr<Backend> CreateThreadedBackend(std::unique_ptr<Backend> inner) { return std::make_unique<Threaded>(std::move(inner)); }

}  // namespace gpu
