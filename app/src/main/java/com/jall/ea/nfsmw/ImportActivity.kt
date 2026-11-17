package com.jall.ea.nfsmw

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.graphics.Outline
import android.graphics.Typeface
import android.graphics.drawable.ClipDrawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.LayerDrawable
import android.net.Uri
import android.os.Bundle
import android.os.StatFs
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.View
import android.view.ViewOutlineProvider
import android.view.WindowManager
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.security.MessageDigest

/**
 * The app's first screen. With the game's files in place it opens the game at once; otherwise the player picks their
 * own copy of NFS Most Wanted (Xbox) as an ISO (redump or XISO), which is read and copied into
 * Android/data/<package>/files/game.
 */
class ImportActivity : Activity() {
    companion object {
        // the default.xbe the recompiled code was made from: any other release would not run
        const val XBE_SHA1 = "f636a16af04171089f879a4bed646e185c734233"
        private const val PICK_ISO = 1
        private const val SECTOR = 2048L
        private val XISO_MAGIC = "MICROSOFT*XBOX*MEDIA".toByteArray()
        // where the game partition starts: XISO, then the redump layouts (XGD1, XGD2, XGD3)
        private val PARTITIONS = longArrayOf(0L, 0x18300000L, 0x0FD90000L, 0x02080000L)

        // the screen's colours
        private val BG = Color.rgb(13, 15, 19)
        private val CARD = Color.rgb(23, 26, 32)
        private val LINE = Color.rgb(38, 42, 50)
        private val TEXT = Color.rgb(240, 242, 245)
        private val MUTED = Color.rgb(140, 147, 158)
        private val ACCENT = Color.rgb(255, 153, 28)
        private val ERROR = Color.rgb(255, 107, 97)

        fun gameDir(a: Activity) = File(a.getExternalFilesDir(null) ?: a.filesDir, "game")

        fun sha1(f: File): String {
            val md = MessageDigest.getInstance("SHA-1")
            FileInputStream(f).use { i ->
                val b = ByteArray(1 shl 16)
                while (true) {
                    val n = i.read(b)
                    if (n < 0) break
                    md.update(b, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        fun gameReady(a: Activity): Boolean {
            val xbe = File(gameDir(a), "default.xbe")
            return xbe.isFile && File(gameDir(a), "NFS").isDirectory
        }
    }

    private class Stop(message: String) : Exception(message)

    private lateinit var button: TextView
    private lateinit var panel: LinearLayout
    private lateinit var status: TextView
    private lateinit var percent: TextView
    private lateinit var detail: TextView
    private lateinit var progress: ProgressBar
    @Volatile private var busy = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (gameReady(this)) {
            startGame()
            return
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.statusBarColor = BG
        window.navigationBarColor = BG
        setContentView(buildUi())
    }

    private fun startGame() {
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }

    // ---- screen ----

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun rounded(color: Int, radius: Int, stroke: Int = 0) = GradientDrawable().apply {
        cornerRadius = dp(radius).toFloat()
        setColor(color)
        if (stroke != 0) setStroke(dp(1), stroke)
    }

    private fun text(s: String, size: Float, color: Int, bold: Boolean = false) = TextView(this).apply {
        text = s
        textSize = size
        setTextColor(color)
        typeface = Typeface.create("sans-serif" + if (bold) "-medium" else "", Typeface.NORMAL)
        includeFontPadding = false
    }

    private fun LinearLayout.add(v: View, top: Int = 0, width: Int = LinearLayout.LayoutParams.MATCH_PARENT) =
        addView(v, LinearLayout.LayoutParams(width, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(top) })

    private fun buildUi(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(28), dp(56), dp(28), dp(32))
        }
        // the game's icon and name
        root.addView(ImageView(this).apply {
            setImageResource(R.mipmap.ic_launcher)
            clipToOutline = true
            outlineProvider = object : ViewOutlineProvider() {
                override fun getOutline(view: View, outline: Outline) = outline.setRoundRect(0, 0, view.width, view.height, dp(22).toFloat())
            }
        }, LinearLayout.LayoutParams(dp(88), dp(88)))
        root.add(text("NEED FOR SPEED", 12f, MUTED, true).apply { letterSpacing = 0.18f }, top = 28)
        root.add(text("Most Wanted", 30f, TEXT, true), top = 6)
        root.add(text("Impor salinan game Xbox Anda untuk mulai bermain.", 15f, MUTED).apply { setLineSpacing(0f, 1.25f) }, top = 12)

        // what is needed
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = rounded(CARD, 16, LINE)
            setPadding(dp(18), dp(16), dp(18), dp(16))
        }
        val needs = arrayOf(
            "File ISO NFS Most Wanted (Xbox), redump atau XISO",
            "Ruang kosong sekitar 3 GB",
            "File ISO boleh dihapus setelah impor selesai"
        )
        needs.forEachIndexed { i, s ->
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
            row.addView(text("${i + 1}", 12f, ACCENT, true).apply {
                gravity = Gravity.CENTER
                background = rounded(Color.argb(40, 255, 153, 28), 11)
            }, LinearLayout.LayoutParams(dp(22), dp(22)))
            row.addView(text(s, 14f, TEXT).apply { setLineSpacing(0f, 1.2f) },
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { leftMargin = dp(12) })
            card.add(row, top = if (i == 0) 0 else 12)
        }
        root.add(card, top = 28)

        // progress, shown while importing and for the outcome
        panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = rounded(CARD, 16, LINE)
            setPadding(dp(18), dp(16), dp(18), dp(18))
            visibility = View.GONE
        }
        val head = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        status = text("", 15f, TEXT, true)
        percent = text("", 15f, ACCENT, true)
        head.addView(status, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        head.addView(percent)
        panel.add(head)
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 1000
            progressDrawable = LayerDrawable(arrayOf(
                rounded(LINE, 3),
                ClipDrawable(rounded(ACCENT, 3), Gravity.START, ClipDrawable.HORIZONTAL)
            )).apply {
                setId(0, android.R.id.background)
                setId(1, android.R.id.progress)
            }
        }
        panel.addView(progress, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(6)).apply { topMargin = dp(14) })
        detail = text("", 13f, MUTED).apply { setLineSpacing(0f, 1.25f) }
        panel.add(detail, top = 12)
        root.add(panel, top = 16)

        // the one action, at the bottom
        root.addView(View(this), LinearLayout.LayoutParams(1, 0, 1f))
        button = text("Pilih file ISO", 16f, Color.rgb(20, 16, 10), true).apply {
            gravity = Gravity.CENTER
            background = rounded(ACCENT, 14)
            setPadding(0, dp(17), 0, dp(17))
            isClickable = true
            setOnClickListener { if (!busy) pick() }
        }
        root.add(button, top = 24)
        root.add(text("Data game tetap milik Anda dan hanya disimpan di HP ini.", 12f, MUTED).apply {
            gravity = Gravity.CENTER
        }, top = 14)

        return ScrollView(this).apply {
            setBackgroundColor(BG)
            isFillViewport = true
            addView(root)
        }
    }

    private fun pick() {
        val i = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            // some file managers label ISO files oddly: everything is offered, the content is checked
            type = "*/*"
        }
        try {
            startActivityForResult(i, PICK_ISO)
        } catch (e: Exception) {
            fail("Pemilih file tidak tersedia di HP ini.")
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data
        if (requestCode != PICK_ISO || resultCode != RESULT_OK || uri == null) return
        run(uri)
    }

    @Volatile private var lastShow = 0L
    // progress at most ten times a second (a copy reports every block)
    private fun show(s: String, d: String = "", frac: Double = -1.0) {
        val now = System.nanoTime()
        if (frac in 0.0..0.999 && now - lastShow < 100_000_000L) return
        lastShow = now
        runOnUiThread {
            panel.visibility = View.VISIBLE
            status.text = s
            status.setTextColor(TEXT)
            detail.text = d
            progress.visibility = View.VISIBLE
            if (frac >= 0) {
                progress.isIndeterminate = false
                progress.progress = (frac * 1000).toInt()
                percent.text = "${(frac * 100).toInt()}%"
            } else {
                percent.text = ""
            }
        }
    }

    private fun fail(why: String) = runOnUiThread {
        panel.visibility = View.VISIBLE
        status.text = "Impor gagal"
        status.setTextColor(ERROR)
        percent.text = ""
        detail.text = why
        progress.visibility = View.GONE
    }

    private fun setBusy(b: Boolean) = runOnUiThread {
        busy = b
        button.alpha = if (b) 0.35f else 1f
        button.text = if (b) "Mengimpor…" else "Pilih file ISO"
    }

    // ---- import ----

    private fun run(uri: Uri) {
        setBusy(true)
        Thread {
            val base = getExternalFilesDir(null) ?: filesDir
            val work = File(base, "import.tmp")
            try {
                work.deleteRecursively()
                work.mkdirs()
                importIso(uri, work)
                show("Memeriksa data game", "Memastikan versi game cocok")
                val found = findGameRoot(work) ?: throw Stop("default.xbe tidak ditemukan. Pastikan ini data game Xbox NFS Most Wanted.")
                if (!File(found, "NFS").isDirectory) throw Stop("Folder NFS tidak ditemukan di samping default.xbe. Data game tidak lengkap.")
                val hash = sha1(File(found, "default.xbe"))
                if (hash != XBE_SHA1)
                    throw Stop("Versi game tidak cocok.\ndefault.xbe ini (SHA-1 ${hash.take(12)}…) bukan versi yang didukung (${XBE_SHA1.take(12)}…).")
                val dest = gameDir(this)
                dest.deleteRecursively()
                if (!found.renameTo(dest)) throw Stop("Gagal memindahkan data ke ${dest.path}")
                work.deleteRecursively()
                show("Selesai", "Membuka game…", 1.0)
                runOnUiThread { startGame() }
            } catch (e: Stop) {
                work.deleteRecursively()
                fail(e.message ?: "")
                setBusy(false)
            } catch (e: Throwable) {
                work.deleteRecursively()
                fail("Terjadi kesalahan saat membaca ISO (${e.javaClass.simpleName}).")
                setBusy(false)
            }
        }.start()
    }

    private fun needSpace(bytes: Long, where: File) {
        val free = StatFs(where.path).availableBytes
        if (free < bytes + (200L shl 20))
            throw Stop("Ruang penyimpanan kurang: butuh ${bytes / (1 shl 20)} MB, tersedia ${free / (1 shl 20)} MB.")
    }

    private fun sizeOf(uri: Uri): Long {
        contentResolver.query(uri, arrayOf(OpenableColumns.SIZE, OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst() && !c.isNull(0)) return c.getLong(0)
        }
        return -1
    }

    private fun mb(b: Long) = "%.0f MB".format(b / 1048576.0)

    // the directory holding default.xbe (the folder itself, or one inside it)
    private fun findGameRoot(dir: File, depth: Int = 0): File? {
        if (File(dir, "default.xbe").isFile) return dir
        if (depth >= 4) return null
        dir.listFiles()?.filter { it.isDirectory }?.forEach { d -> findGameRoot(d, depth + 1)?.let { return it } }
        // a differently cased default.xbe
        dir.listFiles()?.firstOrNull { it.isFile && it.name.equals("default.xbe", true) }?.let {
            it.renameTo(File(dir, "default.xbe"))
            return dir
        }
        return null
    }

    // -- ISO (Xbox XDVDFS) --

    private fun importIso(uri: Uri, out: File) {
        val pfd = contentResolver.openFileDescriptor(uri, "r") ?: throw Stop("File ISO tidak bisa dibuka")
        pfd.use { FileInputStream(it.fileDescriptor).channel.use { ch -> extractIso(ch, out) } }
    }

    private fun readAt(ch: FileChannel, pos: Long, n: Int): ByteBuffer {
        val b = ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN)
        var at = pos
        while (b.hasRemaining()) {
            val r = ch.read(b, at)
            if (r <= 0) break
            at += r
        }
        b.flip()
        return b
    }

    private class IsoFile(val path: String, val sector: Long, val size: Long)

    private fun extractIso(ch: FileChannel, out: File) {
        show("Membaca ISO", "Mencari data game di dalam ISO")
        val part = PARTITIONS.firstOrNull { off ->
            off + 0x10000 + 20 <= ch.size() && readAt(ch, off + 0x10000, 20).let { b ->
                val a = ByteArray(20); b.get(a); a.contentEquals(XISO_MAGIC)
            }
        } ?: throw Stop("Ini bukan ISO Xbox (tidak ada tanda MICROSOFT*XBOX*MEDIA).")
        val head = readAt(ch, part + 0x10000 + 20, 8)
        val rootSector = head.int.toLong() and 0xFFFFFFFFL
        val rootSize = head.int.toLong() and 0xFFFFFFFFL
        val files = mutableListOf<IsoFile>()
        val dirs = mutableListOf<String>()
        listDir(ch, part, rootSector, rootSize, "", files, dirs, 0)
        if (files.isEmpty()) throw Stop("ISO kosong atau rusak.")
        val total = files.sumOf { it.size }
        needSpace(total, out)
        dirs.forEach { File(out, it).mkdirs() }
        var done = 0L
        val buf = ByteBuffer.allocateDirect(1 shl 20)
        for (f in files) {
            val dst = File(out, f.path)
            dst.parentFile?.mkdirs()
            FileOutputStream(dst).channel.use { w ->
                var pos = part + f.sector * SECTOR
                var left = f.size
                while (left > 0) {
                    buf.clear()
                    if (left < buf.capacity()) buf.limit(left.toInt())
                    val r = ch.read(buf, pos)
                    if (r <= 0) throw Stop("ISO terpotong: ${f.path}")
                    buf.flip()
                    while (buf.hasRemaining()) w.write(buf)
                    pos += r
                    left -= r
                    done += r
                    show("Menyalin data game", "${mb(done)} dari ${mb(total)}  ·  ${f.path.substringAfterLast('/')}", done.toDouble() / total)
                }
            }
        }
    }

    // a directory table: a binary tree of entries, each pointing left and right in dwords
    private fun listDir(ch: FileChannel, part: Long, sector: Long, size: Long, prefix: String,
                        files: MutableList<IsoFile>, dirs: MutableList<String>, depth: Int) {
        if (size <= 0 || depth > 32) return
        val t = readAt(ch, part + sector * SECTOR, size.toInt())
        val seen = HashSet<Int>()
        val stack = ArrayDeque<Int>()
        stack.addLast(0)
        while (stack.isNotEmpty()) {
            val at = stack.removeLast()
            if (at < 0 || at + 14 > t.limit() || !seen.add(at)) continue
            val left = t.getShort(at).toInt() and 0xFFFF
            val right = t.getShort(at + 2).toInt() and 0xFFFF
            if (left == 0xFFFF && right == 0xFFFF) continue
            val start = t.getInt(at + 4).toLong() and 0xFFFFFFFFL
            val len = t.getInt(at + 8).toLong() and 0xFFFFFFFFL
            val attr = t.get(at + 12).toInt() and 0xFF
            val nlen = t.get(at + 13).toInt() and 0xFF
            if (at + 14 + nlen > t.limit()) continue
            val nb = ByteArray(nlen)
            for (i in 0 until nlen) nb[i] = t.get(at + 14 + i)
            val name = String(nb, Charsets.ISO_8859_1)
            if (left != 0) stack.addLast(left * 4)
            if (right != 0) stack.addLast(right * 4)
            if (name.isEmpty() || name.contains('/') || name == "." || name == "..") continue
            val path = if (prefix.isEmpty()) name else "$prefix/$name"
            if (attr and 0x10 != 0) {
                dirs += path
                listDir(ch, part, start, len, path, files, dirs, depth + 1)
            } else {
                files += IsoFile(path, start, len)
            }
        }
    }
}
