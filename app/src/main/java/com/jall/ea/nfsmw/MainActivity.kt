package com.jall.ea.nfsmw

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import android.os.Bundle
import org.libsdl.app.SDLActivity

/** The game: SDL's activity running libGame.so with the imported files (Android/data/<package>/files/game). */
class MainActivity : SDLActivity() {
    companion object {
        // the on-screen controller's labels, drawn here with the system's font (touch_pad.cpp)
        private val LABELS = arrayOf("LT", "RT", "LB", "RB", "BACK", "START", "A", "B", "X", "Y")

        @JvmStatic external fun nativeLabel(text: String, w: Int, h: Int, alpha: ByteArray)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // no game files (opened some other way than the app's icon): the import screen first
        if (!ImportActivity.gameReady(this)) {
            startActivity(Intent(this, ImportActivity::class.java))
            finish()
            return
        }
        sendLabels()
    }

    // each label as white coverage, cut to its letters, large enough to stay sharp when scaled down
    private fun sendLabels() {
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 120f
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
        }
        val bounds = Rect()
        for (text in LABELS) {
            paint.getTextBounds(text, 0, text.length, bounds)
            val pad = 6
            val w = bounds.width() + 2 * pad
            val h = bounds.height() + 2 * pad
            val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            Canvas(bmp).drawText(text, (pad - bounds.left).toFloat(), (pad - bounds.top).toFloat(), paint)
            val px = IntArray(w * h)
            bmp.getPixels(px, 0, w, 0, 0, w, h)
            bmp.recycle()
            val alpha = ByteArray(w * h) { (px[it] ushr 24).toByte() }
            try {
                nativeLabel(text, w, h, alpha)
            } catch (_: UnsatisfiedLinkError) {
                return  // the library did not load: SDL shows its own message
            }
        }
    }

    override fun getArguments(): Array<String> = arrayOf(ImportActivity.gameDir(this).absolutePath)
    override fun getLibraries() = arrayOf("SDL3", "Game")
    override fun getMainSharedObject() = "libGame.so"
    override fun getMainFunction() = "SDL_main"
}
