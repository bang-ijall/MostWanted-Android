# SDL's native code calls its Java classes and methods by name
-keep class org.libsdl.app.** { *; }
# the app's activities (named in the manifest) and anything the native side may reach
-keep class com.jall.ea.nfsmw.** { *; }
-dontwarn org.libsdl.app.**
