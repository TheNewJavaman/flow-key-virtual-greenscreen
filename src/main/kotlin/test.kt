class test {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            val byteArray = ByteArray(size = 2_764_800)
            val floatArray = floatArrayOf(0.01f)
            println("${byteArray.size}, $floatArray")
        }
    }
}