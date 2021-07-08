// Adapted from https://docs.opencv.org/3.4/d4/dbd/tutorial_filter_2d.html

package net.javaman.flowkey

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

class Filter constructor(
    private val kernelSize: Int = 10,
    private val dDepth: Int = -1,
    private val delta: Double = 0.0
) {
    fun apply(input: Mat): Mat {
        val ones = Mat.ones(kernelSize, kernelSize, CvType.CV_32F)
        val kernel = Mat()
        val output = Mat()
        val anchor = Point(-1.0, -1.0)
        Core.multiply(ones, Scalar(1 / (kernelSize * kernelSize).toDouble()), kernel)
        Imgproc.filter2D(input, output, dDepth, kernel, anchor, delta, Core.BORDER_DEFAULT)
        return output
    }
}