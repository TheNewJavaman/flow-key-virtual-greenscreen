# Flow Key

Intelligent greenscreen software that adapts to uneven lighting

## Notes

- Cuda is currently the priority because it is easier to profile kernels. I will continue supporting OpenCL for greater
  platform compatibility, however
- Virtual camera support is not implemented
- GUI support for changing settings is only implemented for filters, not general settings
- `Splash` and `SplashPrep` filters are low priority and should not be used
- I'm just going to keep rewriting this software until it's performant and easy to use. Fun!

## Next Up

- Improve FPS counter and frame latency counter
- Implement virtual camera
    - Output modified frames to a virtual camera device so that other applications can use the video feed
- Use Cuda fatbins instead of cubins
    - Better multi-architecture GPU support
- Rewrite Cuda pipeline to use a pixel map instead of always operating on the entire image
    - Use boolean logic on a pixel map; instead of comparing an entire color to compare whether a pixel is greenscreen,
      use a binary color map so that there's only one read and one write operation per pixel instead of three each, one
      per color
    - Profiling is easier in Cuda than OpenCL, so it makes sense to first optimize the kernels on Cuda
- Rewrite entire OpenCL pipeline to match Cuda
- Save settings as a JSON file
- Add support for other hardware APIs
    - I've heard that Vulkan has a compute API. Maybe that would be better for AMD devices?

## General Flow

Flow Key reads image data from a real device, applies a custom set of filters to the image, and then sends the modified
frame to a virtual camera. This application is meant to be used in tandem with other software that apply their own
greenscreen filters to the virtual camera feed; Flow Key simply replaces a selected color gradient from an image with a
custom static color.

1. Read image frame from camera
2. (Sync, but should be async) Apply filter using GPU processing
    1. Send frame and settings to GPU buffer
    2. GPU applies filter
    3. Read output image and/or updated pixel map
3. (Async) Display modified frame
    1. Send frame to virtual camera (not implemented)
    2. Display original and modified frames in application

## Filter Flow

There are several implemented filters. The descriptions below are a planned expansion of the pipeline. Currently, each
filter operates on a full RGB image, but the plan is to move to binary "is this a greenscreen? y/n" bitmaps.

- `InitialComparison`: Generic greenscreen filter; checks if pixels are close to a specified color key, then writes a
  replacement color to those pixels
    - In: `original`: Original frame in RGB pixels
    - In: `colorKey`: Color to compare against
    - In: `tolerance`: Int in [0,255] to show the maximum difference that a greenscreen pixel can have from an original
      pixel
    - Out: `output`: Bitmap that identifies pixels as either 1/true/greenscreen or 0/false/original
- `NoiseReduction`: Identifies greenscreen pixels that aren't surrounded by many other greenscreen pixels, then writes
  the original color to those pixels
    - In: `input`: Input binary bitmap that shows where discovered greenscreen pixels are
    - In: `width`: Width of the image
    - In: `height`: Height of the image
    - Out: `output`: Binary bitmap that shows where the greenscreen pixels are
- `FlowKey`: Identifies pixels that are close to greenscreen pixels, then writes a replacement color to those pixels if
  they are close
    - In: `input`: Input binary bitmap
    - In: `original`: Original frame
    - In: `tolerance`: Int in [0,255] to show maximum difference
    - In: `width`: Width of the image
    - In: `height`: Height of the image
    - Out: `output`: Output binary bitmap
- `GapFiller`: Fills in the gaps between greenscreen pixels; the inverse of `NoiseReduction`
    - In: `input`: Input binary bitmap
    - In: `width`: Width of the image
    - In: `height`: Height of the image
    - Out: `output`: Output binary bitmap
- `ApplyBitmap`: Applies the results of the bitmap to the original image
    - In: `input`: Input binary bitmap
    - In: `original`: Original image
    - In: `replacementKey`: Greenscreen overlay color  
    - Out: `modified`: Modified image
- `Splash`: Compares an original image to a new image, then replaces stale pixels with a replacement color
    - WIP; will focus on this later
- `SplashPrep`: Generates a block map of average pixel values
    - WIP; will focus on this later