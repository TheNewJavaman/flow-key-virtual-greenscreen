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

- Implement virtual camera
    - Output modified frames to a virtual camera device so that other applications can use the video feed
- Do not build Cuda program at runtime
    - Generate fatbins (binaries built for multiple GPU architectures) instead of runtime ptx/cubin compilations
    - This will decrease startup time
    - Users won't need developer tools to be installed for the program to run
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

There are several implemented filters:

- `InitialComparison`: Generic greenscreen filter; checks if pixels are close to a specified color key, then writes a
  replacement color to those pixels
- `NoiseReduction`: Identifies greenscreen pixels that aren't surrounded by many other greenscreen pixels, then writes
  the original color to those pixels
- `FlowKey`: Identifies pixels that are close to greenscreen pixels, then writes a replacement color to those pixels if
  they are close
- `Splash`: Compares an original image to a new image, then replaces stale pixels with a replacement color
- `SplashPrep`: Generates a block map of average pixel values
