# Flow Key

Intelligent greenscreen software that adapts to uneven lighting

## Notes

- Cuda support is not fully implemented; OpenCL is the priority for cross-platform support
- Virtual camera support is not fully implemented
- GUI support for changing settings is not implemented
- Breaking up the pictures into blocks (so not working on a pixel-by-pixel basis) is only implemented for the `Splash` and `SplashPrep` filters
- I'm just going to keep rewriting this software until it's performant and easy to use. Fun!

## General Flow

Flow Key reads image data from a real device, applies a filter to the image, and then sends the modified frame to a
virtual camera. This application is meant to be used in tandem with other software that apply their own greenscreen
filters to the virtual camera feed; Flow Key simply replaces a selected color gradient from the image with a static
green color.

1. Read image frame from camera
2. (Async) Apply filter using GPU processing
    1. Send frame to GPU buffer
    2. GPU applies filter
    3. Display modified frame
    4. Send frame to virtual camera
3. Display original frame

## Filter Flow

There are several implemented filters:

- `InitialComparison`: Generic greenscreen filter; checks if pixels are close to a specified color key, then writes a replacement color to those pixels
- `NoiseReduction`: Identifies greenscreen pixels that aren't surrounded by many other greenscreen pixels, then writes the original color to those pixels
- `FlowKey`: Identifies pixels that are close to greenscreen pixels, then writes a replacement color to those pixels if they are close
- `Splash`: Compares an original image to a new image, then replaces stale pixels with a replacement color
- `SplashPrep`: Generates a block map of average pixel values

morphological operations

dilation (opening)

erosion (closing)

loop on GPU, not host
