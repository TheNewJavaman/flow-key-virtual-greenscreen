# Flow Key

Intelligent greenscreen software that adapts to uneven lighting

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

## Filter flow

The filter works by first comparing pixels to a certain preselected color key, then moving outward from each valid pixel
and comparing adjacent pixels to the previous pixel. This "flow" outward from the original pixels is where this
application gets its name.

1. Search image for pixels matching a preselected color key
2. For each matching pixel, search the surrounding pixels for slightly different, but still matching, colors
   1. Repeat until no changes occur
3. Replace all the matching pixels with a static color