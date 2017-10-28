# farris-symmetry
This repo contains code for generating images with the 17 types of wallpaper symmetries utilizing the techniques described in the book "Creating Symmetry - The Artful Mathematics of Wallpaper Patterns" by Frank A. Farris.  To date, only wallpaper code has been implemented (i.e. no friezes, no color-reversing, etc.)





Here are some projects and extensions which I hope to eventually address:

1.  Using an ndarray/tensor framework like Tensorflow, MXNet, or PyTorch, this code could be speeded up significantly.
2.  It would be an interesting project to develop similar code for creating symmetric functions on the sphere for use in virtual reality (Gear VR, Rift, etc.)
3.  An even more ambitious VR-based project would be a visualization of functions invariant under the 3d space groups.  These functions would be functions on R^3 which we could approximate by colored voxel arrays, i.e. the 3d analogue of images.  But how to visualize these voxel arrays in VR?  One way would be to situate the viewer at the center of a translatable, rotatable, expandable/contractible sphere.  The sphere would be colored by the voxels which it intersects.
4.  Algorithmic detection of symmetry and "near" symmetry is a difficult problem. See the paper "Computational Symmetry in Computer
Vision and Computer Graphics" by Liu, et al.  We could use the code in this repo to generate training data for a deep learning symmetry detector/classifier.

