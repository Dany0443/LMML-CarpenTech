# Twins Separated at Birth

## Approach

The solution finds pairs of images that show the same scene, including pairs
where one image is mirrored.

1. Load images from a directory or archive. The loader also tries to recover
   images from damaged ZIP files.
2. Convert images to grayscale and resize only when needed.
3. Use perceptual DCT hashes, together with rotated and mirrored variants, to
   quickly generate a small set of possible matches.
4. Verify each candidate with SIFT feature matching and a ratio test.
5. Estimate an affine transformation with RANSAC and accept a pair only when
   it has enough inliers, a strong inlier ratio, good normalized correlation,
   and sufficient image coverage.
6. Join accepted matches into groups with a disjoint-set structure and output
   all valid image pairs as `output.csv`.

The verification stage keeps the fast hash search from producing false
matches, while the mirrored variants handle images that were flipped.
